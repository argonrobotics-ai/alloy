# SPDX-License-Identifier: Apache-2.0
"""U-Net denoiser network for 1-4 spp path-traced frames (Cycles denoising-pass conventions).

    from alloy.camera.denoiser.model import NetConfig, UNet
    net = UNet(NetConfig(enc=[8, 8, 8, 8, 8], unshuffle=2, out_mode="kpnd"))

Network input (10 channels): log1p(rgb), albedo, normal, log2(spp) / 2. Output: log1p of the denoised linear
RGB (or 7 channels in "mix" mode). Optional space-to-depth stem (``unshuffle=2``): the U-Net then runs at
H/2 x W/2 with 4x the channels. Use :class:`alloy.camera.denoiser.Denoiser` for the user-facing API.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ._torch import F, nn, torch

IN_CH = 10
TAPS = 25  # 5x5 kernels
DEMOD_EPS = 0.02
KPN_MODES = ("kpn", "kpnd")
OUT_MODES = ("direct", "mix", "kpn", "kpnd")


@dataclass
class NetConfig:
    """Network hyper-parameters. Field names and defaults match the released checkpoints.

    Example: ``NetConfig(enc=[8, 8, 8, 8, 8], unshuffle=2, out_mode="kpnd")``.
    """

    enc: list[int] = field(default_factory=lambda: [32, 48, 64, 80, 96])  # channels per level (level 0 = stem)
    unshuffle: int = 1      # 1: level 0 at full res; 2: space-to-depth stem
    dec_extra: bool = True  # second conv in each decoder block
    enc_deep: list[int] = field(default_factory=list)  # extra 3x3 conv + ReLU per encoder level
    dec_deep: list[int] = field(default_factory=list)  # ... per decoder level (identity-initialised)
    demod: bool = False     # KPN kernels filter irradiance = rgb / (albedo + 0.02); result is multiplied back
    out_mode: str = "direct"  # "direct": log1p RGB; "mix": direct blended with albedo x irradiance;
    #                           "kpn": 3-scale 5x5 kernel prediction on the noisy linear RGB;
    #                           "kpnd": kpn blended per pixel with a direct prediction (starts at 95 % kpn)

    def validate(self) -> None:
        """Raise ValueError naming the field to change if the configuration cannot build a network."""
        if len(self.enc) < 2:
            raise ValueError(f"NetConfig.enc needs at least 2 levels, got {self.enc}; pass e.g. enc=[32, 48, 64].")
        if self.unshuffle not in (1, 2):
            raise ValueError(f"NetConfig.unshuffle must be 1 or 2, got {self.unshuffle}.")
        if self.out_mode not in OUT_MODES:
            raise ValueError(f"NetConfig.out_mode must be one of {OUT_MODES}, got {self.out_mode!r}.")
        if self.out_mode in KPN_MODES and self.unshuffle != 2:
            raise ValueError(f"out_mode={self.out_mode!r} needs the space-to-depth stem; set unshuffle=2.")


def conv(cin: int, cout: int) -> nn.Conv2d:
    """3x3 convolution with padding 1."""
    return nn.Conv2d(cin, cout, 3, padding=1)


def deep_block(c: int, n: int) -> nn.Sequential:
    """n extra 3x3 conv + ReLU at c channels, initialised to the identity (centre tap 1, bias 0).

    Their input is a ReLU output (>= 0), so a fresh block leaves the network's function unchanged.
    Example: ``deep_block(8, 2)``.
    """
    mods: list[nn.Module] = []
    for _ in range(n):
        cv = conv(c, c)
        with torch.no_grad():
            cv.weight.zero_()
            cv.bias.zero_()
            cv.weight[torch.arange(c), torch.arange(c), 1, 1] = 1.0
        mods += [cv, nn.ReLU(inplace=True)]
    return nn.Sequential(*mods)


def _decoder_block(e: list[int], i: int, cin: int, dec_extra: bool) -> nn.Sequential:
    """Decoder level i: input = upsampled previous level + skip e[i] (+ network input at level 0)."""
    c = e[i] if i else max(e[0], 32)
    skip = e[i] + (cin if i == 0 else 0)
    mods: list[nn.Module] = [conv(e[i + 1] + skip, c), nn.ReLU(inplace=True)]
    if dec_extra:
        mods += [conv(c, c), nn.ReLU(inplace=True)]
    return nn.Sequential(*mods)


class UNet(nn.Module):
    """U-Net with optional space-to-depth stem and direct / mix / KPN output heads.

    Example: ``UNet(NetConfig(enc=[8, 8, 8], unshuffle=1))(torch.zeros(1, 10, 8, 8))``.
    """

    def __init__(self, cfg: NetConfig) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        u, e = cfg.unshuffle, cfg.enc
        cin = IN_CH * u * u
        self.stem = nn.Sequential(conv(cin, e[0]), nn.ReLU(inplace=True), conv(e[0], e[0]), nn.ReLU(inplace=True))
        self.down = nn.ModuleList(
            nn.Sequential(conv(e[i], e[i + 1]), nn.ReLU(inplace=True)) for i in range(len(e) - 2))
        self.bottom = nn.Sequential(conv(e[-2], e[-1]), nn.ReLU(inplace=True), conv(e[-1], e[-1]),
                                    nn.ReLU(inplace=True))
        self.ups = nn.ModuleList(_decoder_block(e, i, cin, cfg.dec_extra) for i in range(len(e) - 2, -1, -1))
        c0 = max(e[0], 32)
        ed = list(cfg.enc_deep) + [0] * (len(e) - len(cfg.enc_deep))
        dd = list(cfg.dec_deep) + [0] * (len(e) - 1 - len(cfg.dec_deep))
        self.enc_x = nn.ModuleList(deep_block(e[i], ed[i]) for i in range(len(e))) if any(ed) else None
        self.dec_x = (nn.ModuleList(deep_block(e[i] if i else c0, dd[i]) for i in range(len(e) - 1))
                      if any(dd) else None)
        if cfg.out_mode in KPN_MODES:
            self.kpn = KPNHeads(c0, e[1], u, direct=cfg.out_mode == "kpnd")
        else:
            self.head = conv(c0, (7 if cfg.out_mode == "mix" else 3) * u * u)

    def trunk(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Return (level-0 decoder features, level-1 decoder features, the (unshuffled) input)."""
        u = self.cfg.unshuffle
        if u > 1:
            x = F.pixel_unshuffle(x, u)
        h = self._enc_extra(0, self.stem(x))
        feats = [h]
        for j, d in enumerate(self.down):
            h = self._enc_extra(j + 1, d(F.max_pool2d(h, 2)))
            feats.append(h)
        h = self._enc_extra(len(feats), self.bottom(F.max_pool2d(h, 2)))
        h1 = None
        for k, up in enumerate(self.ups):
            i = len(feats) - 1 - k
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            cat = [h, feats[i]] + ([x] if i == 0 else [])
            h = up(torch.cat(cat, 1))
            if self.dec_x is not None:
                h = self.dec_x[i](h)
            if i == 1:
                h1 = h
        return h, h1, x

    def _enc_extra(self, i: int, t: torch.Tensor) -> torch.Tensor:
        return self.enc_x[i](t) if self.enc_x is not None else t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N,10,H,W) network input -> (N,3|7,H,W) log1p-space output; H, W multiples of the stride."""
        u = self.cfg.unshuffle
        h, h1, x = self.trunk(x)
        if self.cfg.out_mode in KPN_MODES:
            full = x if u == 1 else F.pixel_shuffle(x, u)
            rgb = torch.expm1(full[:, :3])
            if self.cfg.demod:
                mod = full[:, 3:6].float().clamp_min(0) + DEMOD_EPS
                return self.kpn(h, h1, rgb.float() / mod, mod)
            return self.kpn(h, h1, rgb)
        out = self.head(h)
        return F.pixel_shuffle(out, u) if u > 1 else out


def apply_kernel(logits: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
    """Apply per-pixel 5x5 softmax kernels (N,25,H,W) to img (N,3,H,W), replicate border.

    Example: ``apply_kernel(torch.zeros(1, 25, 4, 4), torch.ones(1, 3, 4, 4))`` is all ones.
    """
    n, _, h, w = img.shape
    wgt = torch.softmax(logits.float(), 1).unsqueeze(1)                       # (N,1,25,H,W)
    pat = F.unfold(F.pad(img, (2, 2, 2, 2), mode="replicate"), 5).view(n, 3, TAPS, h, w)
    return (pat * wgt).sum(2)


class KPNHeads(nn.Module):
    """Multi-scale kernel prediction (after Vogels et al. 2018).

    5x5 kernels at full, 1/2 and 1/4 resolution on the noisy linear RGB pyramid (box down-sampling), combined
    coarse to fine with learned per-pixel weights: C = F_fine - a * U(D(F_fine)) + a * U(C_coarse). Returns
    log1p(output) so :func:`to_linear` applies unchanged. Example: ``KPNHeads(32, 8, 2, direct=True)``.
    """

    def __init__(self, c0: int, c1: int, u: int, direct: bool = False) -> None:
        super().__init__()
        if u != 2:
            raise ValueError(f"KPN output modes need the space-to-depth stem; set NetConfig.unshuffle=2 (got {u}).")
        self.u = u
        self.hd = conv(c0, 4 * u * u) if direct else None  # direct log1p RGB (3) + blend logit (1), full res
        self.h0 = conv(c0, (TAPS + 1) * u * u)   # full-res kernels + alpha0 (via pixel shuffle)
        self.h1 = conv(c0, TAPS + 1)             # 1/2-res kernels + alpha1
        self.h2 = conv(c1, TAPS)                 # 1/4-res kernels

    def forward(self, f0: torch.Tensor, f1: torch.Tensor, rgb: torch.Tensor,
                mod: torch.Tensor | None = None) -> torch.Tensor:
        """Filter rgb with the predicted kernels. With ``mod`` (demod mode) the result is multiplied by it."""
        k0 = F.pixel_shuffle(self.h0(f0), self.u)
        k1, k2 = self.h1(f0), self.h2(f1)
        hd = F.pixel_shuffle(self.hd(f0), self.u) if self.hd is not None else None
        with torch.autocast(rgb.device.type, enabled=False):
            c0 = self._combine(k0, k1, k2, rgb.float().clamp_min(0))
            if mod is not None:
                c0 = c0 * mod
            if hd is not None:
                d = hd.float()
                w = torch.sigmoid(d[:, 3:4] + 3.0)
                c0 = w * c0 + (1 - w) * torch.expm1(d[:, :3].clamp(0, 12))
            return torch.log1p(c0.clamp_min(0))

    @staticmethod
    def _combine(k0: torch.Tensor, k1: torch.Tensor, k2: torch.Tensor, l0: torch.Tensor) -> torch.Tensor:
        l1 = F.avg_pool2d(l0, 2)
        l2 = F.avg_pool2d(l1, 2)

        def up(t: torch.Tensor) -> torch.Tensor:
            return F.interpolate(t, scale_factor=2, mode="nearest")

        c2 = apply_kernel(k2[:, :TAPS], l2)
        f1_ = apply_kernel(k1[:, :TAPS], l1)
        a1 = torch.sigmoid(k1[:, TAPS:].float())
        c1 = f1_ - a1 * up(F.avg_pool2d(f1_, 2)) + a1 * up(c2)
        f0_ = apply_kernel(k0[:, :TAPS], l0)
        a0 = torch.sigmoid(k0[:, TAPS:].float())
        return f0_ - a0 * up(F.avg_pool2d(f0_, 2)) + a0 * up(c1)


def make_input(rgb: torch.Tensor, albedo: torch.Tensor, normal: torch.Tensor,
               spp: float | torch.Tensor) -> torch.Tensor:
    """Build the (N,10,H,W) float network input from (N,H,W,3) rgb / albedo / normal and spp.

    spp is a number, an (N,) tensor, or an (N,H,W) per-pixel tensor. Example:
    ``make_input(rgb, albedo, normal, 4).shape == (N, 10, H, W)``.
    """
    n, h, w, _ = rgb.shape
    if not torch.is_tensor(spp):
        spp = torch.full((n,), float(spp), device=rgb.device)
    if spp.dim() == 3:  # per-pixel spp (N, H, W), e.g. adaptive sampling
        code = (torch.log2(spp.float()) / 2).reshape(n, h, w, 1)
    else:
        code = (torch.log2(spp.float()) / 2).view(n, 1, 1, 1).expand(n, h, w, 1)
    x = torch.cat([torch.log1p(rgb.clamp_min(0)), albedo, normal, code], -1)
    return x.permute(0, 3, 1, 2)


def to_linear(out: torch.Tensor, albedo: torch.Tensor | None = None) -> torch.Tensor:
    """Convert network output (N,3|7,H,W) to linear RGB (N,H,W,3), >= 0.

    7 channels ("mix"): direct log1p RGB (3), log1p irradiance (3) multiplied by (albedo + 0.02), and a blend
    logit (1); albedo is then required. Example: ``to_linear(torch.zeros(1, 3, 4, 4))`` is all zeros.
    """
    out = out.float()
    direct = torch.expm1(out[:, :3].clamp(0, 12)).permute(0, 2, 3, 1)
    if out.shape[1] == 3:
        return direct
    if albedo is None:
        raise ValueError("to_linear: a 7-channel ('mix') output needs the albedo; pass albedo=(N,H,W,3).")
    irr = torch.expm1(out[:, 3:6].clamp(0, 12)).permute(0, 2, 3, 1)
    w = torch.sigmoid(out[:, 6:7]).permute(0, 2, 3, 1)
    return w * direct + (1 - w) * (albedo.float().clamp(0, 1) + 0.02) * irr
