#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Train the denoiser on packed teacher jobs (``run_jobs.py`` / ``pack.py`` output), resumable.

    python tools/denoiser/train.py --packed /data/packed --eval /data/packed_eval --out /data/ckpt --name v1 \\
        [--init default] [--enc 96,128,160,192,256] [--steps 150000] [--batch 64] [--lr 3e-4] \\
        [--pool 12000 --pool-large 300 --frac-large 0.4] [--seedmix 0.3] [--resume]

Batches mix whole 128x128 frames with 128x128 crops of the large (640x480) frames (``--frac-large``), drawn from
GPU frame pools that are refreshed from disk as training runs (``dataset.GpuPool``). The loss is display-space
L1 + LPIPS + log-L1 (``alloy.camera.denoiser.training``). Every ``--val-every`` steps the held-out eval jobs are
scored per (job, spp) against Cycles' own OIDN on the same samples. Writes ``OUT/NAME.pt`` (loadable by
``alloy.camera.denoiser.load``), ``OUT/NAME.pt.train`` (optimiser state) and ``OUT/train_NAME.jsonl`` (rows).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import GpuPool, PackedIndex

from alloy.camera.denoiser.infer import read_checkpoint
from alloy.camera.denoiser.model import NetConfig, UNet, make_input, to_linear
from alloy.camera.denoiser.training import Batch, Loss, TrainConfig, train
from alloy.camera.denoiser.weights import resolve_weights

log = logging.getLogger("denoiser.train")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--packed", required=True, help="comma list of packed-job folders (training)")
    p.add_argument("--eval", default=None, help="comma list of packed eval-job folders")
    p.add_argument("--out", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--init", default=None, help="start from these weights (path, name or 'default')")
    p.add_argument("--enc", default="96,128,160,192,256")
    p.add_argument("--out-mode", dest="out_mode", default="kpnd")
    p.add_argument("--demod", action="store_true",
                   help="KPN kernels filter irradiance (rgb / albedo) and multiply the albedo back; also on --init weights")
    p.add_argument("--steps", type=int, default=150000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--w-lpips", dest="w_lpips", type=float, default=0.3)
    p.add_argument("--w-log", dest="w_log", type=float, default=0.1)
    p.add_argument("--exp-stops", dest="exp_stops", type=float, default=1.5)
    p.add_argument("--seedmix", type=float, default=0.3)
    p.add_argument("--pool", type=int, default=12000, help="128x128 frames held on the GPU")
    p.add_argument("--pool-large", dest="pool_large", type=int, default=300, help="large frames held on the GPU")
    p.add_argument("--large-res", dest="large_res", default="640x480")
    p.add_argument("--frac-large", dest="frac_large", type=float, default=0.4)
    p.add_argument("--refresh-every", dest="refresh_every", type=int, default=50)
    p.add_argument("--val-every", dest="val_every", type=int, default=5000)
    p.add_argument("--rescan-every", dest="rescan_every", type=int, default=2000,
                   help="add newly packed jobs every this many steps (0: never)")
    p.add_argument("--ckpt-every", dest="ckpt_every", type=int, default=2500)
    p.add_argument("--split", default="train", help="job split to train on (per-scene sets use 'finetune')")
    p.add_argument("--exclude", default=None, help="leave out training jobs whose id contains this")
    p.add_argument("--spp-weights", dest="spp_weights", default="",
                   help="input sampling weight by spp, e.g. 16:3,32:4,64:3 (others 1)")
    p.add_argument("--job-weights", dest="job_weights", default="",
                   help="sampling weight multipliers by job-id substring, e.g. c2v2=3,_v000=2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    return p.parse_args(argv)


def build_net(a: argparse.Namespace, ckpt: Path) -> UNet:
    """A fresh network, the --init weights, or (with --resume) the run's own last checkpoint."""
    src = ckpt if a.resume and ckpt.exists() else (resolve_weights(a.init) if a.init else None)
    if src is None:
        enc = [int(c) for c in a.enc.split(",")]
        return UNet(NetConfig(enc=enc, unshuffle=2, out_mode=a.out_mode, demod=a.demod))
    ck = read_checkpoint(src)
    net = UNet(NetConfig(**{**ck["cfg"], **({"demod": True} if a.demod else {})}))
    net.load_state_dict(ck["state"])
    log.info("weights from %s", src)
    return net


def make_sampler(pools: list[tuple[GpuPool, float]], a: argparse.Namespace, index: PackedIndex, loss: Loss):
    """Batches split across pools by share; pools refresh every --refresh-every calls, and every --rescan-every
    calls new packed jobs on disk join the index (the loss gets their display LUTs)."""
    calls = [0]

    def draw(b: int, g: torch.Generator) -> Batch:
        calls[0] += 1
        if a.rescan_every and calls[0] % a.rescan_every == 0 and index.update():
            loss.display = index.display("cuda")
            log.info("index: %s frames", {r: index.frames(r) for r in index.by_res})
        if calls[0] % a.refresh_every == 0:
            for pool, _ in pools:
                pool.refresh(max(8, pool.cap // 200))  # grows the pool first, then replaces
        sizes = [round(b * s) for _, s in pools]
        sizes[0] += b - sum(sizes)
        parts = [p.sample(n, 128, g, a.seedmix) for (p, _), n in zip(pools, sizes) if n > 0]
        return Batch(*(torch.cat([getattr(x, f) for x in parts]) for f in Batch.__dataclass_fields__))
    return draw


@torch.no_grad()
def validate(net: UNet, index: PackedIndex, lp) -> dict:
    """Display L1 / LPIPS per '<res>:<job>|<spp>' for the network and Cycles' OIDN on the same samples."""
    net.eval()
    disp, acc = index.display("cuda"), {}
    for res, jobs in index.by_res.items():
        for job in jobs:
            _validate_job(net, job, res, disp, lp, acc)
    net.train()
    return {k: [round(v[0] / v[2], 5), round(v[1] / v[2], 5)] for k, v in sorted(acc.items())}


def _validate_job(net, job, res: str, disp, lp, acc: dict) -> None:
    a = job.arrays
    name = job.path.parent.name
    for f in range(job.frames):
        ref = torch.from_numpy(a["ref"][f:f + 1].astype("float32")).cuda()
        idx = torch.tensor([job.lut], device="cuda")
        dr = disp.apply(ref, idx)
        for k, (_, spp) in enumerate(job.keys):
            x = [torch.from_numpy(a[n][f:f + 1, k].astype("float32")).cuda() for n in ("noisy", "albedo", "normal")]
            inp = make_input(*x, torch.tensor([float(spp)], device="cuda"))
            pad = [(-s) % 32 for s in inp.shape[-2:]]
            inp = torch.nn.functional.pad(inp, (0, pad[1], 0, pad[0]), mode="replicate")
            with torch.autocast("cuda", dtype=torch.float16):
                out = net(inp.contiguous(memory_format=torch.channels_last))
            pred = to_linear(out[..., :ref.shape[1], :ref.shape[2]], x[1])
            cands = {"ours": pred}
            if "oidn" in a:
                cands["oidn"] = torch.from_numpy(a["oidn"][f:f + 1, k].astype("float32")).cuda()
            for m, y in cands.items():
                dy = disp.apply(y, idx)
                lpv = float(lp(dy.permute(0, 3, 1, 2) * 2 - 1, dr.permute(0, 3, 1, 2) * 2 - 1)) if lp else 0.0
                v = acc.setdefault(f"{res}:{name}|{spp}|{m}", [0.0, 0.0, 0])
                v[0] += float((dy - dr).abs().mean())
                v[1] += lpv
                v[2] += 1


def raise_fd_limit() -> None:
    """Allow as many open files as the hard limit (memory-mapped arrays of thousands of jobs)."""
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < hard:
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    raise_fd_limit()
    a = parse_args(argv)
    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = True
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / f"{a.name}.pt"
    rules = {k: float(v) for k, v in (kv.split("=") for kv in a.job_weights.split(",") if kv)}
    index = PackedIndex.scan(a.packed.split(","), split=a.split, exclude=a.exclude, weight_rules=rules)
    sw = {int(k): float(v) for k, v in (kv.split(":") for kv in a.spp_weights.split(",") if kv)}
    pools = [(GpuPool(index, "128x128", a.pool, seed=a.seed, spp_weights=sw), 1 - a.frac_large)]
    if a.frac_large > 0 and a.large_res in index.by_res:
        pools.append((GpuPool(index, a.large_res, a.pool_large, seed=a.seed + 1, spp_weights=sw), a.frac_large))
    evi = PackedIndex.scan(a.eval.split(","), split=None) if a.eval else None
    net = build_net(a, ckpt).cuda()
    cfg = TrainConfig(steps=a.steps, batch=a.batch, lr=a.lr, w_lpips=a.w_lpips, w_log=a.w_log,
                      exp_stops=a.exp_stops, ckpt=str(ckpt), ckpt_every=a.ckpt_every, extra={"name": a.name})
    loss = Loss(index.display("cuda"), a.w_lpips, a.w_log, device="cuda")
    rows = (out / f"train_{a.name}.jsonl").open("a")
    rows.write(json.dumps({"event": "start", **vars(a), "frames": {r: index.frames(r) for r in index.by_res},
                           "keys": index.keys, "params": sum(p.numel() for p in net.parameters())}) + "\n")

    def on_log(row: dict) -> None:
        if evi is not None and row["step"] % a.val_every == 0:
            row = {**row, "val": validate(net, evi, loss.lp)}
        rows.write(json.dumps(row) + "\n")
        rows.flush()

    g = torch.Generator(device="cuda").manual_seed(a.seed)
    train(net, make_sampler(pools, a, index, loss), loss, cfg, g, on_log=on_log, resume=a.resume)
    if evi is not None:
        rows.write(json.dumps({"event": "final", "val": validate(net, evi, loss.lp)}) + "\n")
    rows.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
