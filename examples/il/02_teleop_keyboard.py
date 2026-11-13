# SPDX-License-Identifier: Apache-2.0
"""Drives a gripper by keyboard (or a key script) and records the episodes you keep as an HDF5 file.

Needs h5py: pip install h5py
Run:  python 02_teleop_keyboard.py                 in a terminal: a/d close/open both fingers, w/s lift/lower,
                                                   enter = keep the episode, backspace = discard, esc = quit
      python 02_teleop_keyboard.py --script demo   plays keys unattended (no terminal needed): closes the fingers
                                                   for 25 steps, then lifts while squeezing, then enter. Verified:
                                                   the cube rises to about 0.23 m, the task counts it a success
                                                   from step 60, and one 105-step episode is saved.
      python 02_teleop_keyboard.py --script "a*30 w*20 enter"   your own script (key*steps, .*n idles)
Alloy has no live teleop window in this example: add --preview look.png to write an image of the scene every
--preview-every steps and open that file to see the gripper (rendering is slow on a CPU, so it is off by default).
Prints the key table, one status line every 10 steps, then episodes saved and discarded and the file written
(<out>/teleop.hdf5). An episode still running when a script ends or you press esc is discarded, and the example
says so.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import h5py  # noqa: F401  (pip install h5py)

import alloy.envs
from alloy.api.errors import AlloyAPIError
from alloy.il import HDF5Writer, Recorder, SpaceMouseTeleop
from alloy.il.teleop import KeyboardTeleop, ScriptKeys, TerminalKeys, teleop_loop

DEMO_SCRIPT = "a*25 " + "a w " * 40 + "enter"      # a alternates with w so the squeeze is held while lifting
LINK = {"gripper/close_left": ["gripper/close_right"]}       # one key closes both fingers


def make_device(args: argparse.Namespace, env: Any) -> Any:
    """The keyboard (reading the terminal, or the --script keys), or the SpaceMouse with --spacemouse."""
    if args.spacemouse:                                # the terminal's keys add discard and quit (two buttons only)
        keys = TerminalKeys() if sys.stdin.isatty() else None
        return SpaceMouseTeleop(env.action_names(), axes={"z": ("gripper/lift", 1.0)}, link=LINK, keys=keys,
                                buttons={"left": ("gripper/close_left", 1.0), "right": ("gripper/close_left", -1.0)})
    keys = ScriptKeys.parse(DEMO_SCRIPT if args.script == "demo" else args.script) if args.script else None
    return KeyboardTeleop(env.action_names(), link=LINK, dt=env.sim.dt, keys=keys)


def status_printer(env: Any, args: argparse.Namespace) -> Any:
    """An on_step callback: a status line every 10 steps, and a preview image every --preview-every steps."""
    def on_step(index: int, out: Any) -> None:
        if args.preview and index % args.preview_every == 0:
            env.sim.render(width=320, height=240, path=args.preview)
        if index % 10:
            return
        height = float(env.xp.to_numpy(env.state["cube.pos"])[0, 2])
        success = bool(env.xp.to_numpy(out[4]["success"])[0])
        print(f"\rstep {index:4d}  cube height {height:5.3f} m  success {success}   ", end="", flush=True)
    return on_step


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default="alloy_demos", help="output directory (default ./alloy_demos)")
    p.add_argument("--device", default=None, help="cpu or cuda:0 (default: automatic)")
    p.add_argument("--script", default=None, help='keys to play: "demo" (grasp and lift) or e.g. "a*30 w*20 enter"')
    p.add_argument("--preview", default=None, metavar="PATH", help="write a picture of the scene to PATH; each picture takes seconds on a CPU")
    p.add_argument("--preview-every", type=int, default=25, help="steps between pictures (default 25)")
    p.add_argument("--spacemouse", action="store_true", help="use a SpaceMouse: z axis lifts, left/right button "
                   "closes/opens both fingers (not tested on hardware)")
    return p.parse_args()


def main(args: argparse.Namespace) -> int:
    if not (args.script or args.spacemouse or sys.stdin.isatty()):
        print("No terminal to read keys from. Run this in a terminal, or play keys from a script:")
        print('  python 02_teleop_keyboard.py --script demo')
        return 0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "teleop.hdf5"
    # 30 s per attempt: an attempt that runs into max_steps without enter is dropped, not saved
    env = alloy.envs.make("GripperLift-v0", num_envs=1, device=args.device, max_steps=3000)
    rec = Recorder(env, [HDF5Writer(path, overwrite=True)], task="lift the cube and hold it", keep="all")
    device = make_device(args, env)
    print(device.help())
    if not args.spacemouse:
        print("fixed keys: enter keep episode, backspace discard, esc quit, space zero all")
    rec.reset(seed=0)
    stats = teleop_loop(rec, device, realtime=not args.script, on_step=status_printer(env, args))
    unfinished = int(env.xp.to_numpy(env.state["steps"])[0])
    rec.close()
    print(f"\n{stats['steps']} steps, {rec.saved} episodes saved, {stats['discarded']} discarded")
    if unfinished:
        print(f"the episode still running ({unfinished} steps) was not finished with enter, so it was discarded")
    print(f"file: {path.resolve()}")
    env.close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(parse_args()))
    except AlloyAPIError as e:
        print(f"error: {e.message}\nfix: {e.fix}")
        sys.exit(1)
