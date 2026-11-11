# SPDX-License-Identifier: Apache-2.0
"""``python -m alloy.sysid <excite | fit | report | demo>``: system identification from the command line.

    python -m alloy.sysid excite --robot panda --out excite.csv
    python -m alloy.sysid fit --robot panda --data take.csv --out my_panda.sysid.json --report sysid_report
    python -m alloy.sysid report my_panda.sysid.json
    python -m alloy.sysid demo --robot panda --out take.csv        # a synthetic recording, to try the fit
"""

from __future__ import annotations

import argparse
import sys
from typing import Any


def _names(text: str | None) -> list[str] | None:
    return [x.strip() for x in text.split(",") if x.strip()] if text else None


def _parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(prog="python -m alloy.sysid", description="Fit a simulated robot's joint "
                                  "parameters to a recording of the real one (docs/guide/sysid.md).")
    sub = top.add_subparsers(dest="command", required=True)
    robot = {"required": True, "help": "a shipped robot's name (panda, go2, fr3, ...) or a robot file (MJCF, URDF, "
             "USD)"}
    joints = {"help": "comma-separated joint names (default: every position-servoed joint)"}
    device = {"default": None, "help": "cpu or cuda:0 (default: a GPU when there is one)"}

    p = sub.add_parser("excite", help="write a safe excitation trajectory to play on the real robot")
    p.add_argument("--robot", **robot)
    p.add_argument("--out", default="excite.csv")
    p.add_argument("--joints", **joints)
    p.add_argument("--rate", type=float, default=100.0, help="Hz (default 100)")
    p.add_argument("--seconds", type=float, default=8.0, help="seconds per joint (default 8)")
    p.add_argument("--amplitude", type=float, default=0.15, help="rad each way (default 0.15; slides 0.1 m per rad)")
    p.add_argument("--mode", choices=("sequential", "together"), default="sequential")

    p = sub.add_parser("fit", help="fit joint parameters to a recording")
    p.add_argument("--robot", **robot)
    p.add_argument("--data", required=True, help="recording: CSV, HDF5 or a LeRobot dataset folder")
    p.add_argument("--out", default="robot.sysid.json")
    p.add_argument("--fit", default=None, help="comma-separated parameters: kp,kd,damping,friction,delay "
                   "(the default) and armature,torque_limit; or gains, default, all")
    p.add_argument("--joints", **joints)
    p.add_argument("--worlds", type=int, default=64, help="candidates per generation (default 64)")
    p.add_argument("--generations", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--angle-unit", choices=("rad", "deg"), default="rad", help="unit of the recording's angles")
    p.add_argument("--time-unit", choices=("s", "ms"), default="s", help="unit of the recording's time")
    p.add_argument("--max-delay-ms", type=float, default=50.0)
    p.add_argument("--device", **device)
    p.add_argument("--report", default=None, metavar="DIR", help="also write report.txt and overlay plots there")
    p.add_argument("--robot-name", default="robot", help="the robot's name in your scene, for the randomize lines")
    p.add_argument("--column-names", default=None, help="LeRobot dataset: joint names of the action columns")
    p.add_argument("--state-columns", default=None, help="LeRobot dataset: which observation.state columns are "
                   "the joint positions, e.g. 0,1,2,3,4,5")

    p = sub.add_parser("report", help="print the report of a saved fit")
    p.add_argument("file")
    p.add_argument("--robot-name", default="robot", help="the robot's name in your scene, for the randomize lines")

    p = sub.add_parser("demo", help="write a synthetic recording (a simulated robot with hidden parameters)")
    p.add_argument("--robot", **robot)
    p.add_argument("--out", default="take.csv")
    p.add_argument("--joints", **joints)
    p.add_argument("--seconds", type=float, default=6.0)
    p.add_argument("--seed", type=int, default=0)
    return top


def _fit(args: Any) -> int:
    import alloy.sysid as S  # noqa: PLC0415
    from alloy.sysid.readers import load_recording  # noqa: PLC0415

    cols = [int(c) for c in _names(args.state_columns)] if args.state_columns else None
    rec = load_recording(args.data, _names(args.column_names), cols)
    result = S.fit(args.robot, rec, parameters=_names(args.fit), joints=_names(args.joints), worlds=args.worlds,
                   generations=args.generations, seed=args.seed, angle_unit=args.angle_unit,
                   time_unit=args.time_unit, max_delay=args.max_delay_ms / 1e3, device=args.device)
    path = result.save(args.out)
    print(S.report(result, args.report, args.robot_name))
    print(f"\nwrote {path}" + (f" and the report in {args.report}/" if args.report else ""))
    print(f'apply it: scene.add_robot("{args.robot_name}", <robot file>, sysid="{path}")')
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run one command; returns the exit code. Example: ``main(["report", "my_robot.sysid.json"])``."""
    from alloy.api.errors import AlloyAPIError  # noqa: PLC0415

    args = _parser().parse_args(argv)
    try:
        import alloy.sysid as S  # noqa: PLC0415

        if args.command == "fit":
            return _fit(args)
        if args.command == "report":
            print(S.report(args.file, robot_name=args.robot_name))
        elif args.command == "excite":
            exc = S.excite(args.robot, _names(args.joints), rate=args.rate, seconds=args.seconds,
                           amplitude=args.amplitude, mode=args.mode)
            path = exc.save(args.out)
            print(f"wrote {path}: {len(exc.joints)} joints, {exc.t[-1]:.1f} s at {args.rate:g} Hz. Bring the robot "
                  "to the first row's pose, play the cmd.* columns as joint targets at that rate, and log t, the "
                  "commanded targets and the measured joint positions (docs/guide/sysid.md).")
        else:
            rec, hidden = S.demo(args.robot, _names(args.joints), seconds=args.seconds, seed=args.seed)
            path = S.save_csv(rec, args.out)
            print(f"wrote {path}: a synthetic recording of {rec.joints} ({rec.duration:.1f} s). Hidden values a fit "
                  "should find: " + ", ".join(f"{k} {[round(float(x), 4) for x in v]}" for k, v in hidden.items()))
    except AlloyAPIError as exc:
        print(f"error: {exc.message}\n  fix: {exc.fix}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
