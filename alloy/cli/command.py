# SPDX-License-Identifier: Apache-2.0
"""The ``alloy`` command: every operation with JSON out on stdout (logs on stderr), plus batch, call, ops and mcp.

Exit codes: 0 the call ran and passed; 1 it ran but the result reports a failure (a check with errors, INVALID
worlds); 2 it could not run (the JSON says what to change). Examples::

    alloy list scenes --match peg
    alloy scene '{"objects": [{"add": "ground"}, {"add": "box", "name": "cube", "size": [0.1, 0.1, 0.1]}]}'
    alloy step demo_peg_d12 --steps 50 --names peg_d12 --device cpu
    alloy check demo_peg_d12 --quick
    alloy render demo_peg_d12 --out peg.png --width 320 --height 240
    echo '[{"op": "build", "scene": "demo_peg_d12"}, {"op": "step", "steps": 20}]' | alloy batch -
    alloy mcp                                  # the same operations as an MCP tool server on stdio
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import IO, Any

from alloy.cli.errors import RequestError, error_payload
from alloy.cli.jsonio import dumps, json_stdout, loads_request
from alloy.cli.registry import OPS, passed, respond
from alloy.cli.session import Session
from alloy.cli.spec import add_arguments, describe, flags_to_args

EXAMPLES = __doc__.split("Examples::", 1)[1]


class _Parser(argparse.ArgumentParser):
    """Argument errors become a ``RequestError`` (and so a JSON reply), not argparse's text and exit."""

    def error(self, message: str) -> None:  # type: ignore[override]
        raise RequestError(f"{self.prog}: {message}", f"see `{self.prog} --help`", field="argv")


def version() -> str:
    """The installed Alloy version. Example: ``version()`` -> ``"0.1.0"``."""
    from importlib import metadata  # noqa: PLC0415

    for dist in ("alloy-engine", "alloy"):
        try:
            return metadata.version(dist)
        except metadata.PackageNotFoundError:
            continue
    import alloy  # noqa: PLC0415 - a checkout on the path, not installed

    return str(getattr(alloy, "__version__", "0+unknown"))


def _common(default: object) -> argparse.ArgumentParser:
    common = _Parser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true", default=default, help="info logs on stderr")
    common.add_argument("--pretty", action="store_true", default=default, help="indented JSON")
    return common


def build_parser() -> argparse.ArgumentParser:
    """The ``alloy`` argument parser (one sub-command per operation). Example: ``build_parser().parse_args([...])``."""
    p = _Parser(prog="alloy", description="Alloy physics for agents: JSON in, JSON out.", parents=[_common(False)],
                epilog="examples:" + EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = _common(argparse.SUPPRESS)          # after the command too, without resetting a flag given before it
    p.add_argument("--version", action="version", version=f"alloy {version()}")
    sub = p.add_subparsers(dest="command", required=True, metavar="COMMAND")
    for op in OPS.values():
        sp = sub.add_parser(op.name, help=op.summary, description=op.description, parents=[common])
        add_arguments(sp, op)
        sp.add_argument("--json", dest="json_args", metavar="JSON", help="all arguments as one JSON object ('-' "
                        "reads stdin, @file reads a file); flags override its keys")
    b = sub.add_parser("batch", parents=[common], help="run a list of requests in one process (simulations stay open)")
    b.add_argument("source", help="a JSON array or JSON lines of {\"op\": ..., ...}: a file, or - for stdin")
    b.add_argument("--keep-going", action="store_true", help="run the rest after a request fails")
    c = sub.add_parser("call", parents=[common], help="run one request {\"op\": ..., ...}")
    c.add_argument("request", help="the request JSON, - for stdin, or @file")
    o = sub.add_parser("ops", parents=[common], help="every operation with its JSON schema and an example")
    o.add_argument("name", nargs="?", help="only this operation")
    sub.add_parser("mcp", parents=[common], help="serve the operations as an MCP tool server on stdio")
    return p


def main(argv: Sequence[str] | None = None, stdout: IO[str] | None = None) -> int:
    """Entry point of the ``alloy`` console script. ``stdout``: where the JSON goes (default: the real stdout, with
    everything else that prints redirected to stderr). Example: ``main(["list", "scenes"])``."""
    try:
        args = build_parser().parse_args(argv)
    except RequestError as exc:
        return _emit({"ok": False, "op": None, "error": error_payload(exc)}, stdout, False)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, stream=sys.stderr,
                        format="%(levelname)s %(name)s: %(message)s")
    if args.command == "mcp":
        from alloy.mcp.server import serve_stdio  # noqa: PLC0415

        return serve_stdio()
    if stdout is not None:
        return _run(args, stdout)
    with json_stdout() as out:
        return _run(args, out)


def _run(args: argparse.Namespace, out: IO[str]) -> int:
    session = Session()
    try:
        reply = _reply(args, session)
    finally:
        session.close_all()
    return _emit(reply, out, args.pretty)


def _reply(args: argparse.Namespace, session: Session) -> dict[str, Any]:
    if args.command == "ops":
        ops = [OPS[args.name]] if args.name in OPS else list(OPS.values())
        if args.name and args.name not in OPS:
            return {"ok": False, "op": "ops", "error": error_payload(RequestError(
                f"unknown operation {args.name!r}", f"use one of {sorted(OPS)}", field="name"))}
        return {"ok": True, "op": "ops", "result": {"ops": describe(ops)}}
    if args.command == "batch":
        return run_batch(session, args.source, args.keep_going)
    try:
        request = _read_json(args.request, "request") if args.command == "call" else _op_request(args)
    except RequestError as exc:
        return {"ok": False, "op": args.command, "error": error_payload(exc)}
    return respond(session, request)


def _op_request(args: argparse.Namespace) -> dict[str, Any]:
    op = OPS[args.command]
    base = _read_json(args.json_args, "--json") if args.json_args else {}
    if not isinstance(base, dict):
        raise RequestError("--json must be a JSON object", '--json \'{"steps": 10}\'', field="json")
    return {**base, **flags_to_args(op, args), "op": op.name}


def _read_json(text: str, source: str) -> Any:
    if text == "-":
        text = sys.stdin.read()
    elif text.startswith("@"):
        path = Path(text[1:]).expanduser()
        if not path.is_file():
            raise RequestError(f"{source}: no file {str(path)!r}", "pass @<existing file> or the JSON itself",
                               field=source)
        text = path.read_text(encoding="utf-8")
    return loads_request(text, source)


def read_requests(text: str) -> list[Any]:
    """Requests from a JSON array, one JSON object, or JSON lines. Example: ``read_requests('{"op": "list"}')``."""
    stripped = text.strip()
    if not stripped:
        raise RequestError("no requests given", '[{"op": "build", "scene": "..."}, {"op": "step", "steps": 10}]',
                           field="source")
    if stripped[0] in "[{" and "\n{" not in stripped:
        data = loads_request(stripped, "batch")
        return data if isinstance(data, list) else [data]
    return [loads_request(line, f"batch line {i + 1}") for i, line in enumerate(stripped.splitlines())
            if line.strip()]


def run_batch(session: Session, source: str, keep_going: bool = False) -> dict[str, Any]:
    """Run requests in order in one session (a build stays open for the next steps); stop at the first failed call
    unless ``keep_going``. Example: ``run_batch(Session(), "requests.json")``."""
    try:
        if source != "-" and not Path(source).is_file():
            raise RequestError(f"batch: no file {source!r}", "pass a JSON file, or - to read stdin", field="source")
        requests = read_requests(sys.stdin.read() if source == "-" else Path(source).read_text(encoding="utf-8"))
    except RequestError as exc:
        return {"ok": False, "op": "batch", "error": error_payload(exc)}
    replies, stopped = [], None
    for i, request in enumerate(requests):
        replies.append(respond(session, request))
        if not replies[-1]["ok"] and not keep_going:
            stopped = i
            break
    ok = all(r["ok"] for r in replies)
    result = {"replies": replies, "ran": len(replies), "of": len(requests), "stopped_at": stopped,
              "passed": all(passed(r) for r in replies)}
    return {"ok": ok, "op": "batch", "result": result}


def _emit(reply: dict[str, Any], out: IO[str] | None, pretty: bool) -> int:
    text = dumps(reply, pretty=pretty) + "\n"
    if out is None:
        with json_stdout() as real:
            real.write(text)
    else:
        out.write(text)
    if reply.get("op") == "batch" and reply.get("ok"):
        return 0 if reply["result"]["passed"] else 1
    return 0 if passed(reply) else (1 if reply.get("ok") else 2)
