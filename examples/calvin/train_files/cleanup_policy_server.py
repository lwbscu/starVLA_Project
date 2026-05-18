#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path


TCP_STATES = {
    "01": "ESTABLISHED",
    "02": "SYN_SENT",
    "03": "SYN_RECV",
    "04": "FIN_WAIT1",
    "05": "FIN_WAIT2",
    "06": "TIME_WAIT",
    "07": "CLOSE",
    "08": "CLOSE_WAIT",
    "09": "LAST_ACK",
    "0A": "LISTEN",
    "0B": "CLOSING",
    "0C": "NEW_SYN_RECV",
}


@dataclass(frozen=True)
class PortOwner:
    port: int
    state: str
    pid: int
    proc_file: str
    argv: tuple[str, ...]

    @property
    def cmdline(self) -> str:
        return " ".join(self.argv) if self.argv else "<unreadable command>"


def _read_proc_tcp(ports: set[int]) -> dict[str, tuple[int, str, str]]:
    inode_to_port: dict[str, tuple[int, str, str]] = {}
    for proc_file in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = Path(proc_file).read_text(encoding="utf-8").splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) < 10:
                continue
            local_addr = parts[1]
            state = parts[3]
            inode = parts[9]
            try:
                port = int(local_addr.rsplit(":", 1)[1], 16)
            except (IndexError, ValueError):
                continue
            if port in ports:
                inode_to_port[inode] = (port, TCP_STATES.get(state, state), proc_file)
    return inode_to_port


def _read_argv(pid: int) -> tuple[str, ...]:
    proc_dir = Path("/proc") / str(pid)
    try:
        raw = (proc_dir / "cmdline").read_bytes()
    except OSError:
        raw = b""
    argv = tuple(
        part.decode("utf-8", errors="replace")
        for part in raw.split(b"\0")
        if part
    )
    if argv:
        return argv
    try:
        comm = (proc_dir / "comm").read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        comm = "<unreadable command>"
    return (comm,)


def _owners_for_ports(ports: set[int]) -> list[PortOwner]:
    inode_to_port = _read_proc_tcp(ports)
    owners: list[PortOwner] = []
    if not inode_to_port:
        return owners

    for pid_text in sorted((p for p in os.listdir("/proc") if p.isdigit()), key=int):
        pid = int(pid_text)
        fd_dir = Path("/proc") / pid_text / "fd"
        try:
            fds = list(fd_dir.iterdir())
        except OSError:
            continue
        matches: set[tuple[int, str, str]] = set()
        for fd in fds:
            try:
                target = os.readlink(fd)
            except OSError:
                continue
            if not (target.startswith("socket:[") and target.endswith("]")):
                continue
            inode = target[len("socket:[") : -1]
            if inode in inode_to_port:
                matches.add(inode_to_port[inode])
        if not matches:
            continue
        argv = _read_argv(pid)
        for port, state, proc_file in sorted(matches):
            owners.append(PortOwner(port=port, state=state, pid=pid, proc_file=proc_file, argv=argv))
    return owners


def _arg_value(argv: tuple[str, ...], flag: str) -> str | None:
    for idx, arg in enumerate(argv):
        if arg == flag and idx + 1 < len(argv):
            return argv[idx + 1]
        prefix = f"{flag}="
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return None


def _matches_policy_server_command(owner: PortOwner, ckpt_path: str | None) -> bool:
    script_match = any(arg.endswith("deployment/model_server/server_policy.py") for arg in owner.argv)
    if not script_match:
        return False
    port_arg = _arg_value(owner.argv, "--port")
    if port_arg != str(owner.port):
        return False
    if ckpt_path is None:
        return True
    ckpt_arg = _arg_value(owner.argv, "--ckpt_path")
    if ckpt_arg is None:
        return False
    expected = {ckpt_path}
    try:
        expected.add(str(Path(ckpt_path).resolve()))
    except OSError:
        pass
    return ckpt_arg in expected


def _matches_policy_server(owner: PortOwner, ckpt_path: str | None) -> bool:
    return owner.state == "LISTEN" and _matches_policy_server_command(owner, ckpt_path)


def _processes_matching_command(port: int, ckpt_path: str | None) -> list[PortOwner]:
    matches: list[PortOwner] = []
    for pid_text in sorted((p for p in os.listdir("/proc") if p.isdigit()), key=int):
        pid = int(pid_text)
        argv = _read_argv(pid)
        owner = PortOwner(port=port, state="CMDLINE", pid=pid, proc_file="/proc/cmdline", argv=argv)
        if _matches_policy_server_command(owner, ckpt_path):
            matches.append(owner)
    return matches


def _is_same_process(pid: int, port: int, ckpt_path: str | None) -> bool:
    argv = _read_argv(pid)
    owner = PortOwner(port=port, state="CMDLINE", pid=pid, proc_file="/proc/cmdline", argv=argv)
    return _matches_policy_server_command(owner, ckpt_path)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def cleanup_policy_server(port: int, ckpt_path: str | None, timeout: float, dry_run: bool) -> int:
    owners = _owners_for_ports({port})
    matching_by_pid: dict[int, PortOwner] = {}
    for owner in owners:
        if _matches_policy_server(owner, ckpt_path):
            matching_by_pid[owner.pid] = owner
    for owner in _processes_matching_command(port, ckpt_path):
        if _matches_policy_server_command(owner, ckpt_path):
            matching_by_pid.setdefault(owner.pid, owner)
    matching = list(matching_by_pid.values())
    if not matching:
        if owners:
            print(f"No matching policy server found on port {port}; refusing to kill non-matching owners.")
            for owner in owners:
                print(f"port={owner.port} state={owner.state} pid={owner.pid} source={owner.proc_file} cmd={owner.cmdline}")
            return 2
        print(f"No policy server listener found on port {port}.")
        return 0

    pids = sorted({owner.pid for owner in matching})
    for owner in matching:
        print(f"matched policy server: port={owner.port} state={owner.state} pid={owner.pid} cmd={owner.cmdline}")

    if dry_run:
        print("dry-run: no process was signaled")
        return 0

    for pid in pids:
        print(f"sending SIGTERM to policy server pid={pid}")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        alive = [pid for pid in pids if _is_same_process(pid, port, ckpt_path)]
        if not alive:
            print(f"policy server port {port} released after SIGTERM")
            return 0
        time.sleep(0.2)

    failed: list[int] = []
    for pid in pids:
        if not _is_same_process(pid, port, ckpt_path):
            continue
        print(f"sending SIGKILL to still-listening policy server pid={pid}")
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        failed.append(pid)

    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        alive = [pid for pid in failed if _is_same_process(pid, port, ckpt_path)]
        if not alive:
            print(f"policy server port {port} released after SIGKILL")
            return 0
        time.sleep(0.2)

    alive = [pid for pid in failed if _pid_alive(pid)]
    print(f"policy server cleanup failed for port {port}; alive pids={alive}")
    return 3


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stop only the StarVLA policy server that owns a specific TCP port."
    )
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--ckpt-path", default=None)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.port < 1 or args.port > 65535:
        parser.error(f"--port must be in [1, 65535], got {args.port}")
    if args.timeout < 0:
        parser.error(f"--timeout must be non-negative, got {args.timeout}")
    return cleanup_policy_server(args.port, args.ckpt_path, args.timeout, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
