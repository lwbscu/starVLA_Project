#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
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


def _read_cmdline(pid: str) -> str:
    proc_dir = Path("/proc") / pid
    try:
        raw = (proc_dir / "cmdline").read_bytes()
    except OSError:
        raw = b""
    cmdline = raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
    if cmdline:
        return cmdline
    try:
        return (proc_dir / "comm").read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return "<unreadable command>"


def describe_ports(ports: set[int]) -> int:
    inode_to_port = _read_proc_tcp(ports)
    if not inode_to_port:
        print(f"No /proc TCP listener found for ports: {', '.join(map(str, sorted(ports)))}")
        return 1

    rows: list[tuple[int, str, str, str]] = []
    for pid in sorted((p for p in os.listdir("/proc") if p.isdigit()), key=int):
        fd_dir = Path("/proc") / pid / "fd"
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
        for port, state, proc_file in sorted(matches):
            rows.append((port, state, pid, proc_file))

    if not rows:
        print(
            "Port appears in /proc/net/tcp, but no readable owning process was found. "
            "It may belong to another user namespace or may have exited."
        )
        return 1

    for port, state, pid, proc_file in rows:
        print(f"port={port} state={state} pid={pid} source={proc_file} cmd={_read_cmdline(pid)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Describe Linux processes using TCP ports.")
    parser.add_argument("ports", nargs="+", type=int)
    args = parser.parse_args()
    ports = set(args.ports)
    invalid = [port for port in ports if port < 1 or port > 65535]
    if invalid:
        parser.error(f"ports must be in [1, 65535], got: {invalid}")
    return describe_ports(ports)


if __name__ == "__main__":
    raise SystemExit(main())
