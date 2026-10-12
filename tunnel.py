"""Local SSH port-forward to the Video-3D-LLM WebUI (paramiko, no sshpass needed).

This script mirrors the remote
port onto http://127.0.0.1:<local-port> so you can just open a browser.

    python tunnel.py                 # local 8099 -> remote 8099
    python tunnel.py 18099           # local 18099 -> remote 8099

Keep this window open while using the WebUI, then press Ctrl+C.
"""

from __future__ import annotations

import argparse
import getpass
import os
import select
import socket
import socketserver
import sys
import threading
import time

import paramiko

REMOTE_HOST, REMOTE_PORT = "127.0.0.1", 8099


def make_handler(transport: paramiko.Transport, remote: tuple):
    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            try:
                chan = transport.open_channel("direct-tcpip", remote, self.request.getpeername())
            except Exception as exc:
                print(f"[tunnel] open_channel failed: {exc}", file=sys.stderr)
                return
            if chan is None:
                return
            try:
                while True:
                    r, _, _ = select.select([self.request, chan], [], [], 1.0)
                    if self.request in r:
                        data = self.request.recv(4096)
                        if not data:
                            break
                        chan.sendall(data)
                    if chan in r:
                        data = chan.recv(4096)
                        if not data:
                            break
                        self.request.sendall(data)
            except Exception:
                pass
            finally:
                try:
                    chan.close()
                except Exception:
                    pass

    return Handler


class ThreadedServer(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("local_port", nargs="?", type=int, default=int(REMOTE_PORT))
    ap.add_argument("--remote-host", default=REMOTE_HOST)
    ap.add_argument("--remote-port", type=int, default=REMOTE_PORT)
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--ssh-host", default=os.getenv("SSH_HOST"))
    ap.add_argument("--ssh-port", type=int, default=int(os.getenv("SSH_PORT", "22")))
    ap.add_argument("--ssh-user", default=os.getenv("SSH_USER", getpass.getuser()))
    ap.add_argument("--password", action="store_true", help="Prompt securely for an SSH password; otherwise use SSH keys/agent")
    args = ap.parse_args()
    if not args.ssh_host: ap.error("Provide --ssh-host or SSH_HOST")
    password = getpass.getpass("SSH password: ") if args.password else os.getenv("SSH_PASSWORD")

    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect(hostname=args.ssh_host, port=args.ssh_port, username=args.ssh_user, password=password,
                   timeout=20, look_for_keys=True, allow_agent=True)
    transport = client.get_transport()
    transport.set_keepalive(30)
    print(f"[tunnel] {args.ssh_user}@{args.ssh_host}:{args.ssh_port} connected")
    print(f"[tunnel] http://{args.bind}:{args.local_port}  ->  {args.remote_host}:{args.remote_port}")
    print("[tunnel] keep this window open; Ctrl+C to stop")

    server = ThreadedServer((args.bind, args.local_port),
                            make_handler(transport, (args.remote_host, args.remote_port)))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[tunnel] closing")
    finally:
        server.server_close()
        client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
