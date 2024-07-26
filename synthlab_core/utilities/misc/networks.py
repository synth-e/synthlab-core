import socket


def is_port_ready(port: int, host: str = "127.0.0.1", timeout=1000):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0
