import os
from dataclasses import dataclass

from quip_proto import miner_pb2


class MissingToken(Exception):
    pass


@dataclass
class SessionConfig:
    miner_id: str
    queue_depth: int
    idle_timeout_s: int
    heartbeat_s: int
    reconnect_window_s: int


def build_hello(miner_id, backend, algorithm, supported_kinds):
    token = os.environ.get("QUIP_SESSION_TOKEN")
    if not token:
        raise MissingToken("QUIP_SESSION_TOKEN unset")
    return miner_pb2.Hello(
        miner_id=miner_id,
        session_token=token,
        protocol_version=1,
        backend=backend,
        algorithm=algorithm,
        supported_kinds=list(supported_kinds),
    )


def session_config_from_configure(miner_id, configure):
    def d(v, default):
        return default if v == 0 else v
    return SessionConfig(
        miner_id=miner_id,
        queue_depth=d(configure.queue_depth, 3),
        idle_timeout_s=d(configure.idle_timeout_s, 300),
        heartbeat_s=d(configure.heartbeat_s, 15),
        reconnect_window_s=d(configure.reconnect_window_s, 60),
    )
