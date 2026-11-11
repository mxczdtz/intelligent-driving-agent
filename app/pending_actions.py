
import secrets
import threading
import time

EXPIRE_SECONDS = 300
SESSION_EXPIRE_SECONDS = 1800
SESSION_COOKIE = "vehicle_session"

pending_actions = {}
active_sessions = {}
pending_lock = threading.Lock()


class SessionMismatchError(Exception):
    """确认令牌不属于当前会话。"""
    pass


def get_or_create_session(candidate: str | None) -> str:
    """复用有效的服务端会话，或生成一个新会话。"""

    now = time.monotonic()

    with pending_lock:
        # 清理过期会话
        for key, expires_at in list(active_sessions.items()):
            if expires_at <= now:
                del active_sessions[key]

        # 只认可服务端已经签发的会话
        if candidate and candidate in active_sessions:
            active_sessions[candidate] = (
                now + SESSION_EXPIRE_SECONDS
            )
            return candidate

        session_token = secrets.token_urlsafe(32)

        active_sessions[session_token] = (
            now + SESSION_EXPIRE_SECONDS
        )

        return session_token


def create_pending_action(
    name: str,
    arguments: dict,
    session_token: str
) -> str:
    """创建与服务端会话绑定的待确认操作。"""

    if name != "set_cabin_temp":
        raise ValueError("不支持的待确认工具")

    if not isinstance(arguments, dict):
        raise ValueError("工具参数必须为字典")

    token = secrets.token_urlsafe(32)
    now = time.monotonic()

    with pending_lock:
        if active_sessions.get(session_token, 0) <= now:
            raise ValueError("无效或过期的会话")

        # 清理过期操作
        for key, item in list(pending_actions.items()):
            if item["expires_at"] <= now:
                del pending_actions[key]

        pending_actions[token] = {
            "name": name,
            "arguments": arguments.copy(),
            "session_token": session_token,
            "expires_at": now + EXPIRE_SECONDS
        }

    return token


def consume_pending_action(
    token: str,
    session_token: str | None = None
):
    """仅所属会话能够一次性消费确认令牌。"""

    with pending_lock:
        record = pending_actions.get(token)

        if record is None:
            return None

        now = time.monotonic()

        if record["expires_at"] <= now:
            pending_actions.pop(token, None)
            return None

        # 会话不匹配时，不消费原令牌
        session_valid = (
            session_token is not None
            and active_sessions.get(session_token, 0) > now
            and secrets.compare_digest(
                session_token,
                record["session_token"]
            )
        )

        if not session_valid:
            raise SessionMismatchError(
                "该操作不属于当前会话"
            )

        pending_actions.pop(token)

        return {
            "name": record["name"],
            "arguments": record["arguments"].copy()
        }


def discard_pending_action(token: str):
    """清理辅助函数，主要用于测试。"""

    with pending_lock:
        pending_actions.pop(token, None)