import json
from typing import Any


def emit_ui_event(event_type: str, icon: str, title: str, message: str, **extras: Any) -> None:
    """Emit a structured UI event line that the app can parse from stdout.

    Tools can call this to send rich, structured events to the side panel without
    coupling to the UI. The agent log consumer parses lines starting with
    "UI_EVENT " and renders them as cards.
    """
    payload: dict[str, Any] = {
        "type": event_type,
        "icon": icon,
        "title": title,
        "message": message,
    }
    if extras:
        payload.update(extras)
    print("UI_EVENT " + json.dumps(payload), flush=True)


