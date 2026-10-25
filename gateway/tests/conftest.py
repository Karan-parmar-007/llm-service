import os
import sys
import time

# Tests run both halves with a stubbed speech model and no token.
os.environ["SPEECH_ENABLED"] = "1"
os.environ["LLM_ENABLED"] = "1"
os.environ["TARA_EAGER"] = "0"
os.environ["TARA_STUB"] = "1"
os.environ["FLOW_TOKEN"] = ""
os.environ["KEEPALIVE_ENABLED"] = "0"
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def finished_body(client, response, timeout: float = 5, headers: dict | None = None):
    """The model call either finished on this response or left a job to poll.

    Pass the same headers as the original call (e.g. X-Flow-Token): polls are
    authenticated too.
    """
    body = response.get_json()
    if response.status_code == 200 and isinstance(body, dict) and "text" in body:
        return body
    assert response.status_code == 202, body
    deadline = time.time() + timeout
    last = body
    while time.time() < deadline:
        last = client.get(f"/jobs/{body['id']}", headers=headers).get_json()
        if last.get("status") == "done":
            result = last.get("result") or {}
            assert isinstance(result, dict) and "text" in result
            return result
        if last.get("status") == "error":
            raise AssertionError(last)
        time.sleep(0.02)
    raise AssertionError(last)
