import time

import requests


def get_with_retries(session, url, params=None, timeout=30, retries=3, backoff=2):
    """
    GET with simple retry-with-backoff for transient failures -- both
    connection-level (read timeouts, connection errors) and server-level
    (5xx responses). nfldata.org has shown all three failure modes within
    one session: stale data (see completed_weeks_for_season's docstring),
    a read timeout, and a bare 500 Internal Server Error, so a single
    flaky request shouldn't crash an entire scheduled notebook run.

    On persistent connection-level failure, re-raises the last exception
    (a genuinely down endpoint should still surface as an error, not hang
    or return nothing silently). On persistent 5xx, returns the last
    response as-is rather than raising here -- callers already call
    resp.raise_for_status() or check resp.status_code themselves, so this
    keeps that contract unchanged and just adds retries transparently
    underneath it.
    """
    resp = None
    last_exc = None
    for attempt in range(retries):
        try:
            resp = session.get(url, params=params, timeout=timeout)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_exc = e
            resp = None

        if resp is not None and resp.status_code < 500:
            return resp
        if attempt < retries - 1:
            time.sleep(backoff * (attempt + 1))

    if resp is not None:
        return resp
    raise last_exc
