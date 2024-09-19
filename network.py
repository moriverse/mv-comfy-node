import requests
import typing as t

from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry  # type: ignore

USER_AGENT = "Moriverse/Comfy"


def requests_session(auth_key: t.Optional[str] = None) -> requests.Session:
    session = requests.Session()
    session.headers["user-agent"] = (
        USER_AGENT + " " + str(session.headers["user-agent"])
    )

    if auth_key:
        session.headers["Authorization"] = f"Bearer {auth_key}"

    return session


def requests_session_with_retries(
    auth_key: t.Optional[str] = None,
) -> requests.Session:
    # This session will retry requests up to 12 times, with exponential
    # backoff. In total it'll try for up to roughly 320 seconds, providing
    # resilience through temporary networking and availability issues.
    session = requests_session(auth_key)
    adapter = HTTPAdapter(
        max_retries=Retry(
            total=10,
            backoff_factor=0.1,
            status_forcelist=[i for i in range(400, 600)],
            allowed_methods=["POST", "GET"],
        )
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session
