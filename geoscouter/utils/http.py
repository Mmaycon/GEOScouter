"""HTTP helpers for GEO/NCBI fetches."""

import requests

_session = requests.Session()
_session.trust_env = False


def ncbi_get(url: str, **kwargs) -> requests.Response:
    kwargs.setdefault("timeout", 30)
    return _session.get(url, **kwargs)
