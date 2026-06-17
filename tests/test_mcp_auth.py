"""OAuth Resource-Server tests — token verification + per-tool scope gating, fully offline.

No network and no IdP: we mint RSA-signed JWTs with a throwaway key and inject the matching public
key into ``JwtTokenVerifier`` (its ``_signing_key`` hook), so verification runs against locally
generated tokens. Covers the accept path and every reject path (expired / wrong-aud / wrong-iss /
bad-signature / missing-exp), scope parsing, ``require_scope``, and ``build_auth`` config wiring.
"""

from __future__ import annotations

import asyncio
import time

import pytest

pytest.importorskip("mcp")
pytest.importorskip("jwt")  # the mcp-auth extra (pyjwt[crypto])

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa

from argus.mcp import auth

ISSUER = "https://idp.example/realms/argus"
RESOURCE = "http://localhost:8000/mcp"

# One throwaway RSA keypair for the whole module: PRIV signs tokens, the verifier trusts its public
# half. A separate key (OTHER) stands in for a token signed by the wrong issuer.
PRIV = rsa.generate_private_key(public_exponent=65537, key_size=2048)
OTHER = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _make_token(*, key=PRIV, **overrides) -> str:
    now = int(time.time())
    payload = {
        "iss": ISSUER,
        "aud": RESOURCE,
        "sub": "user-123",
        "azp": "argus-mcp",
        "iat": now,
        "exp": now + 3600,
        "scope": "argus:peek argus:track",
    }
    payload.update(overrides)
    payload = {k: v for k, v in payload.items() if v is not None}  # allow dropping a claim
    return jwt.encode(payload, key, algorithm="RS256")


def _verifier() -> auth.JwtTokenVerifier:
    # _signing_key bypasses JWKS/network: the verifier trusts PRIV's public half directly.
    return auth.JwtTokenVerifier(ISSUER, RESOURCE, _signing_key=PRIV.public_key())


def _verify(token: str):
    return asyncio.run(_verifier().verify_token(token))


# ---- accept path -----------------------------------------------------------------------


def test_valid_token_returns_access_token():
    tok = _verify(_make_token())
    assert tok is not None
    assert tok.scopes == ["argus:peek", "argus:track"]
    assert tok.client_id == "argus-mcp"
    assert tok.subject == "user-123"
    assert tok.resource == RESOURCE
    assert tok.claims["iss"] == ISSUER


def test_scope_claim_parsed_space_delimited():
    tok = _verify(_make_token(scope="argus:search"))
    assert tok is not None and tok.scopes == ["argus:search"]


def test_scp_list_claim_unioned_with_scope():
    tok = _verify(_make_token(scope="argus:peek", scp=["argus:audio", "argus:peek"]))
    assert tok is not None
    assert tok.scopes == ["argus:peek", "argus:audio"]  # union, order-preserving, de-duped


# ---- reject paths (all -> None, which the SDK turns into 401) ---------------------------


def test_expired_token_rejected():
    assert _verify(_make_token(exp=int(time.time()) - 10)) is None


def test_wrong_audience_rejected():
    assert _verify(_make_token(aud="http://evil.example/mcp")) is None


def test_wrong_issuer_rejected():
    assert _verify(_make_token(iss="https://attacker.example/realms/argus")) is None


def test_bad_signature_rejected():
    # Signed by a key the verifier does not trust.
    assert _verify(_make_token(key=OTHER)) is None


def test_missing_exp_rejected():
    assert _verify(_make_token(exp=None)) is None


def test_garbage_token_rejected():
    assert _verify("not-a-jwt") is None


# ---- per-tool scope gating -------------------------------------------------------------


@pytest.fixture
def _tok(monkeypatch):
    """Helper to set the 'current request' access token require_scope reads."""
    import mcp.server.auth.middleware.auth_context as ctx
    from mcp.server.auth.provider import AccessToken

    def _set(scopes: list[str] | None):
        token = (
            None
            if scopes is None
            else AccessToken(token="t", client_id="argus-mcp", scopes=scopes)
        )
        monkeypatch.setattr(ctx, "get_access_token", lambda: token)

    return _set


def test_require_scope_noop_when_auth_disabled(monkeypatch, _tok):
    monkeypatch.setattr(auth, "_auth_enabled", False)
    _tok(None)  # no token at all
    auth.require_scope("argus:track")  # must not raise


def test_require_scope_passes_when_scope_present(monkeypatch, _tok):
    monkeypatch.setattr(auth, "_auth_enabled", True)
    _tok(["argus:peek", "argus:track"])
    auth.require_scope("argus:track")  # must not raise


def test_require_scope_raises_when_scope_missing(monkeypatch, _tok):
    monkeypatch.setattr(auth, "_auth_enabled", True)
    _tok(["argus:peek"])
    with pytest.raises(auth.InsufficientScopeError):
        auth.require_scope("argus:track")


def test_require_scope_raises_when_no_token(monkeypatch, _tok):
    monkeypatch.setattr(auth, "_auth_enabled", True)
    _tok(None)
    with pytest.raises(auth.InsufficientScopeError):
        auth.require_scope("argus:peek")


# ---- config wiring ---------------------------------------------------------------------


def test_parse_scopes():
    assert auth._parse_scopes("argus:peek, argus:track") == ["argus:peek", "argus:track"]
    assert auth._parse_scopes("argus:peek argus:track") == ["argus:peek", "argus:track"]
    assert auth._parse_scopes("") == [] and auth._parse_scopes(None) == []


def test_auth_config_from_env(monkeypatch):
    monkeypatch.setenv("ARGUS_MCP_AUTH", "on")
    monkeypatch.setenv("ARGUS_OAUTH_ISSUER", ISSUER + "/")  # trailing slash trimmed
    monkeypatch.setenv("ARGUS_OAUTH_RESOURCE", RESOURCE)
    monkeypatch.setenv("ARGUS_OAUTH_SCOPES", "argus:use")
    cfg = auth.AuthConfig.from_env()
    assert cfg.enabled is True
    assert cfg.issuer == ISSUER  # rstrip("/")
    assert cfg.resource == RESOURCE
    assert cfg.scopes == ["argus:use"]


def test_auth_config_default_off(monkeypatch):
    monkeypatch.delenv("ARGUS_MCP_AUTH", raising=False)
    assert auth.AuthConfig.from_env().enabled is False


def test_build_auth_disabled_returns_none():
    verifier, settings = auth.build_auth(auth.AuthConfig(enabled=False))
    assert verifier is None and settings is None


def test_build_auth_enabled_requires_issuer_and_resource():
    with pytest.raises(ValueError):
        auth.build_auth(auth.AuthConfig(enabled=True, issuer="", resource=RESOURCE))
    with pytest.raises(ValueError):
        auth.build_auth(auth.AuthConfig(enabled=True, issuer=ISSUER, resource=""))


def test_build_auth_enabled_builds_verifier_and_settings():
    cfg = auth.AuthConfig(enabled=True, issuer=ISSUER, resource=RESOURCE, scopes=["argus:use"])
    verifier, settings = auth.build_auth(cfg)
    assert isinstance(verifier, auth.JwtTokenVerifier)
    assert verifier.issuer == ISSUER and verifier.resource == RESOURCE
    assert str(settings.issuer_url).rstrip("/") == ISSUER
    assert settings.required_scopes == ["argus:use"]
