"""OAuth 2.1 for the MCP server — the Resource Server (RS) half of MCP authorization.

The MCP authorization model is OAuth 2.1 at the transport layer (see ``context/mcp-oauth-plan.md``).
The token *issuer* is a separate Authorization Server (an IdP, e.g. Keycloak); argus only acts as a
**Resource Server**: it validates the bearer tokens an MCP client attaches and gates each tool by
scope. The MCP SDK does the discovery/challenge plumbing — it serves the RFC 9728 Protected
Resource Metadata document and the ``401 WWW-Authenticate`` challenge from ``AuthSettings``, and
enforces any *global* ``required_scopes`` via middleware. The one contract we implement is a
``TokenVerifier`` (the SDK ships none): :class:`JwtTokenVerifier` below.

Validation is stateless JWT-via-JWKS: fetch + cache the IdP's signing keys and verify the RS256
signature, ``iss``, ``aud`` (RFC 8707 resource binding — the token must have been issued *for this
server*), and ``exp`` locally — no introspection round-trip. Everything here is lazy: it imports
PyJWT only when auth is actually enabled (``ARGUS_MCP_AUTH=on``), keeping the no-auth path and the
synthetic test suite free of the ``mcp-auth`` extra.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, field

import anyio
from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings

# Per-tool scope map. The transport layer (SDK) only checks that a token is valid (+ any blanket
# ``required_scopes``); each tool additionally requires its own scope below via ``require_scope``.
TOOL_SCOPES: dict[str, str] = {
    "list_clips": "argus:peek",
    "peek_folder": "argus:peek",
    "peek_clip": "argus:peek",
    "track_clip": "argus:track",
    "search_face": "argus:search",
    "classify_audio": "argus:audio",
    # Face-ID over the sighting store. Reads share argus:read; search reuses argus:search; the
    # write/mutating ops get their own scope so an operator can grant them narrowly. Note
    # argus:audit (compliance trail) is distinct from argus:audio (sound classification).
    "ingest_clip": "argus:ingest",
    "list_sightings": "argus:read",
    "list_identities": "argus:read",
    "get_face_chip": "argus:read",
    "search_similar": "argus:search",
    "enroll_identity": "argus:enroll",
    "cluster_sightings": "argus:cluster",
    "audit_log": "argus:audit",
}

# Set by ``server.main()`` from ``ARGUS_MCP_AUTH``. When False, ``require_scope`` is a no-op so the
# default loopback/dev path (and ``tests/test_mcp.py``, which calls tools directly) is unaffected.
_auth_enabled = False
# Per-tool scope gating. When False (``ARGUS_MCP_TOOL_SCOPES=off``), a valid token authorizes every
# tool — useful when clients authenticate via dynamic client registration, whose tokens often carry
# only base scopes (no ``argus:*``). Transport auth (valid token) is still required when auth is on.
_tool_scopes_enabled = True


def set_auth_enabled(enabled: bool) -> None:
    """Record whether auth is live (called once at server startup)."""
    global _auth_enabled
    _auth_enabled = enabled


def set_tool_scopes_enabled(enabled: bool) -> None:
    """Record whether per-tool scope checks are enforced (called once at server startup)."""
    global _tool_scopes_enabled
    _tool_scopes_enabled = enabled


class InsufficientScopeError(Exception):
    """A tool was called with a valid token that lacks the tool's required scope.

    Raised inside a tool body (post-authentication), so it surfaces to the client as an MCP tool
    error rather than an HTTP ``403`` — all tools share one ``POST /mcp`` route, so HTTP-level
    per-tool gating isn't possible with FastMCP. HTTP ``403 insufficient_scope`` is reserved for the
    SDK's *blanket* ``required_scopes`` (``ARGUS_OAUTH_SCOPES``).
    """


def require_scope(scope: str) -> None:
    """Assert the current request's token carries ``scope``; no-op when auth is disabled.

    Reads the access token the SDK's bearer-auth middleware stashed for this request. When auth is
    on, the middleware has already rejected tokenless requests with ``401`` before any tool runs, so
    a missing token here is defensive only.
    """
    if not _auth_enabled or not _tool_scopes_enabled:
        return
    from mcp.server.auth.middleware.auth_context import get_access_token

    token = get_access_token()
    if token is None or scope not in token.scopes:
        raise InsufficientScopeError(f"insufficient scope: this tool requires {scope!r}")


def _parse_scopes(raw: str | None) -> list[str]:
    """Split a comma/space-separated scope string into a clean list."""
    return [s for s in (raw or "").replace(",", " ").split() if s]


@dataclass
class AuthConfig:
    """RS auth configuration, read from the environment (see ``.env.example``)."""

    enabled: bool = False
    issuer: str = ""  # IdP base URL, e.g. http://localhost:8080/realms/argus
    resource: str = ""  # this server's canonical URI, e.g. http://localhost:8000/mcp
    scopes: list[str] = field(default_factory=list)  # optional *blanket* required scopes

    @classmethod
    def from_env(cls) -> AuthConfig:
        enabled = os.environ.get("ARGUS_MCP_AUTH", "off").strip().lower() in (
            "on",
            "1",
            "true",
            "yes",
        )
        return cls(
            enabled=enabled,
            issuer=os.environ.get("ARGUS_OAUTH_ISSUER", "").strip().rstrip("/"),
            resource=os.environ.get("ARGUS_OAUTH_RESOURCE", "").strip(),
            scopes=_parse_scopes(os.environ.get("ARGUS_OAUTH_SCOPES")),
        )


def _discover_jwks_uri(issuer: str) -> str:
    """Resolve the IdP's JWKS endpoint from its OIDC discovery document."""
    url = f"{issuer}/.well-known/openid-configuration"
    with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310 (operator-configured IdP)
        doc = json.load(resp)
    jwks_uri = doc.get("jwks_uri")
    if not jwks_uri:
        raise ValueError(f"OIDC discovery at {url} has no jwks_uri")
    return jwks_uri


class JwtTokenVerifier(TokenVerifier):
    """Verify IdP-issued JWTs against the IdP's JWKS (RS256), enforcing iss/aud/exp.

    Signing keys are fetched once from the IdP's JWKS endpoint and cached (PyJWT's ``PyJWKClient``
    handles caching + key rotation). ``aud`` MUST equal :attr:`resource` (RFC 8707): a token issued
    for another resource is rejected. Any failure — bad signature, wrong ``iss``/``aud``, expired,
    malformed — returns ``None``, which the SDK turns into a ``401``.
    """

    def __init__(
        self,
        issuer: str,
        resource: str,
        *,
        jwks_uri: str | None = None,
        _signing_key: object | None = None,
    ) -> None:
        self.issuer = issuer.rstrip("/")
        self.resource = resource
        self._jwks_uri = jwks_uri
        self._signing_key = _signing_key  # test hook: a public key, bypassing JWKS/network
        self._jwk_client = None  # lazily built on first verify (avoids network at construction)

    def _get_signing_key(self, token: str):
        """Return the public key (PyJWT key object) that signed ``token``."""
        if self._signing_key is not None:
            return self._signing_key
        if self._jwk_client is None:
            import jwt

            uri = self._jwks_uri or _discover_jwks_uri(self.issuer)
            self._jwk_client = jwt.PyJWKClient(uri)
        return self._jwk_client.get_signing_key_from_jwt(token).key

    def _verify_sync(self, token: str) -> AccessToken | None:
        import jwt

        try:
            key = self._get_signing_key(token)
            claims = jwt.decode(
                token,
                key,
                algorithms=["RS256"],
                audience=self.resource,
                issuer=self.issuer,
                options={"require": ["exp"]},
            )
        except Exception:
            return None

        # Keycloak/OIDC put granted scopes in the space-delimited `scope` claim; some IdPs use a
        # `scp` list. Accept either (union).
        scopes = _parse_scopes(claims.get("scope"))
        scp = claims.get("scp")
        if isinstance(scp, list):
            scopes = list(dict.fromkeys(scopes + [str(s) for s in scp]))

        return AccessToken(
            token=token,
            client_id=claims.get("azp") or claims.get("client_id") or claims.get("sub") or "",
            scopes=scopes,
            expires_at=claims.get("exp"),
            resource=self.resource,
            subject=claims.get("sub"),
            claims=claims,
        )

    async def verify_token(self, token: str) -> AccessToken | None:
        # PyJWT is sync; the key lookup may (rarely) hit the network on a JWKS refresh, so run it
        # off the event loop. The cached-key path is effectively instant.
        return await anyio.to_thread.run_sync(self._verify_sync, token)


def build_auth(cfg: AuthConfig) -> tuple[TokenVerifier | None, AuthSettings | None]:
    """Translate config into the ``(token_verifier, AuthSettings)`` pair FastMCP wants.

    Returns ``(None, None)`` when auth is disabled — the server constructs FastMCP without auth and
    behaves exactly as before.
    """
    if not cfg.enabled:
        return None, None
    if not cfg.issuer or not cfg.resource:
        raise ValueError(
            "ARGUS_MCP_AUTH=on requires ARGUS_OAUTH_ISSUER and ARGUS_OAUTH_RESOURCE to be set"
        )
    verifier = JwtTokenVerifier(cfg.issuer, cfg.resource)
    settings = AuthSettings(
        issuer_url=cfg.issuer,
        resource_server_url=cfg.resource,
        required_scopes=cfg.scopes or None,
    )
    return verifier, settings
