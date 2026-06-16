# MCP OAuth — plan for adding & testing auth

## Goal & context

The argus MCP server (`argus/mcp/`, streamable HTTP) currently has **no auth** — fine for a
LAN/loopback prototype, but a real deployment must not expose `track`/`peek` over an open port.
A project requirement is a **full OAuth 2.1 flow**, tested with the **MCPJam** inspector.

This plan adds OAuth per the MCP Authorization spec and a repeatable test loop. It's a follow-up
to [mcp-server.md](mcp-server.md); the "no auth" note there is what this closes.

## How MCP auth works (the model we must implement)

MCP authorization is OAuth 2.1 at the transport layer (HTTP only; stdio uses env creds). Roles:

- **MCP server = OAuth 2.1 Resource Server (RS)** — validates access tokens, serves discovery
  metadata, returns `401`/`403` challenges. **This is the part we build.**
- **Authorization Server (AS)** — issues tokens, runs the user-facing login/consent. The spec
  explicitly puts the AS *out of scope* and recommends a **separate** AS (an IdP), not the MCP
  server issuing its own tokens.
- **MCP client** — discovers the AS, runs the OAuth dance (PKCE), attaches `Bearer` tokens.

What the **RS (our server) MUST do** (2025-11-25 spec):
1. Serve **Protected Resource Metadata (RFC 9728)** at
   `/.well-known/oauth-protected-resource` with an `authorization_servers` field pointing at the
   IdP.
2. On unauthenticated requests, return **`401` with `WWW-Authenticate: Bearer
   resource_metadata="…", scope="…"`**.
3. **Validate every token**: signature/active, `exp`, issuer, and crucially the **audience** —
   the token MUST have been issued *for this server* (RFC 8707 resource binding). Reject
   otherwise with `401`; insufficient scope → `403` + `WWW-Authenticate: …error="insufficient_scope"`.
4. **Never** accept tokens issued for other resources, and **never** pass the client's token
   through to upstream services (confused-deputy).

PKCE (S256), dynamic client registration / Client-ID-Metadata-Documents / pre-registration are
**client + AS** concerns — the IdP provides them; we don't implement them.

## SDK support — what we get for free

We use the official SDK's `mcp.server.fastmcp.FastMCP`. It implements the **RS pattern** natively:

- `FastMCP(name, token_verifier=<TokenVerifier>, auth=AuthSettings(...))`.
- `AuthSettings(issuer_url=…, resource_server_url=…, required_scopes=[…])` — the SDK uses this to
  **serve the RFC 9728 PRM document and emit the 401 `WWW-Authenticate` challenge automatically**.
- We implement one class: a `TokenVerifier.verify_token(token) -> AccessToken | None` (return
  `token, client_id, scopes, expires_at, resource`).

So the RS work is small. The SDK does **not** ship a full AS — that's by design; we bring an IdP.
(Reference: the SDK's `examples/servers/simple-auth/`.)

## Architecture decision

**RS (FastMCP) + external AS (IdP). Recommended IdP for on-prem: Keycloak** (self-hosted, OAuth
2.1 + OIDC discovery, PKCE, DCR, JWKS — runs in a container next to argus). A managed IdP
(Auth0 / Okta / Scalekit / WorkOS) is a drop-in alternative if cloud is acceptable.

**Token validation: JWT via JWKS** (stateless — fetch the IdP's signing keys, verify signature +
`iss` + `aud` + `exp` + scopes locally). Simpler and faster than RFC 7662 introspection; switch to
introspection only if the IdP issues opaque tokens.

**Auth is a toggle** (`ARGUS_MCP_AUTH=off|on`, default `off`): loopback/dev stays frictionless;
real deployments set it `on`. When on, all four tools require a valid token (optionally
scope-gated, e.g. `track_clip` needs `argus:track`).

## Implementation plan

1. **Dependencies** — add an auth extra in `pyproject.toml`:
   `mcp-auth = ["mcp>=1.9", "pyjwt[crypto]>=2.8", "httpx>=0.27"]` (JWKS fetch + JWT verify). Keep
   it optional; fold into the `mcp` Docker stage install.
2. **`argus/mcp/auth.py`** *(new)* — `JwtTokenVerifier(TokenVerifier)`:
   - Lazily fetch + cache the IdP JWKS (`issuer_url/.well-known/...`), verify RS256 signature.
   - Check `iss == issuer`, `aud`/`resource == resource_server_url` (RFC 8707), `exp`, and map
     token scopes → `AccessToken.scopes`. Return `None` on any failure (SDK → 401).
   - A `build_auth(settings)` helper returning `(verifier, AuthSettings)` or `(None, None)` when
     auth is off.
3. **`argus/mcp/server.py`** — read auth env in `main()`; when enabled, construct
   `FastMCP("argus", token_verifier=verifier, auth=AuthSettings(issuer_url, resource_server_url,
   required_scopes))`. Tools unchanged. Optionally annotate per-tool required scopes.
4. **Config (`.env` / `.env.example`)** — add `ARGUS_MCP_AUTH`, `ARGUS_OAUTH_ISSUER`
   (IdP URL), `ARGUS_OAUTH_RESOURCE` (this server's canonical URI, e.g.
   `http://localhost:8000/mcp`), `ARGUS_OAUTH_SCOPES`.
5. **Local IdP for dev/test (`docker-compose.yml`)** — add a `keycloak` service (dev mode,
   imported realm JSON under `deploy/keycloak/realm-argus.json` defining a realm, an `argus-mcp`
   client/audience, scopes `argus:peek`/`argus:track`, and a test user). The `mcp`/`mcp-gpu`
   services gain the `ARGUS_OAUTH_*` env and depend on `keycloak`.
6. **Docs** — extend [mcp-server.md](mcp-server.md) with an "Authentication" section (enable
   auth, the IdP, the mcpjam test loop) and replace the "no auth" caveat.

## Testing plan

**a. Unit (no network, no IdP)** — `tests/test_mcp_auth.py`: feed `JwtTokenVerifier` locally-minted
JWTs (sign with a throwaway RSA key, point the verifier at an in-test JWKS/monkeypatched key):
assert a valid token → `AccessToken`; expired / wrong-`aud` / wrong-`iss` / bad-signature /
missing-scope → `None`. Pure, fast, CI-safe.

**b. MCPJam OAuth Debugger (the required tool)** — end-to-end conformance against a running
server + Keycloak:
```bash
docker compose up -d keycloak mcp        # IdP + RS with ARGUS_MCP_AUTH=on
npx @mcpjam/inspector@latest             # open the printed localhost URL
```
In the inspector's **OAuth Debugger**, point at `http://localhost:8000/mcp` and run the guided
flow — it walks and shows each step: PRM **discovery (RFC 9728)** → AS metadata → **client
registration** (DCR / CIMD / pre-reg) → **authorization redirect + PKCE** → **token exchange** →
**authenticated MCP request**. It runs conformance checks across spec versions (03-26 / 06-18 /
11-25). Green across all steps = pass. Use raw mode to inspect the 401 challenge + token claims.

**c. Negative checks** — with the inspector / `curl`: no token → `401` + `WWW-Authenticate` with
`resource_metadata`; tampered/expired token → `401`; token with wrong audience → `401`; valid
token missing a tool's scope → `403 insufficient_scope`.

**d. Regression** — `uv run pytest` stays green with auth **off** (default), so the existing
suite and the no-auth dev loop are unaffected.

## Phasing & effort

- **Phase 1 — RS + JWT verify + toggle + unit tests** (the core; SDK does most of the lifting). Small.
- **Phase 2 — Keycloak compose + realm + mcpjam green** (the bulk of the wall-clock: standing up
  and configuring the IdP, not code). Medium.
- **Phase 3 — per-tool scopes, docs, harden** (short-lived tokens, HTTPS/TLS termination). Small.

## Open questions / decisions

- **IdP**: Keycloak (on-prem, recommended) vs a managed IdP? Does the org already run one?
- **Token format**: JWT/JWKS (assumed) vs opaque + RFC 7662 introspection?
- **Scope granularity**: single `argus:use` vs per-tool (`argus:peek`, `argus:track`)?
- **TLS**: the spec requires HTTPS for AS endpoints and non-localhost redirects — terminate TLS at
  a reverse proxy (Caddy/nginx) in front of the container for non-loopback deployments.
- **Stay on the SDK's FastMCP** (RS-only auth, fits this plan) vs migrate to standalone `fastmcp`
  v2 (richer built-in IdP/OAuth-proxy providers) — only worth it if we later want the server to
  broker third-party auth.

## Sources

- [MCP Authorization spec (2025-11-25)](https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization)
- [Understanding Authorization in MCP](https://modelcontextprotocol.io/docs/tutorials/security/authorization)
- [MCP Python SDK — simple-auth example](https://github.com/modelcontextprotocol/python-sdk/tree/main/examples/servers/simple-auth)
- [MCPJam Inspector — OAuth Debugger](https://docs.mcpjam.com/inspector/guided-oauth) ·
  [getting started](https://docs.mcpjam.com/getting-started)
- [Auth0 — MCP spec June 2025 auth update](https://auth0.com/blog/mcp-specs-update-all-about-auth/)
