#!/usr/bin/env python3
"""thinking-tier-proxy - "the port is the tier".

A thin HTTP proxy in front of an OpenAI-compatible vLLM endpoint. One upstream,
several local ports, each port forcing a different reasoning tier onto every
chat-completions request that passes through it. The caller does not get to pick
the tier; the port picks it. Everything that is not a chat completion is passed
through byte-for-byte, streaming included.

Why: the two model families this was built against expose thinking toggles under
`chat_template_kwargs`, but with different key names. Qwen3.8-Flash-Next reads
`chat_template_kwargs.enable_thinking` and a top-level `reasoning_effort` drawn
from {xhigh, medium, low}; DeepSeek V4 Flash reads `chat_template_kwargs.thinking`
and a top-level `reasoning_effort` from {low, medium, high, xhigh}. Both also
reject any `reasoning_effort` they do not understand with HTTP 400. Forcing the
value at the port means a caller that sends a family-illegal effort (e.g. "max"
at a Qwen port) is corrected before it can reach the engine.

Only POST /v1/chat/completions is rewritten; every other method/path is proxied
unchanged. Pure stdlib (asyncio + json). SSE responses are streamed back as-is.

Config: `--upstream http://127.0.0.1:8000` (or TIERTIER_UPSTREAM env), `--bind`
(or TIERTIER_BIND env, default 127.0.0.1; use 0.0.0.0 to expose the ports beyond
loopback), and `--tiers '<JSON>'` (or TIERTIER_TIERS env), a mapping of local port -> tier spec:
    {
      "<port>": {
        "enable_thinking": true|false,
        "effort": "low"|"medium"|"high"|"xhigh"|null,
        "kwarg": "enable_thinking"|"thinking",
        "sampling": {"temperature":1.0,"top_p":0.95,"top_k":20,"presence_penalty":0.0}  // optional
      }
    }
`kwarg` is the chat_template_kwargs key the engine reads for the on/off toggle;
Qwen uses "enable_thinking", DeepSeek uses "thinking". `effort` is the top-level
reasoning_effort forced on the body (null drops the field entirely). `sampling`
entries are applied with setdefault, so a caller-chosen value always wins.

Selftest: `python3 thinking_tier_proxy.py --selftest`.
"""
import asyncio
import json
import os
import sys

DEFAULT_TIERS = {
    "8901": {
        "enable_thinking": True,
        "effort": "low",
        "kwarg": "enable_thinking",
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0},
    },
    "8902": {
        "enable_thinking": True,
        "effort": "medium",
        "kwarg": "enable_thinking",
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0},
    },
}


def _normalize_tier(raw):
    """Validate/normalize one tier dict from config so inject() can trust it."""
    if not isinstance(raw, dict):
        raise ValueError(f"tier must be an object, got {type(raw).__name__}")
    if "kwarg" not in raw:
        raise ValueError("tier is missing required field 'kwarg'")
    kwarg = raw["kwarg"]
    if kwarg not in ("enable_thinking", "thinking"):
        raise ValueError(f"kwarg must be 'enable_thinking' or 'thinking', got {kwarg!r}")
    if "enable_thinking" not in raw:
        raise ValueError("tier is missing required field 'enable_thinking'")
    # enable_thinking must be a real JSON boolean, not a truthy string/"null" that
    # bool() would silently flip ("false" -> True, None -> False).
    if not isinstance(raw["enable_thinking"], bool):
        raise ValueError(f"enable_thinking must be a JSON boolean (true/false), "
                         f"got {raw['enable_thinking']!r}")
    effort = raw.get("effort")
    if effort is not None and effort not in ("low", "medium", "high", "xhigh"):
        raise ValueError(f"effort must be low|medium|high|xhigh|null, got {effort!r}")
    sampling = dict(raw.get("sampling") or {})
    # sampling is applied with setdefault, so a key here that collides with a
    # policy field the proxy already controls would let the config undo the tier
    # enforcement after the fact. Reject those collisions up front.
    reserved = {"reasoning_effort", "chat_template_kwargs"}
    collisions = sorted(k for k in sampling if k in reserved)
    if collisions:
        raise ValueError(f"sampling keys collide with policy fields: {collisions}")
    return {
        "kwarg": kwarg,
        "enable_thinking": raw["enable_thinking"],
        "effort": effort,
        "sampling": sampling,
    }


def inject(body, tier):
    """Rewrite one chat-completions body in place for the tier. Pure function.

    Returns the same dict (mutated) so callers can chain. The tier forcibly
    overrides whatever the caller sent for the thinking toggle and the top-level
    reasoning_effort; sampling defaults are filled with setdefault so a caller
    value is never clobbered. Any other chat_template_kwargs the caller set are
    preserved.

    This function is deliberately separate from the HTTP handler so it can be
    unit-tested without sockets (see tests/test_inject.py).
    """
    ck = dict(body.get("chat_template_kwargs") or {})
    # Drop any thinking-toggle keys this family does not use, so a body shaped for
    # the other family does not smuggle a conflicting value through.
    for alien in ("enable_thinking", "thinking", "reasoning_effort"):
        if alien != tier["kwarg"]:
            ck.pop(alien, None)
    ck[tier["kwarg"]] = tier["enable_thinking"]
    body["chat_template_kwargs"] = ck

    for k, v in tier["sampling"].items():
        body.setdefault(k, v)                    # official sampling only when the caller did not choose

    # Apply the effort policy AFTER the sampling setdefault, so a null effort
    # really removes the field even if the config/caller tried to set it.
    eff = tier["effort"]
    if eff is None:
        body.pop("reasoning_effort", None)      # fast tier: no effort field at all
    else:
        body["reasoning_effort"] = eff          # replace any caller value, legal or illegal
    return body


def _selftest():
    tiers = {p: _normalize_tier(t) for p, t in DEFAULT_TIERS.items()}
    t_fast = tiers["8901"]   # Qwen-style: kwarg=enable_thinking, effort=low
    t_qual = tiers["8902"]   # Qwen-style: kwarg=enable_thinking, effort=medium

    # (a) each tier injects the right kwarg and effort
    o = inject({"model": "x", "messages": [], "reasoning_effort": "high"},
               t_fast)
    assert o["chat_template_kwargs"] == {"enable_thinking": True}, o
    assert o["reasoning_effort"] == "low", o
    assert o["temperature"] == 1.0 and o["presence_penalty"] == 0.0, o

    o = inject({"model": "x", "messages": [], "temperature": 0.2}, t_qual)
    assert o["chat_template_kwargs"] == {"enable_thinking": True}, o
    assert o["reasoning_effort"] == "medium", o
    assert o["temperature"] == 0.2 and o["top_p"] == 0.95, o   # caller temp kept, default top_p filled

    # DeepSeek-style tier: kwarg=thinking, effort dropped (null)
    ds = _normalize_tier({"enable_thinking": True, "effort": None, "kwarg": "thinking"})
    o = inject({"model": "x", "messages": [],
                "chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "high"},
                "reasoning_effort": "max"}, ds)
    assert o["chat_template_kwargs"] == {"thinking": True}, o    # alien toggles stripped, family kwarg set
    assert "reasoning_effort" not in o, o                        # null effort drops the field

    # effort=None tier drops the field even when the caller sent a legal value
    o = inject({"model": "x", "messages": [], "reasoning_effort": "medium"},
               _normalize_tier({"enable_thinking": False, "effort": None, "kwarg": "thinking"}))
    assert "reasoning_effort" not in o, o
    assert o["chat_template_kwargs"] == {"thinking": False}, o

    # caller's unrelated chat_template_kwargs are preserved
    o = inject({"model": "x", "messages": [],
                "chat_template_kwargs": {"preserve_thinking": False}}, t_qual)
    assert o["chat_template_kwargs"] == {"preserve_thinking": False, "enable_thinking": True}, o

    # (b) an incoming illegal reasoning_effort is replaced by the tier value
    o = inject({"model": "x", "messages": [], "reasoning_effort": "max"}, t_fast)
    assert o["reasoning_effort"] == "low", o
    o = inject({"model": "x", "messages": [], "reasoning_effort": "HIGH"}, t_qual)
    assert o["reasoning_effort"] == "medium", o

    # (c) the handler never shadows the name `inject` with a local variable,
    # checked against the bytecode, not a source-string match. A source match
    # can be defeated by indentation/spacing tricks or a parameter named
    # `inject`. co_varnames lists every local name (and parameters) the
    # compiler bound in this scope; co_cellvars/co_freevars cover names captured
    # by nested closures. We recurse into nested code objects (defaults,
    # comprehensions, nested defs) so a closure binding `inject` is caught too.
    # Real production bug: a boolean named `inject` in the handler made the
    # `inject(obj, ...)` call raise TypeError inside a bare `except:`, so every
    # request passed through untouched and nobody noticed.
    def _code_names(code):
        names = set(code.co_varnames) | set(code.co_cellvars) | set(code.co_freevars)
        for const in code.co_consts:
            if hasattr(const, "co_varnames"):  # nested code object
                names |= _code_names(const)
        return names

    assert "inject" not in _code_names(handle.__code__), \
        "inject() must not be shadowed by a local/parameter/freevar named inject in handle()"

    print("selftest ok")


async def read_chunked(reader, buf):
    """Decode an RFC 7230 chunked body. Returns the full body bytes (trailers dropped)."""
    async def fill(cond):
        nonlocal buf
        while not cond(buf):
            d = await reader.read(65536)
            if not d:
                raise ConnectionError("client EOF mid-chunked")
            buf += d
    body = b""
    while True:
        await fill(lambda b: b"\r\n" in b)
        line, _, buf = buf.partition(b"\r\n")
        size = int(line.split(b";")[0].strip() or b"0", 16)
        if size == 0:
            while True:  # trailers until the empty line
                await fill(lambda b: b"\r\n" in b)
                line, _, buf = buf.partition(b"\r\n")
                if not line:
                    return body
        await fill(lambda b, n=size: len(b) >= n + 2)
        body += buf[:size]
        buf = buf[size + 2:]


async def pump(reader, writer):
    try:
        while True:
            chunk = await reader.read(65536)
            if not chunk:
                break
            writer.write(chunk)
            await writer.drain()
    except (ConnectionResetError, BrokenPipeError):
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def handle(client_r, client_w, tier, upstream_host, upstream_port):
    try:
        head = b""
        while b"\r\n\r\n" not in head:
            b_ = await client_r.read(65536)
            if not b_:
                return
            head += b_
        header_blob, _, body_start = head.partition(b"\r\n\r\n")
        lines = header_blob.split(b"\r\n")
        request_line = lines[0].decode("latin1")
        method, path, _ = request_line.split(" ", 2)
        headers = {}
        for ln in lines[1:]:
            k, _, v = ln.decode("latin1").partition(":")
            headers[k.strip().lower()] = v.strip()

        if "chunked" in headers.get("transfer-encoding", "").lower():
            body = await read_chunked(client_r, body_start)   # de-chunk, then recompute content-length
        else:
            clen = int(headers.get("content-length", "0"))
            body = body_start
            while len(body) < clen:
                chunk = await client_r.read(65536)
                if not chunk:
                    return  # client half-open: returning avoids a busy-loop burning CPU
                body += chunk

        # Match the rewrite path exactly: parse the URL and compare the path
        # component, so /v1/chat/completions-extra (or any other prefix match)
        # is proxied untouched. A query string is allowed.
        from urllib.parse import urlsplit
        url_path = urlsplit(path).path
        do_inject = method == "POST" and url_path == "/v1/chat/completions"
        # NOTE: do NOT name a local `inject` here. A boolean with that name once
        # shadowed the inject() function; the call raised TypeError inside a bare
        # except and every request passed through untouched. See README pitfalls +
        # the selftest assertion that guards this.
        if do_inject:
            try:
                obj = json.loads(body.decode("utf-8"))
                inject(obj, tier)
                body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            except Exception as e:
                # Never swallow: log loudly and pass the original body through so
                # the caller sees an upstream error instead of a silent no-op.
                print(f"thinking-tier-proxy: inject failed, passing through: "
                      f"{type(e).__name__}: {e}", flush=True)

        up_r, up_w = await asyncio.open_connection(upstream_host, upstream_port)
        out = [f"{method} {path} HTTP/1.1"]
        for k, v in headers.items():
            if k in ("content-length", "host", "connection", "transfer-encoding"):
                continue
            out.append(f"{k}: {v}")
        out.append(f"host: {upstream_host}:{upstream_port}")
        out.append("connection: close")
        out.append(f"content-length: {len(body)}")
        up_w.write(("\r\n".join(out) + "\r\n\r\n").encode("latin1") + body)
        await up_w.drain()
        await pump(up_r, client_w)   # stream the response back, SSE unchanged
        up_w.close()
    except Exception as e:
        try:
            msg = json.dumps({"error": {"message": f"thinking-tier-proxy: {e}"}}).encode()
            client_w.write(b"HTTP/1.1 502 Bad Gateway\r\ncontent-type: application/json\r\n"
                           b"content-length: " + str(len(msg)).encode() + b"\r\n"
                           b"connection: close\r\n\r\n" + msg)
            await client_w.drain()
        except Exception:
            pass
    finally:
        try:
            client_w.close()
        except Exception:
            pass


def _parse_upstream(s):
    # accept "http://127.0.0.1:8000", "127.0.0.1:8000", or ":8000" -> host, port
    s = s.strip()
    if s.startswith("http://"):
        s = s[len("http://"):]
    if s.startswith("/"):
        s = s[1:]
    host, _, port = s.partition(":")
    return host or "127.0.0.1", int(port)


async def main(upstream, tiers, bind):
    host, port = _parse_upstream(upstream)
    upstream_bind = bind
    servers = []
    bind = upstream_bind
    for port_str, tier in tiers.items():
        listen_port = int(port_str)
        srv = await asyncio.start_server(
            lambda r, w, kw=tier: handle(r, w, kw, host, port), bind, listen_port)
        servers.append(srv)
        ck_key = tier["kwarg"]
        print(f"thinking-tier-proxy: :{listen_port} -> {host}:{port} "
              f"{ck_key}={tier['enable_thinking']} effort={tier['effort']}", flush=True)
    await asyncio.gather(*(s.serve_forever() for s in servers))


def _build_parser():
    import argparse
    p = argparse.ArgumentParser(
        prog="thinking_tier_proxy.py",
        description="A thinking-tier HTTP proxy for vLLM. The port — not the caller — "
                    "decides whether a request thinks and how hard.",
    )
    p.add_argument("--upstream", default=os.environ.get("TIERTIER_UPSTREAM", "http://127.0.0.1:8000"),
                   help="upstream vLLM endpoint as http://host:port (default: %(default)s; "
                        "env TIERTIER_UPSTREAM). Must be a plain HTTP host:port — no HTTPS, "
                        "base path, or IPv6 literal.")
    p.add_argument("--bind", default=os.environ.get("TIERTIER_BIND", "127.0.0.1"),
                   help="bind address for the listening ports (default: %(default)s; "
                        "env TIERTIER_BIND; use 0.0.0.0 to expose beyond loopback).")
    p.add_argument("--tiers", default=os.environ.get("TIERTIER_TIERS"),
                   help='JSON mapping of port -> tier spec (env TIERTIER_TIERS). Default: '
                        'the built-in Qwen-family tiers.')
    p.add_argument("--selftest", action="store_true",
                   help="run built-in self-test and exit (no sockets opened).")
    return p


def _load_args(argv):
    parser = _build_parser()
    # argparse handles --help (prints usage, exits 0) and unknown args (exits 2)
    # before any socket is opened, because __main__ calls us before asyncio.run.
    args = parser.parse_args(argv)
    tiers_env = args.tiers
    if tiers_env:
        raw = json.loads(tiers_env)
    else:
        raw = DEFAULT_TIERS
    tiers = {str(p): _normalize_tier(t) for p, t in raw.items()}
    return args.upstream, tiers, args.bind


if __name__ == "__main__":
    # --help/--selftest are handled by argparse and exit before main(), so no
    # socket is opened for them. Unknown arguments make argparse exit 2.
    parser = _build_parser()
    args = parser.parse_args(sys.argv[1:])
    if args.selftest:
        _selftest(); sys.exit(0)
    tiers_env = args.tiers
    raw = json.loads(tiers_env) if tiers_env else DEFAULT_TIERS
    tiers = {str(p): _normalize_tier(t) for p, t in raw.items()}
    try:
        asyncio.run(main(args.upstream, tiers, args.bind))
    except KeyboardInterrupt:
        sys.exit(0)
