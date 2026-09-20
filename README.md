![banner](docs/assets/banner.png)

# PORT IS THE TIER — a thinking-tier proxy for vLLM on Dell Pro Max with GB10

> One upstream vLLM engine, several local ports, and the port — not the caller — decides whether a request thinks, and how hard. This is a thin stdlib-only Python proxy that sits in front of any OpenAI-compatible vLLM endpoint and rewrites `POST /v1/chat/completions` per port: each port forces a `chat_template_kwargs` thinking toggle and a top-level `reasoning_effort`. Response bytes are streamed unchanged; the request line and hop-by-hop headers are rewritten as any HTTP proxy does; chunked request bodies are de-chunked. The point of this repo is the **trap**: a tiny proxy like this is one shadowed-variable bug away from doing nothing at all, and nothing about the request will tell you.

The proxy rewrites exactly these request fields: it forces the tier's `chat_template_kwargs.<toggle>` key (overwriting the caller's value), replaces the top-level `reasoning_effort`, strips the other model family's toggle key from `chat_template_kwargs`, and fills unspecified sampling keys with `setdefault`. Everything else in the body is left as the caller sent it; the proxy does not generally repair bodies the upstream would reject with HTTP 400.

## TL;DR

| What | On this machine | Evidence |
|---|---|---|
| Thinking changes the answer size, a lot | Same question on one Dell Pro Max with GB10 running Qwen3.8-27B: **8 output tokens** with thinking off vs **77** with the default | historical observation from our runs; the repo ships no benchmark harness |
| Thinking costs wall-clock, a lot | Two Dell Pro Max with GB10 nodes running DeepSeek V4 Flash: thinking on made responses **6–13× slower**, and a 16K total-token budget could be **fully consumed by reasoning** (content empty) | historical observation from our runs; the repo ships no benchmark harness |
| Effort tiers are not free, but low ≈ xhigh here | Qwen3.8-Flash-Next, our private eval bank, three-run median: **xhigh / medium / low all scored 91.7**; low used **4.6–5.2K tokens** vs **6.3–9.2K** for xhigh, and finished in **252–271 s** vs **313–433 s** | three-run medians on one private category; see table note below |

These numbers are historical observations from our runs, not results the repo reproduces: the repo ships no benchmark harness, prompt set, or per-run transcripts. `--selftest` only asserts the body-rewrite logic; it does not call a model.

The effort ladder table is three-run medians on one private category; `token` = output tokens per run; `wall` = per-run wall clock. The choice of `low` for the fast port and `medium` for the quality port is a cost choice made at equal medians, not a proven quality equivalence: equal three-run medians on one category do not establish that the tiers are equivalent on quality, only that this sample did not separate them. The fast port runs `effort=low`; the quality port runs `effort=medium`. The port is the tier.

Scope: one machine family, two model families (Qwen and DeepSeek), stdlib only. Nothing here says which effort tier wins on a different workload or a different engine.

## Hardware and software

| | |
|---|---|
| Machine | Dell Pro Max with GB10 (128 GB unified memory, sm_121, aarch64) |
| Upstream | a plain HTTP `host:port` vLLM endpoint (e.g. `http://127.0.0.1:8000`). No HTTPS, no base path, no IPv6 literals — the parser splits on the first `:` after stripping an optional `http://` prefix. |
| Proxy | Python 3.8+, **stdlib only** (`asyncio`, `json`) — no `pip install` |

## How it works

The proxy opens one listening socket per tier. For each request it reads the headers, buffers the body (de-chunking if needed), and:

- if the request is `POST /v1/chat/completions` (exact path match; a query string is allowed), runs the pure `inject(body, tier)` function to force the tier's `chat_template_kwargs` key, replace any caller `reasoning_effort`, and fill sampling defaults with `setdefault`;
- everything else — other paths, other methods, the whole response stream (SSE) — is proxied unchanged. Response bytes are streamed back as received; only the request line and hop-by-hop headers are rewritten, and chunked request bodies are de-chunked.

The two model families this was built against read different `chat_template_kwargs` keys for the on/off toggle: Qwen3.8-Flash-Next reads `enable_thinking` and a `reasoning_effort` from `{xhigh, medium, low}`; DeepSeek V4 Flash reads `thinking` and accepts `reasoning_effort` from `{low, medium, high, xhigh}`. Both reject a `reasoning_effort` they do not understand with HTTP 400. The `kwarg` field in the tier config selects which key the port writes, and `inject()` strips the other family's keys so a body shaped for one family does not smuggle a conflicting value through a port bound to the other. The validator stays generic — it accepts `effort ∈ {low, medium, high, xhigh, null}` regardless of `kwarg` — because which effort values a given engine version accepts is the operator's responsibility to confirm for that build, not something this proxy hard-codes per family.

```python
# the whole policy, pure and testable:
def inject(body, tier):
    ck = dict(body.get("chat_template_kwargs") or {})
    for alien in ("enable_thinking", "thinking", "reasoning_effort"):
        if alien != tier["kwarg"]:
            ck.pop(alien, None)
    ck[tier["kwarg"]] = tier["enable_thinking"]
    body["chat_template_kwargs"] = ck
    if tier["effort"] is None:
        body.pop("reasoning_effort", None)
    else:
        body["reasoning_effort"] = tier["effort"]
    for k, v in tier["sampling"].items():
        body.setdefault(k, v)
    return body
```

## How to run

```bash
# default tiers: 8901=effort=low, 8902=effort=medium (Qwen family)
python3 proxy/thinking_tier_proxy.py --upstream http://127.0.0.1:8000
# expose the ports beyond loopback with --bind 0.0.0.0 (default bind is 127.0.0.1)

# custom tiers (JSON), e.g. a DeepSeek family: kwarg="thinking", effort dropped
python3 proxy/thinking_tier_proxy.py \
  --upstream http://127.0.0.1:8000 \
  --tiers '{"8901":{"enable_thinking":false,"effort":null,"kwarg":"thinking"},
            "8902":{"enable_thinking":true,"effort":null,"kwarg":"thinking"}}'

# selftest (asserts the three correctness properties below)
python3 proxy/thinking_tier_proxy.py --selftest

# unit tests
python3 -m unittest discover -s tests
```

Config also reads `TIERTIER_UPSTREAM`, `TIERTIER_BIND`, and `TIERTIER_TIERS` env vars.

## Results

On Dell Pro Max with GB10. The numbers below are historical observations from our runs; the repo does not ship a benchmark harness, the prompt set, or per-run transcripts, so they are not independently reproducible from this repo alone. `--selftest` asserts the body-rewrite logic only.

- **Qwen3.8-27B, single node.** The same question produced **8 output tokens** with thinking off and **77** with the default thinking setting. Thinking is not free; the question is whether the extra tokens buy correctness.
- **DeepSeek V4 Flash, two nodes.** With thinking on, responses were **6–13× slower** than thinking off. A 16K total-token budget could be **entirely consumed by reasoning**, leaving the content field empty — the request "succeeded" (HTTP 200, no error) but returned no answer. This is the failure mode the effort tiers exist to avoid.
- **Qwen3.8-Flash-Next, effort tiers.** On our private eval bank, three-run medians were **identical at 91.7** for `xhigh`, `medium`, and `low`. `low` used **4.6–5.2K tokens** and finished in **252–271 s**; `xhigh` used **6.3–9.2K tokens** and finished in **313–433 s**. Equal medians on this one category — a cost choice, not a proven quality equivalence — so the fast port runs `low`, the quality port runs `medium`.

## Pitfalls, in the order we hit them

1. **A local variable named `inject` shadows the `inject()` function, and a bare `except:` swallows the resulting `TypeError` — so every request passes through untouched and nobody notices.** This is a real production bug. The handler here gates on `do_inject` instead and logs loudly on any failure; the selftest inspects `handle.__code__` (recursively across nested code objects) for a bound name `inject`, and the unit tests assert the failure branch logs rather than bare-passes.
2. **A launch probe must include one request that would FAIL if the proxy did nothing.** Send an illegal `reasoning_effort` (one the upstream rejects with HTTP 400) through each port. If the port does not rewrite it, you get a 400 and you know the proxy is a no-op. A probe that only checks HTTP 200 on a legal request will pass against a proxy that does nothing.
3. **When deriving a new proxy from an old one by editing the docstring, a cut at the wrong triple-quote leaves the module half-docstring, half-code** — it imports but the first real statement is inside a string. Edit docstrings as whole `"""..."""` blocks; do not cut mid-line.
4. **`pkill -f <script name>` kills the SSH session whose command line contains the same string** (e.g. the editor running the script). Prefer the recorded PID or a service manager to stop the proxy; a `[n]ame` character-class pattern — `pkill -f thi[n]king_tier_proxy` — only avoids matching the pattern text itself, and still matches any other command line that happens to contain the full script name.
5. **A bare `except:` on the rewrite path is the silent-passthrough mechanism.** Any exception there must be logged, never swallowed; the handler writes the failure to stdout (`flush=True`) and forwards the original body so the caller sees the real upstream behavior.

## Files

- `proxy/thinking_tier_proxy.py` — the proxy + pure `inject()` + `--selftest`
- `tests/test_inject.py` — unit tests for `inject()` and the no-shadow guard, runnable with `python3 -m unittest`
- `docs/make_banner.py` — pure-PIL banner generator (no generated imagery); run to produce `docs/assets/banner.png`
- `docs/assets/banner.png` — the banner
- `LICENSE` — Apache 2.0
- `.gitignore`

## License

Apache-2.0.
