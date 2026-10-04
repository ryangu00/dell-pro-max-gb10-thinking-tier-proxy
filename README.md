![banner](docs/assets/banner.png)

# PORT IS THE TIER — a thinking-tier proxy for vLLM on Dell Pro Max with GB10

> One upstream vLLM engine, several local ports, and the port — not the caller — decides whether a request thinks, and how hard. This is a thin stdlib-only Python proxy that sits in front of any OpenAI-compatible vLLM endpoint and rewrites `POST /v1/chat/completions` per port: each port forces a `chat_template_kwargs` thinking toggle and a top-level `reasoning_effort`. Response bytes are streamed unchanged; the request line and hop-by-hop headers are rewritten as any HTTP proxy does; chunked request bodies are de-chunked. The point of this repo is the **trap**: a tiny proxy like this is one shadowed-variable bug away from doing nothing at all, and nothing about the request will tell you.

The proxy rewrites exactly these request fields: it forces the tier's `chat_template_kwargs.<toggle>` key (overwriting the caller's value), replaces the top-level `reasoning_effort`, strips the other model family's toggle key from `chat_template_kwargs`, and fills unspecified sampling keys with `setdefault`. Everything else in the body is left as the caller sent it; the proxy does not generally repair bodies the upstream would reject with HTTP 400.

## Update (2026-10)

One allowed value was missing, and two operational pitfalls surfaced. All observations below concern Dell Pro Max with GB10, with the node count stated for each run or incident. **No live engine probe was run for this update:** the proxy service that fronted the engine was disabled, and no new request was sent to any engine. The evidence is a secondary-to-primary mix of recorded operation, logs, configuration, source reading and upstream documentation. Local unit tests check proxy behavior, not the accepted set of a current engine build.

**The missing value is `max`.** Our DeepSeek-family proxy uses the same port-based tier design: its first port sets `thinking: false`, and its second (quality-tier) port sets `thinking: true` and `reasoning_effort: "max"`. The explicit `max` setting was added on **2026-09-17**, remains in the **2026-09-19** snapshot, and is recorded in configuration notes. During the **2026-09-16/17** stack A/B, the server-side default differed by stack: one defaulted to `max`, while a community recipe defaulted to `high` with `thinking: true`, as shown in engine startup output. Explicit injection avoids inheriting that difference. Source reading showed that the public validator raised `ValueError` for this production value; it now accepts `max` alongside the existing generic values.

The recorded accepted-values evidence for **DeepSeek-V4-Flash-Vision-Exp** is:

| Value | Evidence level | Evidence and conditions |
|---|---|---|
| `max` | observed operation, not a unit probe | Used in production since **2026-09-17**. On **2026-09-25**, a dual-node long-generation run sent `thinking: true, reasoning_effort: max` directly to the engine until the engine died. On **2026-09-26**, a single-node run went through the quality-tier port for **102 minutes** in total at concurrency **2 to 6**, with no request rejected. |
| `high` | observed | The recipe's own default in the **2026-09-25** engine startup log, plus a high-effort round in the same **2026-09-26** single-node run. |
| `low` | documented | A community recipe maps a low switch to `reasoning_effort: low`; research notes on the model's encoding code list `low`, `high` and `max`. Not exercised by us on this engine. |

**Open / not verified:** DeepSeek `medium` and `xhigh` have no record of being sent or accepted; do not rely on them. Research notes record that the official encoder asserts on values outside its set. `low` remains documented, not exercised. The observed `max` acceptance applies to DeepSeek-V4-Flash-Vision-Exp with vLLM builds from **2026-09-13** and the single-node EXL3 build; other engine builds were not tested. The accepted set for the current engine build remains untested for this update.

For contrast, Qwen3.8-Flash-Next accepts `xhigh`, `medium` and `low` only. Rejection of `high` with **HTTP 400** was observed during the original acceptance run. Rejection of `max` is documented in its chat template, read during **September 2026** recipe verification. A generic validator accepting a value does not establish engine support.

**Accepted does not mean inexpensive or reliable on every stack.** The `max` quality-tier workload preceded **two engine deaths on the dual-node stack** and was removed from production fallback chains on **2026-09-25** as a precaution. It was the most expensive and, on that stack, the most failure-prone setting. This is an observed association, not proof that `max` caused the deaths.

**Start nearby tier ports before the engine.** On **2026-09-27**, a single-node engine container had `VLLM_PORT=8899`. Source reading of its network utility showed that vLLM probes upward for free internal-socket ports by binding, closing the test socket, then reusing the number. On the first switch, the engine's core process took port **8901** (confirmed with `ss`), so the proxy started afterward failed with "address already in use". Starting the proxy first to hold **8901** and **8902**, then launching the engine, worked: the second switch had all three ports (**8899**, **8901**, **8902**) up at **14:33:55 local time**. The window covering both attempts lasted about **7 minutes**. Start the proxy first or move its ports well away from the API port, and prove port ownership by PID; an HTTP 200 from another process is not proof of ownership. **Open / not verified:** saving the API port in the entrypoint, unsetting `VLLM_PORT`, and passing the saved port explicitly is a cleaner proposed fix that has never been validated.

**Automatic restart can outlive the upstream.** An older proxy for the previous Qwen-based single-node mode ran under systemd with `Restart=always`. Its upstream container and image were deleted on **2026-09-24**. A switch script killed the proxy, but systemd restarted it within **5 s**, after which every request received **502**. On **2026-09-27 at 15:23 local time**, the unit was stopped and disabled using `systemctl disable --no-reload` plus `systemctl stop`, avoiding a reload of all units. The unit file was kept for rollback, and a check found no consumer references. **Open / not verified:** how long the proxy had been answering 502 before the check is unknown; the upstream had been gone for about **three days**.

## TL;DR

| What | On this machine | Evidence |
|---|---|---|
| Thinking changes the answer size, a lot | Same question on one Dell Pro Max with GB10 running Qwen3.8-27B: **8 output tokens** with thinking off vs **77** with the default | historical observation from our runs; the repo ships no benchmark harness |
| Thinking costs wall-clock, a lot | Two Dell Pro Max with GB10 nodes running DeepSeek V4 Flash: thinking on made responses **6–13× slower**, and a 16K total-token budget could be **fully consumed by reasoning** (content empty) | historical observation from our runs; the repo ships no benchmark harness |

These numbers are historical observations from our runs, not results the repo reproduces: the repo ships no benchmark harness, prompt set, or per-run transcripts. `--selftest` only asserts the body-rewrite logic; it does not call a model.

The default fast port runs `effort=low`; the quality port runs `effort=medium`. These Qwen-family defaults do not establish quality equivalence between tiers. The port is the tier.

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

The two model families this was built against read different `chat_template_kwargs` keys for the on/off toggle. DeepSeek V4 Flash reads `chat_template_kwargs.thinking`; for `reasoning_effort` we have recorded evidence for `high` (the engine's own default in one recipe) and `max` (used in production), and upstream documentation names `low`; we have no record of `medium` or `xhigh` being accepted by that engine, so do not rely on them. Qwen3.8-Flash-Next reads `chat_template_kwargs.enable_thinking` and accepts `xhigh`, `medium` and `low` only; `high` and `max` are rejected (HTTP 400 on `high` observed). The `kwarg` field in the tier config selects which key the port writes, and `inject()` strips the other family's keys so a body shaped for one family does not smuggle a conflicting value through a port bound to the other. The validator stays generic — it accepts `effort` in `{low, medium, high, xhigh, max, null}` regardless of `kwarg` — because which effort values a given engine version accepts is the operator's responsibility to confirm for that build, not something this proxy hard-codes per family.

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

## Pitfalls, in the order we hit them

1. **A local variable named `inject` shadows the `inject()` function, and a bare `except:` swallows the resulting `TypeError` — so every request passes through untouched and nobody notices.** This is a real production bug. The handler here gates on `do_inject` instead and logs loudly on any failure; the selftest inspects `handle.__code__` (recursively across nested code objects) for a bound name `inject`, and the unit tests assert the failure branch logs rather than bare-passes.
2. **A launch probe must include one request that would FAIL if the proxy did nothing.** Send an illegal `reasoning_effort` (one the upstream rejects with HTTP 400) through each port. Choose a value that your engine really rejects. `max` is rejected by the Qwen3.8-Flash-Next engine but is valid on the DeepSeek-V4-Flash engine we ran, so it is a good failing probe for the first and a useless one for the second. If the port does not rewrite it, you get a 400 and you know the proxy is a no-op. A probe that only checks HTTP 200 on a legal request will pass against a proxy that does nothing.
3. **When deriving a new proxy from an old one by editing the docstring, a cut at the wrong triple-quote leaves the module half-docstring, half-code** — it imports but the first real statement is inside a string. Edit docstrings as whole `"""..."""` blocks; do not cut mid-line.
4. **`pkill -f <script name>` kills the SSH session whose command line contains the same string** (e.g. the editor running the script). Prefer the recorded PID or a service manager to stop the proxy; a `[n]ame` character-class pattern — `pkill -f thi[n]king_tier_proxy` — only avoids matching the pattern text itself, and still matches any other command line that happens to contain the full script name. That advice is incomplete for a service manager with automatic restart: with a systemd unit using `Restart=always`, killing by PID or pattern is undone within seconds; ours came back **5 s** after a switch script killed it, as observed on **2026-09-27**. Stop and disable the unit instead (`systemctl stop` and `systemctl disable`), and disable it when you remove its upstream.
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
