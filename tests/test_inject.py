"""Unit tests for the pure inject() function. Run with: python3 -m unittest discover -s tests"""
import importlib.util
import os
import unittest

# Import the proxy module without requiring a package; the proxy dir may or may
# not be on sys.path depending on how the suite is launched.
_here = os.path.dirname(os.path.abspath(__file__))
_proxy = os.path.join(_here, "..", "proxy", "thinking_tier_proxy.py")
_spec = importlib.util.spec_from_file_location("thinking_tier_proxy", _proxy)
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)


def tier(enable_thinking=True, effort="low", kwarg="enable_thinking",
         sampling=None):
    return m._normalize_tier({
        "enable_thinking": enable_thinking,
        "effort": effort,
        "kwarg": kwarg,
        "sampling": sampling or {"temperature": 1.0, "top_p": 0.95,
                                  "top_k": 20, "presence_penalty": 0.0},
    })


class InjectBasic(unittest.TestCase):
    def test_qwen_fast_tier_sets_enable_thinking_and_low(self):
        o = m.inject({"model": "x", "messages": [], "reasoning_effort": "high"},
                     tier(effort="low"))
        self.assertEqual(o["chat_template_kwargs"], {"enable_thinking": True})
        self.assertEqual(o["reasoning_effort"], "low")

    def test_qwen_quality_tier_sets_medium(self):
        o = m.inject({"model": "x", "messages": []}, tier(effort="medium"))
        self.assertEqual(o["reasoning_effort"], "medium")
        self.assertEqual(o["chat_template_kwargs"], {"enable_thinking": True})

    def test_deepseek_kwarg_is_thinking(self):
        ds = tier(enable_thinking=True, effort=None, kwarg="thinking")
        o = m.inject({"model": "x", "messages": []}, ds)
        self.assertEqual(o["chat_template_kwargs"], {"thinking": True})

    def test_toggle_can_be_false(self):
        off = tier(enable_thinking=False, effort=None, kwarg="thinking")
        o = m.inject({"model": "x", "messages": []}, off)
        self.assertEqual(o["chat_template_kwargs"], {"thinking": False})


class IllegalEffortReplaced(unittest.TestCase):
    def test_max_replaced_by_tier_effort(self):
        o = m.inject({"model": "x", "messages": [], "reasoning_effort": "max"},
                     tier(effort="low"))
        self.assertEqual(o["reasoning_effort"], "low")

    def test_high_replaced_by_tier_effort(self):
        o = m.inject({"model": "x", "messages": [], "reasoning_effort": "HIGH"},
                     tier(effort="medium"))
        self.assertEqual(o["reasoning_effort"], "medium")

    def test_null_effort_drops_field(self):
        o = m.inject({"model": "x", "messages": [], "reasoning_effort": "low"},
                     tier(effort=None))
        self.assertNotIn("reasoning_effort", o)


class SamplingDefaults(unittest.TestCase):
    def test_setdefault_fills_missing_keys(self):
        o = m.inject({"model": "x", "messages": []}, tier(effort="low"))
        self.assertEqual(o["temperature"], 1.0)
        self.assertEqual(o["top_p"], 0.95)
        self.assertEqual(o["top_k"], 20)
        self.assertEqual(o["presence_penalty"], 0.0)

    def test_caller_value_wins(self):
        o = m.inject({"model": "x", "messages": [], "temperature": 0.2},
                     tier(effort="medium"))
        self.assertEqual(o["temperature"], 0.2)
        self.assertEqual(o["top_p"], 0.95)


class CrossFamilyStripping(unittest.TestCase):
    def test_alien_toggle_stripped_from_chat_template_kwargs(self):
        # a body shaped for Qwen (enable_thinking) routed to a DeepSeek (thinking) port
        ds = tier(enable_thinking=True, effort=None, kwarg="thinking")
        o = m.inject({"model": "x", "messages": [],
                       "chat_template_kwargs": {"enable_thinking": False,
                                                 "reasoning_effort": "high"}}, ds)
        self.assertNotIn("enable_thinking", o["chat_template_kwargs"])
        self.assertEqual(o["chat_template_kwargs"], {"thinking": True})

    def test_caller_other_kwargs_preserved(self):
        o = m.inject({"model": "x", "messages": [],
                       "chat_template_kwargs": {"preserve_thinking": False}},
                     tier(effort="medium"))
        self.assertEqual(o["chat_template_kwargs"],
                         {"preserve_thinking": False, "enable_thinking": True})


class HandlerNoInjectShadow(unittest.TestCase):
    """Guard against the real production bug: a local variable named `inject`
    in handle() shadowed the inject() function; the call raised TypeError inside
    a bare except, so requests passed through untouched.

    The check inspects the bytecode (co_varnames / co_cellvars / co_freevars),
    recursively across nested code objects, rather than grepping the source — a
    source match can be defeated by indentation/spacing tricks or by a
    parameter named `inject`."""

    def _code_names(self, code):
        names = set(code.co_varnames) | set(code.co_cellvars) | set(code.co_freevars)
        for const in code.co_consts:
            if hasattr(const, "co_varnames"):  # nested code object
                names |= self._code_names(const)
        return names

    def test_handle_does_not_shadow_inject(self):
        names = self._code_names(m.handle.__code__)
        self.assertNotIn("inject", names,
                         "handle() binds a local/parameter/freevar named 'inject', "
                         "which would shadow the inject() function")

    def test_negative_local_inject_is_detected(self):
        # A function that assigns a local `inject` MUST trip the check, proving
        # the guard actually catches the bug it exists for (not a vacuous pass).
        def buggy():
            inject = True  # noqa: F841 — deliberately shadows
            return inject
        names = self._code_names(buggy.__code__)
        self.assertIn("inject", names)


class InjectExceptionBranch(unittest.TestCase):
    """Exercise the handler's inject-failure branch end to end: a fake upstream
    + fake writers, force the inject path to raise, assert the log line is
    emitted and the ORIGINAL body is forwarded unchanged."""

    def _make_broken_body(self):
        # json.loads succeeds but inject() then needs a real dict to mutate;
        # a JSON non-object (e.g. a bare array) makes the `inject(obj, tier)`
        # call raise AttributeError, exercising the except branch.
        return b'["not","an","object"]'

    def test_exception_logged_and_original_body_forwarded(self):
        import asyncio

        # Fake client reader: serves a POST /v1/chat/completions request with a
        # body that parses as JSON but is not a dict, so inject() raises.
        request = (
            b"POST /v1/chat/completions HTTP/1.1\r\n"
            b"host: 127.0.0.1:8901\r\n"
            b"content-length: 21\r\n"
            b"\r\n" + self._make_broken_body()
        )
        body_bytes = self._make_broken_body()

        class FakeReader:
            def __init__(self, data):
                self._buf = data
                self._pos = 0
            async def read(self, n):
                chunk = self._buf[self._pos:self._pos + n]
                self._pos += len(chunk)
                return chunk

        class FakeWriter:
            def __init__(self):
                self.written = b""
                self.closed = False
            def write(self, data):
                self.written += data
            async def drain(self):
                pass
            def close(self):
                self.closed = True
            async def wait_closed(self):
                pass

        client_r = FakeReader(request)
        client_w = FakeWriter()
        tier = m._normalize_tier({
            "enable_thinking": True, "effort": "low",
            "kwarg": "enable_thinking",
            "sampling": {"temperature": 1.0},
        })

        # Fake asyncio.open_connection: capture what the proxy forwards upstream.
        forwarded = {}

        class UpReader:
            async def read(self, n):
                return b""  # immediate EOF -> pump returns, handler finishes
        class UpWriter:
            def __init__(self):
                self.written = b""
            def write(self, data):
                self.written += data
            async def drain(self):
                pass
            def close(self):
                pass
            async def wait_closed(self):
                pass

        up_w = UpWriter()

        async def fake_open(host, port):
            forwarded["host"], forwarded["port"] = host, port
            return UpReader(), up_w

        orig_open = asyncio.open_connection
        asyncio.open_connection = fake_open
        log_lines = []
        orig_print = __builtins__.print if hasattr(__builtins__, "print") else print
        import builtins
        builtins.print = lambda *a, **k: log_lines.append(a)
        try:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(
                    m.handle(client_r, client_w, tier, "127.0.0.1", 8000))
            finally:
                loop.close()
        finally:
            asyncio.open_connection = orig_open
            builtins.print = orig_print

        # the failure must have been logged, not bare-passed
        self.assertTrue(any("inject failed, passing through" in str(a) for a in log_lines),
                        f"expected inject-failure log line, got {log_lines}")
        # the ORIGINAL (un-rewritten) body must be forwarded to upstream
        self.assertIn(body_bytes, up_w.written,
                      "original body was not forwarded unchanged after inject failure")


class NormalizeTier(unittest.TestCase):
    def test_bad_kwarg_rejected(self):
        with self.assertRaises(ValueError):
            m._normalize_tier({"enable_thinking": True, "effort": "low",
                               "kwarg": "bogus"})

    def test_bad_effort_rejected(self):
        with self.assertRaises(ValueError):
            m._normalize_tier({"enable_thinking": True, "effort": "turbo",
                               "kwarg": "enable_thinking"})

    def test_enable_thinking_string_rejected(self):
        # bool("false") is True — a string here must be rejected, not coerced.
        with self.assertRaises(ValueError):
            m._normalize_tier({"enable_thinking": "false", "effort": "low",
                               "kwarg": "enable_thinking"})

    def test_enable_thinking_null_rejected(self):
        with self.assertRaises(ValueError):
            m._normalize_tier({"enable_thinking": None, "effort": "low",
                               "kwarg": "enable_thinking"})

    def test_sampling_collision_with_reasoning_effort_rejected(self):
        with self.assertRaises(ValueError):
            m._normalize_tier({"enable_thinking": True, "effort": None,
                               "kwarg": "enable_thinking",
                               "sampling": {"reasoning_effort": "max"}})

    def test_sampling_collision_with_chat_template_kwargs_rejected(self):
        with self.assertRaises(ValueError):
            m._normalize_tier({"enable_thinking": True, "effort": "low",
                               "kwarg": "enable_thinking",
                               "sampling": {"chat_template_kwargs": {}}})


if __name__ == "__main__":
    unittest.main()
