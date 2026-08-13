"""
Tests for the terminal AI assistant.

Run with:  python tests.py

No API key and no network needed: a fake client stands in for the real one, so
the loop, the output formatting and every error branch can be checked offline.
"""

import io
import os
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import openai

import main

# The SDK's exception classes need a request/response object to construct.
# openai 3.x is built on httpx2, while 1.x and 2.x use httpx, so pick whichever
# is installed rather than assuming one of them.
try:
    import httpx2 as httpx
except ImportError:  # pragma: no cover - depends on the installed openai
    import httpx


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def fake_response(text="Hello!", model="gpt-4o-mini-2024-07-18", usage=True):
    """Build an object shaped like a chat completion response."""
    message = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=message)

    usage_object = None
    if usage:
        usage_object = SimpleNamespace(
            prompt_tokens=12, completion_tokens=34, total_tokens=46
        )

    return SimpleNamespace(choices=[choice], model=model, usage=usage_object)


class FakeClient:
    """Stands in for openai.OpenAI.

    `script` is a list of responses or exceptions; each call takes the next one.
    """

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def api_error(exception_class, status_code, message="boom"):
    """Construct a real SDK exception, which needs httpx objects."""
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(status_code, request=request)
    return exception_class(message, response=response, body=None)


def run_loop(typed_lines, script):
    """Run run_chat() with scripted input and scripted API results.

    Returns (printed_output, fake_client, systemexit_or_None).
    """
    client = FakeClient(script)
    output = io.StringIO()
    raised = None

    with patch("builtins.input", side_effect=typed_lines):
        with redirect_stdout(output):
            try:
                main.run_chat(client, "gpt-4o-mini")
            except SystemExit as exit_call:
                raised = exit_call

    return output.getvalue(), client, raised


# ---------------------------------------------------------------------------
# Configuration and setup
# ---------------------------------------------------------------------------


class TestSystemPrompt(unittest.TestCase):
    """The brief requires a prompt about simple explanations and not inventing facts."""

    def test_mentions_simple_explanations(self):
        self.assertIn("simple", main.SYSTEM_PROMPT.lower())

    def test_tells_the_model_not_to_invent_facts(self):
        prompt = main.SYSTEM_PROMPT.lower()
        self.assertIn("never invent", prompt)
        self.assertIn("not sure", prompt)

    def test_uses_a_small_model_by_default(self):
        self.assertIn("mini", main.DEFAULT_MODEL)


class TestLoadApiKey(unittest.TestCase):
    """A missing or placeholder key must stop the program with a clear message."""

    def _load(self, env_value):
        env = {} if env_value is None else {"OPENAI_API_KEY": env_value}
        output = io.StringIO()
        raised = None

        # load_dotenv is patched out so the real .env cannot influence the test.
        with patch.dict(os.environ, env, clear=True):
            with patch("main.load_dotenv", lambda *a, **k: None):
                with redirect_stdout(output):
                    try:
                        value = main.load_api_key()
                    except SystemExit as exit_call:
                        value = None
                        raised = exit_call

        return value, output.getvalue(), raised

    def test_missing_key_exits_with_code_one(self):
        _, output, raised = self._load(None)
        self.assertIsNotNone(raised)
        self.assertEqual(raised.code, 1)
        self.assertIn("no API key found", output)

    def test_empty_key_exits(self):
        _, _, raised = self._load("")
        self.assertIsNotNone(raised)

    def test_whitespace_only_key_exits(self):
        _, _, raised = self._load("   ")
        self.assertIsNotNone(raised)

    def test_untouched_placeholder_exits(self):
        """A straight copy of .env.example must not be mistaken for a real key."""
        _, output, raised = self._load("your-api-key-here")
        self.assertIsNotNone(raised)
        self.assertIn("no API key found", output)

    def test_message_explains_how_to_fix_it(self):
        _, output, _ = self._load(None)
        self.assertIn(".env.example", output)
        self.assertIn("api-keys", output)

    def test_real_looking_key_is_accepted(self):
        value, _, raised = self._load("sk-proj-abc123")
        self.assertIsNone(raised)
        self.assertEqual(value, "sk-proj-abc123")

    def test_surrounding_whitespace_is_stripped(self):
        value, _, _ = self._load("  sk-proj-abc123  ")
        self.assertEqual(value, "sk-proj-abc123")


# ---------------------------------------------------------------------------
# Messages and usage formatting
# ---------------------------------------------------------------------------


class TestBuildMessages(unittest.TestCase):
    def test_system_prompt_comes_first(self):
        messages = main.build_messages([], "hi")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], main.SYSTEM_PROMPT)

    def test_question_comes_last(self):
        messages = main.build_messages([], "what is a list?")
        self.assertEqual(messages[-1], {"role": "user", "content": "what is a list?"})

    def test_history_sits_between_them(self):
        history = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer"},
        ]
        messages = main.build_messages(history, "second")
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[1]["content"], "first")


class TestDescribeUsage(unittest.TestCase):
    def test_formats_all_three_counts(self):
        line = main.describe_usage(fake_response())
        self.assertIn("prompt 12", line)
        self.assertIn("answer 34", line)
        self.assertIn("total 46", line)

    def test_returns_none_when_usage_is_absent(self):
        """usage is optional, so its absence must not crash anything."""
        self.assertIsNone(main.describe_usage(fake_response(usage=False)))

    def test_handles_a_partial_usage_object(self):
        response = SimpleNamespace(usage=SimpleNamespace(total_tokens=99))
        self.assertEqual(main.describe_usage(response), "total 99")


# ---------------------------------------------------------------------------
# The chat loop
# ---------------------------------------------------------------------------


class TestChatLoop(unittest.TestCase):
    def test_prints_answer_model_and_tokens(self):
        output, _, _ = run_loop(["What is a variable?", "exit"], [fake_response("A box.")])

        self.assertIn("Assistant: A box.", output)
        self.assertIn("model: gpt-4o-mini-2024-07-18", output)  # from the response
        self.assertIn("tokens: prompt 12, answer 34, total 46", output)

    def test_missing_usage_still_prints_the_answer(self):
        output, _, _ = run_loop(["hi", "exit"], [fake_response("Hey.", usage=False)])
        self.assertIn("Assistant: Hey.", output)
        self.assertIn("model:", output)
        self.assertNotIn("tokens:", output)

    def test_every_exit_command_works(self):
        for command in ("exit", "quit", "q", "bye", "EXIT", "Quit"):
            output, client, _ = run_loop([command], [])
            self.assertIn("Goodbye!", output)
            self.assertEqual(len(client.calls), 0, f"{command} should not call the API")

    def test_empty_input_is_ignored(self):
        output, client, _ = run_loop(["", "   ", "exit"], [])
        self.assertEqual(len(client.calls), 0)
        self.assertIn("Goodbye!", output)

    def test_ctrl_d_exits_cleanly(self):
        output, _, raised = run_loop(EOFError(), [])
        self.assertIn("Goodbye!", output)
        self.assertIsNone(raised)  # no traceback, no non-zero exit

    def test_ctrl_c_exits_cleanly(self):
        output, _, raised = run_loop(KeyboardInterrupt(), [])
        self.assertIn("Goodbye!", output)
        self.assertIsNone(raised)

    def test_history_is_sent_on_the_next_question(self):
        _, client, _ = run_loop(
            ["first question", "second question", "exit"],
            [fake_response("first answer"), fake_response("second answer")],
        )

        second_call = client.calls[1]["messages"]
        contents = [message["content"] for message in second_call]
        self.assertIn("first question", contents)
        self.assertIn("first answer", contents)
        self.assertEqual(second_call[-1]["content"], "second question")

    def test_reset_clears_the_history(self):
        _, client, output = run_loop(
            ["first", "reset", "second", "exit"],
            [fake_response("a1"), fake_response("a2")],
        )

        # After reset the second request carries only system + the new question.
        self.assertEqual(len(client.calls[1]["messages"]), 2)

    def test_history_is_trimmed_to_the_limit(self):
        turns = main.MAX_HISTORY_TURNS + 5
        typed = [f"question {i}" for i in range(turns)] + ["exit"]
        script = [fake_response(f"answer {i}") for i in range(turns)]

        _, client, _ = run_loop(typed, script)

        # system prompt + at most MAX_HISTORY_TURNS turns + the new question
        largest = max(len(call["messages"]) for call in client.calls)
        self.assertLessEqual(largest, 1 + main.MAX_HISTORY_TURNS * 2 + 1)

    def test_the_request_uses_the_configured_model(self):
        _, client, _ = run_loop(["hi", "exit"], [fake_response()])
        self.assertEqual(client.calls[0]["model"], "gpt-4o-mini")


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestExplainError(unittest.TestCase):
    """Each error type must produce its own message and the right fatality."""

    def test_authentication_error_is_fatal(self):
        message, fatal = main.explain_error(api_error(openai.AuthenticationError, 401))
        self.assertIn("Authentication failed", message)
        self.assertTrue(fatal)

    def test_rate_limit_error_is_not_fatal(self):
        message, fatal = main.explain_error(api_error(openai.RateLimitError, 429))
        self.assertIn("Rate limit", message)
        self.assertFalse(fatal)

    def test_connection_error(self):
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        message, fatal = main.explain_error(
            openai.APIConnectionError(message="nope", request=request)
        )
        self.assertIn("Could not reach the API", message)
        self.assertFalse(fatal)

    def test_timeout_error_is_reported_as_a_timeout(self):
        """APITimeoutError subclasses APIConnectionError, so order matters."""
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        message, _ = main.explain_error(openai.APITimeoutError(request=request))
        self.assertIn("timed out", message)

    def test_model_not_found_is_fatal(self):
        message, fatal = main.explain_error(api_error(openai.NotFoundError, 404))
        self.assertIn("model was not found", message)
        self.assertTrue(fatal)

    def test_permission_denied_is_fatal(self):
        message, fatal = main.explain_error(api_error(openai.PermissionDeniedError, 403))
        self.assertIn("Permission denied", message)
        self.assertTrue(fatal)

    def test_bad_request_is_not_fatal(self):
        message, fatal = main.explain_error(api_error(openai.BadRequestError, 400))
        self.assertIn("rejected the request", message)
        self.assertFalse(fatal)

    def test_other_status_errors_report_the_code(self):
        message, fatal = main.explain_error(api_error(openai.InternalServerError, 503))
        self.assertIn("503", message)
        self.assertFalse(fatal)

    def test_unexpected_exception_names_its_type(self):
        message, fatal = main.explain_error(ValueError("something odd"))
        self.assertIn("ValueError", message)
        self.assertIn("something odd", message)
        self.assertFalse(fatal)


class TestErrorsInsideTheLoop(unittest.TestCase):
    def test_rate_limit_lets_the_user_try_again(self):
        output, client, raised = run_loop(
            ["first try", "second try", "exit"],
            [api_error(openai.RateLimitError, 429), fake_response("worked")],
        )

        self.assertIn("Rate limit", output)
        self.assertIn("Assistant: worked", output)  # loop survived
        self.assertIsNone(raised)

    def test_authentication_error_stops_the_program(self):
        output, _, raised = run_loop(
            ["a question", "never reached"],
            [api_error(openai.AuthenticationError, 401)],
        )

        self.assertIn("Authentication failed", output)
        self.assertIsNotNone(raised)
        self.assertEqual(raised.code, 1)

    def test_unexpected_error_does_not_crash_the_loop(self):
        output, _, raised = run_loop(
            ["boom", "again", "exit"],
            [RuntimeError("kaboom"), fake_response("fine now")],
        )

        self.assertIn("RuntimeError", output)
        self.assertIn("Assistant: fine now", output)
        self.assertIsNone(raised)


if __name__ == "__main__":
    unittest.main(verbosity=2)
