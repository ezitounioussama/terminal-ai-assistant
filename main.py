"""
Terminal AI Assistant
=====================

A small command-line assistant. It loads an API key from a .env file, sends
your questions to an OpenAI model with a fixed system prompt, and prints the
answer along with the model name and token usage.

Run with:  python main.py
Exit with: exit, quit, or Ctrl+C
"""

import os
import sys

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# A small, inexpensive model is plenty for question-and-answer practice.
# Override it in .env with MODEL=... to try a different one.
DEFAULT_MODEL = "gpt-4o-mini"

# The system prompt sets the assistant's behaviour for every turn. The two
# instructions that matter most here: explain simply, and admit uncertainty
# rather than inventing an answer.
SYSTEM_PROMPT = """You are a patient teaching assistant for someone learning to code.

Follow these rules in every answer:
- Explain concepts in simple, plain language. Assume the reader is a beginner.
- Keep answers short by default: a few sentences, or a short list.
- Use a small code example when it makes the idea clearer.
- If you are not sure about something, say so plainly. Never invent facts,
  version numbers, function names, or statistics. Guessing is worse than
  admitting the gap, because the reader cannot tell the difference.
- If a question is ambiguous, state the assumption you are answering under.
"""

# How many previous turns to keep, so follow-up questions have context without
# the request growing forever. Each turn is one question and one answer.
MAX_HISTORY_TURNS = 10

EXIT_COMMANDS = {"exit", "quit", "q", "bye"}


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def load_api_key():
    """Read OPENAI_API_KEY from .env and stop the program if it is missing.

    load_dotenv() reads the .env file in the project folder and puts its values
    into the environment, so os.getenv can see them.

    A placeholder counts as missing: a fresh copy of .env.example contains
    'your-api-key-here', and failing here with a clear message is much friendlier
    than letting the API reject it later with a 401.
    """
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    placeholders = {"your-api-key-here", "sk-your-key-here", "changeme", "..."}

    if not api_key or api_key in placeholders:
        print("Error: no API key found.")
        print()
        print("Fix it in three steps:")
        print("  1. Copy the example file:   cp .env.example .env")
        print("  2. Open .env and put your real key after OPENAI_API_KEY=")
        print("  3. Get a key from:          https://platform.openai.com/api-keys")
        print()
        print("The .env file is listed in .gitignore, so your key stays private.")
        sys.exit(1)

    return api_key


def create_client(api_key):
    """Build the OpenAI client.

    The import sits inside the function so that a missing dependency produces a
    readable install hint instead of a traceback at startup.
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("Error: the 'openai' package is not installed.")
        print("Install the dependencies with:  pip install -r requirements.txt")
        sys.exit(1)

    return OpenAI(api_key=api_key)


# ---------------------------------------------------------------------------
# Talking to the model
# ---------------------------------------------------------------------------


def build_messages(history, question):
    """Assemble the message list for one request.

    The system prompt always comes first, then the recent conversation, then the
    new question. Sending the history is what lets follow-ups like "why?" make
    sense to the model, which has no memory of its own between calls.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": question})
    return messages


def describe_usage(response):
    """Return a readable token-usage line, or None if the API did not send one.

    usage is not guaranteed to be present, so every field is checked with
    getattr rather than assumed.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return None

    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)

    parts = []
    if prompt_tokens is not None:
        parts.append(f"prompt {prompt_tokens}")
    if completion_tokens is not None:
        parts.append(f"answer {completion_tokens}")
    if total_tokens is not None:
        parts.append(f"total {total_tokens}")

    return ", ".join(parts) if parts else None


def ask_model(client, model, history, question):
    """Send one question and return the response object."""
    return client.chat.completions.create(
        model=model,
        messages=build_messages(history, question),
        temperature=0.3,  # low temperature keeps explanations consistent
    )


def explain_error(error):
    """Turn an SDK exception into a message a user can act on.

    Returns (message, is_fatal). Fatal errors end the program; everything else
    just ends the current turn so the user can try again.

    The exception classes are checked from most specific to least, because they
    all inherit from APIError and an early broad match would swallow the
    specific ones.
    """
    import openai

    if isinstance(error, openai.AuthenticationError):
        return (
            "Authentication failed — the API key was rejected.\n"
            "  Check OPENAI_API_KEY in .env for typos or extra spaces, and make\n"
            "  sure the key has not been revoked at platform.openai.com/api-keys.",
            True,  # a bad key will not fix itself, so stop
        )

    if isinstance(error, openai.RateLimitError):
        return (
            "Rate limit or quota reached.\n"
            "  Wait a few seconds and try again. If it keeps happening, check your\n"
            "  usage and billing at platform.openai.com/usage.",
            False,
        )

    if isinstance(error, openai.APITimeoutError):
        return ("The request timed out. Try again.", False)

    if isinstance(error, openai.APIConnectionError):
        return (
            "Could not reach the API — this usually means no internet connection.",
            False,
        )

    if isinstance(error, openai.NotFoundError):
        return (
            f"The model was not found. Check the MODEL value in .env.\n"
            f"  Details: {error}",
            True,
        )

    if isinstance(error, openai.PermissionDeniedError):
        return (
            "Permission denied — this key is not allowed to use that model.",
            True,
        )

    if isinstance(error, openai.BadRequestError):
        return (f"The API rejected the request: {error}", False)

    if isinstance(error, openai.APIStatusError):
        return (
            f"The API returned an error (status {error.status_code}).\n"
            f"  Details: {error}",
            False,
        )

    if isinstance(error, openai.OpenAIError):
        return (f"OpenAI SDK error: {error}", False)

    # Anything else is unexpected; show the type so it can be reported.
    return (f"Unexpected {type(error).__name__}: {error}", False)


# ---------------------------------------------------------------------------
# The chat loop
# ---------------------------------------------------------------------------


def print_banner(model):
    print("=" * 60)
    print("            TERMINAL AI ASSISTANT")
    print("=" * 60)
    print(f"  Model : {model}")
    print("  Type your question and press Enter.")
    print("  Commands: 'exit' to quit, 'reset' to clear the conversation.")
    print("-" * 60)


def run_chat(client, model):
    """Ask, answer, repeat — until the user exits."""
    history = []

    while True:
        # Ctrl+C or Ctrl+D should end the session quietly, not with a traceback.
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return

        if not question:
            continue  # empty line: just show the prompt again

        if question.lower() in EXIT_COMMANDS:
            print("Goodbye!")
            return

        if question.lower() == "reset":
            history.clear()
            print("Conversation cleared.")
            continue

        try:
            response = ask_model(client, model, history, question)
        except Exception as error:  # noqa: BLE001 - sorted out by explain_error
            message, is_fatal = explain_error(error)
            print(f"\n{message}")
            if is_fatal:
                sys.exit(1)
            continue

        answer = response.choices[0].message.content

        print(f"\nAssistant: {answer}")

        # The model name comes from the response, not from our variable, because
        # the API may resolve an alias to a dated version such as
        # gpt-4o-mini-2024-07-18.
        print(f"\n  [model: {getattr(response, 'model', model)}", end="")

        usage_line = describe_usage(response)
        if usage_line:
            print(f" | tokens: {usage_line}", end="")
        print("]")

        # Remember this turn so follow-up questions have context, trimming the
        # oldest turns once the limit is reached.
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        del history[: -MAX_HISTORY_TURNS * 2]


def main():
    api_key = load_api_key()
    model = os.getenv("MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL

    client = create_client(api_key)

    print_banner(model)
    run_chat(client, model)


if __name__ == "__main__":
    main()
