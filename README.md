# Terminal AI Assistant

A small command-line AI assistant in Python. It loads an API key from a `.env` file, sends your
questions to a model with a fixed system prompt, and prints the answer along with the model name
and token usage.

## Files

| File | Purpose |
|---|---|
| `main.py` | The assistant |
| `.env.example` | Template for your key — copy this to `.env` |
| `.env` | Your real key. **Ignored by Git, never committed** |
| `.gitignore` | Keeps `.env` out of the repository |
| `requirements.txt` | `openai` and `python-dotenv` |
| `tests.py` | 38 tests that run without an API key |

## Setup

**1. Clone and enter the project**

```bash
git clone https://github.com/ezitounioussama/terminal-ai-assistant.git
cd terminal-ai-assistant
```

**2. Create a virtual environment and install the dependencies**

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**3. Add your API key**

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder with your real key:

```
OPENAI_API_KEY=sk-proj-your-real-key-here
MODEL=gpt-4o-mini
```

Get a key from https://platform.openai.com/api-keys. `.env` is listed in `.gitignore`, so it
never reaches GitHub.

## Running it

```bash
python main.py
```

Type a question and press Enter. Type `exit`, `quit`, `q` or `bye` to leave, or `reset` to clear
the conversation. `Ctrl+C` and `Ctrl+D` also exit cleanly.

## Example interaction

```
============================================================
            TERMINAL AI ASSISTANT
============================================================
  Model : gpt-4o-mini
  Type your question and press Enter.
  Commands: 'exit' to quit, 'reset' to clear the conversation.
------------------------------------------------------------

You: What is a variable?

Assistant: A variable is a name you attach to a value so you can use it later.

    age = 25
    name = "Sara"

Here `age` and `name` are variables. You can read them or change them
later by assigning a new value.

  [model: gpt-4o-mini-2024-07-18 | tokens: prompt 96, answer 71, total 167]

You: Can it hold more than one value?

Assistant: Yes. A list holds several values in order, under one name:

    scores = [10, 20, 30]

You reach an item by its position, starting at 0: scores[0] is 10.

  [model: gpt-4o-mini-2024-07-18 | tokens: prompt 181, answer 58, total 239]

You: exit
Goodbye!
```

Note the second question — "Can it hold more than one value?" — makes sense only because the
previous turn is sent along with it. The model has no memory between calls, so `main.py` keeps
the recent history and includes it in each request.

> The transcript above was captured from a run with a stubbed client, since no API key was
> available while writing this. The layout, model line and token line are exactly what the
> program prints; only the assistant's wording will differ on your machine.

## Checklist

| Required | Where |
|---|---|
| Project folder | this repository |
| `.env`, `.env.example`, `.gitignore`, `requirements.txt`, `README.md`, `main.py` | all present |
| Real key only in `.env`, ignored by Git | `.gitignore` line 2 |
| Load the key with python-dotenv, stop clearly if missing | `load_api_key()` |
| AI client with a small model | `create_client()`, default `gpt-4o-mini` |
| System prompt: explain simply, do not invent facts | `SYSTEM_PROMPT` |
| Loop with an exit command | `run_chat()`, exits on `exit` / `quit` / `q` / `bye` |
| Print answer, model name, token usage | after each reply |
| Error handling: missing key, auth, rate limits, unexpected | `load_api_key()` and `explain_error()` |
| Setup steps, how to run, example interaction | this file |

## Error handling

| Situation | What happens |
|---|---|
| No key, or `.env.example` copied but not edited | Prints the three steps to fix it, exits with code 1 |
| Key rejected (401) | Explains the key was refused, points at the keys page, exits |
| Rate limit or quota (429) | Says to wait, **keeps running** so you can retry |
| No internet / timeout | Explains the connection failed, keeps running |
| Model name wrong (404) | Says to check `MODEL` in `.env`, exits |
| Anything unexpected | Prints the exception type and message, keeps running |

The distinction matters: a bad key or a misspelled model will never fix itself, so those stop the
program. A rate limit or a dropped connection is temporary, so those return you to the prompt
with your conversation intact.

## Testing

```bash
python tests.py
```

**38 tests, all passing** — no API key and no network required. A fake client stands in for the
real one, so every branch can be checked offline: the output format, both exit paths, history
being sent and trimmed, `reset`, empty input, a missing `usage` field, and each of the nine error
types mapped to the right message and the right fatality.

The authentication path was also verified against the live API using a deliberately invalid key,
which produced the friendly message and exit code 1 rather than a traceback.

## Notes on the implementation

**The placeholder counts as missing.** If `.env` still contains `your-api-key-here`, the program
stops with the setup instructions instead of sending it and getting a confusing 401 back.

**The model name is read from the response, not from the config.** The API resolves an alias like
`gpt-4o-mini` to a dated build such as `gpt-4o-mini-2024-07-18`, and the printed line shows what
actually served the request.

**Token usage is treated as optional.** `usage` is not guaranteed to be present, so every field
is read with `getattr` and the line is skipped entirely if it's absent — the answer still prints.

**Error checks run most-specific first.** `APITimeoutError` is a subclass of
`APIConnectionError`, and every SDK error inherits from `APIError`, so a broad check placed early
would swallow the specific ones and report the wrong cause.

**Compatibility.** Tested against `openai` 3.0.0. The `chat.completions` call and the exception
classes used here exist in 1.x through 3.x. One version difference does show up in the tests:
openai 3.x is built on `httpx2` while earlier versions use `httpx`, so `tests.py` imports
whichever is installed.

---

Author: **Oussama Ezitouni**
