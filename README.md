# Terminal AI Assistant

A command-line AI assistant in Python. Ask it a question, it prints the answer along with the
model that served it and the tokens it cost. Follow-up questions work, because the program keeps
the recent turns and resends them — the model itself remembers nothing between calls.

The part worth the effort was the failure handling. A bad key and a rate limit are not the same
kind of problem: one will never fix itself, so the program exits with instructions; the other is
temporary, so it says to wait and hands you back the prompt with your conversation intact.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then paste your real key

python main.py
python tests.py               # 38 tests, no API key needed
```

## Also in this repo

- **[NOTES.md](NOTES.md)** — setup in full, an example session, the error-handling table, and the
  implementation notes (including the openai 3.x / `httpx2` difference)

---

Author: **Oussama Ezitouni**
