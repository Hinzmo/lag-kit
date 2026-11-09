"""Suite-wide settings."""
import os

# `lag serve` loads the chat model into Ollama at startup. Tests that start the app
# must not reach a real Ollama for that.
os.environ.setdefault("LAG_WARM", "0")

# FastAPI's TestClient addresses the app as http://testserver. The console only serves
# names it knows (web._only_known_callers), so the suite names that one.
os.environ.setdefault("LAG_ALLOWED_HOSTS", "testserver")
