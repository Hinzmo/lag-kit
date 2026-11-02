"""Suite-wide settings."""
import os

# `lag serve` loads the chat model into Ollama at startup. Tests that start the app
# must not reach a real Ollama for that.
os.environ.setdefault("LAG_WARM", "0")
