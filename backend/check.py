"""
check.py — Verify your Groq API key and list available models.
"""
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

API_KEY = os.environ.get("GROQ_API_KEY")
if not API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set!")

client = Groq(api_key=API_KEY)

print("✅ Groq client initialized.\n")
print("Available models:")
models = client.models.list()
for m in models.data:
    print(f"  - {m.id}")
