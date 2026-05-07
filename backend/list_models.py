"""
list_models.py — List all available Groq models for your API key.
"""
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("❌ No GROQ_API_KEY found.")
    exit(1)

client = Groq(api_key=GROQ_API_KEY)
models = client.models.list()

print("Available Groq models:")
for m in models.data:
    print(f"  {m.id}")
