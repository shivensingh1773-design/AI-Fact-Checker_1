"""
test_groq.py — Test Groq API with multiple models.
"""
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
print(f"Key found: {bool(GROQ_API_KEY)}")

if not GROQ_API_KEY:
    print("No API key found in .env")
    exit(1)

client = Groq(api_key=GROQ_API_KEY)

models_to_try = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "llama4-scout-17b-16e-instruct",
]

for model_name in models_to_try:
    print(f"\nTrying model: {model_name}")
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Say hello in one sentence."}],
            max_tokens=50,
        )
        print(f"  ✅ SUCCESS: {response.choices[0].message.content.strip()}")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
