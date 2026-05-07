"""
test_best_model.py — Find the fastest/best available Groq model for your key.
"""
import os
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("❌ No GROQ_API_KEY found.")
    exit(1)

client = Groq(api_key=GROQ_API_KEY)

models_to_test = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "llama4-scout-17b-16e-instruct",
]

print("Testing Groq models...\n")
results = []

for m_name in models_to_test:
    print(f"Testing {m_name}...")
    try:
        start = time.time()
        response = client.chat.completions.create(
            model=m_name,
            messages=[{"role": "user", "content": "Ping"}],
            max_tokens=10,
        )
        elapsed = time.time() - start
        text = response.choices[0].message.content.strip()
        print(f"  ✅ SUCCESS in {elapsed:.2f}s: {text}")
        results.append((elapsed, m_name))
    except Exception as e:
        print(f"  ❌ FAILED: {e}")

if results:
    results.sort()
    print(f"\n🏆 Fastest model: {results[0][1]} ({results[0][0]:.2f}s)")
