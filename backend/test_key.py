"""
test_key.py — Quick test to verify your Groq API key works.
"""
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

API_KEY = os.environ.get("GROQ_API_KEY")
if not API_KEY:
    print("❌ No GROQ_API_KEY found in environment.")
    exit(1)

print(f"API Key found: {API_KEY[:5]}...{API_KEY[-5:]}")

try:
    client = Groq(api_key=API_KEY)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": "test"}],
        max_tokens=10,
    )
    print("✅ Success! Response received:")
    print(response.choices[0].message.content)
except Exception as e:
    print(f"❌ Error: {e}")
