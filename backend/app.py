from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from groq import Groq
import os
import json
from datetime import datetime, timezone
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

load_dotenv()

# ✅ Initialize Flask app FIRST
app = Flask(__name__)
CORS(app)

HAS_VALID_KEY = False
groq_client = None

# Model priority list (fastest/free-tier first) — updated May 2026
GROQ_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "llama4-scout-17b-16e-instruct",
]
CURRENT_MODEL = GROQ_MODELS[0]


def init_groq():
    global groq_client, HAS_VALID_KEY, CURRENT_MODEL
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        print("❌ GROQ_API_KEY not found in environment.")
        HAS_VALID_KEY = False
        return False

    key = key.strip().strip("'\"")
    try:
        groq_client = Groq(api_key=key)
        # Quick validation call
        groq_client.chat.completions.create(
            model=CURRENT_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
        )
        HAS_VALID_KEY = True
        print(f"✅ Groq initialized with model: {CURRENT_MODEL}")
        return True
    except Exception as e:
        err = str(e).lower()
        print(f"❌ Groq init error: {e}")
        if any(x in err for x in ["401", "unauthorized", "invalid", "api_key"]):
            HAS_VALID_KEY = False
        else:
            # Key is valid but quota/network issue
            HAS_VALID_KEY = True
        return HAS_VALID_KEY


init_groq()


# 🗄️ MongoDB Connection
MONGO_URI = os.environ.get("MONGO_URI")
db_available = False
collection = None

try:
    if MONGO_URI:
        client_db = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
        client_db.admin.command("ping")
        db = client_db["chatbot"]
        collection = db["history"]
        db_available = True
        print("✅ Connected to MongoDB Atlas")
except Exception as e:
    print(f"⚠️  MongoDB not available, using local mock storage: {e}")
    db_available = False

LOCAL_HISTORY_FILE = "chat_history.json"


def save_to_history(data):
    if db_available and collection is not None:
        try:
            collection.insert_one(data.copy())
            return
        except Exception:
            pass
    # Fallback to local file
    history = []
    if os.path.exists(LOCAL_HISTORY_FILE):
        try:
            with open(LOCAL_HISTORY_FILE, "r") as f:
                history = json.load(f)
        except Exception:
            history = []
    history.append(data)
    with open(LOCAL_HISTORY_FILE, "w") as f:
        json.dump(history, f)


def get_history(email):
    if db_available and collection is not None:
        try:
            query = {"email": email} if email else {}
            return list(collection.find(query, {"_id": 0}))
        except Exception:
            pass
    if os.path.exists(LOCAL_HISTORY_FILE):
        try:
            with open(LOCAL_HISTORY_FILE, "r") as f:
                history = json.load(f)
                if email:
                    return [c for c in history if c.get("email") == email]
                return history
        except Exception:
            return []
    return []


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    reraise=True,
)
def generate_with_retry(prompt, model_name):
    if not HAS_VALID_KEY or groq_client is None:
        raise ValueError("No valid Groq API key configured.")
    response = groq_client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.3,
    )
    return response.choices[0].message.content


def generate_with_fallback(prompt):
    global CURRENT_MODEL

    for model_name in GROQ_MODELS:
        try:
            print(f"🔄 Attempting generation with model: {model_name}")
            result = generate_with_retry(prompt, model_name)
            CURRENT_MODEL = model_name
            return result
        except Exception as e:
            err = str(e).lower()
            print(f"⚠️  Model {model_name} failed: {e}")
            # Don't retry auth errors
            if any(x in err for x in ["401", "unauthorized", "invalid", "api_key", "no valid"]):
                raise e
            # Otherwise try next model
            continue

    raise RuntimeError("All Groq models failed. Please try again later.")


# ─────────────────────────────────────────────
# 🔹 HEALTH API
# ─────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "online",
            "db": "connected" if db_available else "fallback",
            "api": "configured" if HAS_VALID_KEY else "missing",
            "provider": "groq",
            "model": CURRENT_MODEL,
            "version": "3.0-groq",
        }
    )


# ─────────────────────────────────────────────
# 🔹 FACT CHECK API
# ─────────────────────────────────────────────
@app.route("/fact-check", methods=["POST"])
def fact_check():
    try:
        data = request.json
        statement = data.get("statement", "")
        email = data.get("email", "anonymous")
        print(f"📨 Request from {email}: {statement}")

        if not statement.strip():
            return jsonify({"error": "Please provide a statement to check."}), 400

        prompt = f"""You are an AI fact-checker.

Give output in this EXACT format (no extra text, no follow-up questions):
Verdict: True / False / Uncertain
Confidence: (percentage)
Justification: 2-3 lines explanation

Statement: {statement}"""

        # Re-init if key was missing at startup
        if not HAS_VALID_KEY or groq_client is None:
            if not init_groq():
                return jsonify(
                    {
                        "error": "Fact-checking service is not configured (Groq API key missing or invalid).",
                        "status": "error",
                    }
                ), 503

        try:
            result = generate_with_fallback(prompt)
        except Exception as e:
            error_msg = str(e)
            print(f"❌ AI Generation failed: {error_msg}")

            lower_msg = error_msg.lower()
            if any(x in lower_msg for x in ["quota", "429", "rate_limit", "limit exceeded"]):
                friendly = "API quota exceeded. Please wait a moment and try again."
            elif any(x in lower_msg for x in ["no valid", "api_key"]):
                friendly = "Groq API key is missing or invalid."
            elif any(x in lower_msg for x in ["401", "unauthorized", "invalid"]):
                friendly = "Invalid Groq API key."
            else:
                friendly = f"Fact-checking service unavailable: {error_msg}"

            return jsonify({"error": friendly, "details": error_msg, "status": "error"}), 500

        # 💾 Save to history
        save_to_history(
            {
                "email": email,
                "statement": statement,
                "response": result,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        return jsonify({"result": result})

    except Exception as e:
        print(f"❌ Unhandled error: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


# ─────────────────────────────────────────────
# 🔹 HISTORY API
# ─────────────────────────────────────────────
@app.route("/history", methods=["GET"])
def history():
    email = request.args.get("email")
    chats = get_history(email)
    return jsonify(chats)


@app.route("/clear-history", methods=["POST"])
def clear_history():
    try:
        data = request.json
        email = data.get("email", "anonymous")

        if db_available and collection is not None:
            query = {"email": email} if email else {}
            collection.delete_many(query)

        if os.path.exists(LOCAL_HISTORY_FILE):
            if email and email != "anonymous":
                try:
                    with open(LOCAL_HISTORY_FILE, "r") as f:
                        hist = json.load(f)
                    new_hist = [c for c in hist if c.get("email") != email]
                    with open(LOCAL_HISTORY_FILE, "w") as f:
                        json.dump(new_hist, f)
                except Exception:
                    pass
            else:
                os.remove(LOCAL_HISTORY_FILE)

        return jsonify({"status": "success", "message": "History cleared"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 🚀 Run server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
