"""
HR Policy Chatbot — AI Powered (Ollama + Llama 3.2)
=====================================================
An AI-powered Q&A chatbot that answers questions based strictly
on the contents of a company HR policy document.

How it works:
  1. Loads and splits the HR document into sections
  2. Uses TF-IDF to find the most relevant sections for your question
  3. Sends those sections + your question to a local AI model (Ollama)
  4. The AI generates a natural, conversational answer

Requirements:
  - Install Ollama from https://ollama.com
  - Run: ollama pull llama3.2
  - Run: ollama serve   (keep this open in a separate window)
  - Then run this script: python chatbot_ai.py
"""

import re
import math
import textwrap
import urllib.request
import urllib.error
import json

DOCUMENT_PATH = "hr_policy.txt"
OLLAMA_URL    = "http://localhost:11434/api/generate"
OLLAMA_MODEL  = "llama3.2"
TOP_K         = 3

STOPWORDS = {
    "what", "is", "the", "a", "an", "how", "many", "do", "i", "can",
    "are", "for", "in", "of", "to", "does", "when", "who", "my", "me",
    "we", "our", "and", "or", "be", "if", "on", "at", "it", "this",
    "that", "there", "about", "with", "get", "have", "has", "was",
    "will", "would", "from", "after", "their", "they",
}


# ── DOCUMENT LOADING & CHUNKING ───────────────────────────────────────────────

def load_document(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def tokenize(text):
    words = re.sub(r"[^\w\s]", "", text.lower()).split()
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def split_into_chunks(text):
    sections = re.split(r"━+", text)
    return [s.strip() for s in sections if len(s.strip()) > 40]


# ── TF-IDF RETRIEVAL ──────────────────────────────────────────────────────────

def build_tfidf(chunks):
    tokenized = [tokenize(c) for c in chunks]
    N = len(chunks)

    df = {}
    for tokens in tokenized:
        for word in set(tokens):
            df[word] = df.get(word, 0) + 1

    idf = {word: math.log(N / count) for word, count in df.items()}

    vectors = []
    for tokens in tokenized:
        tf = {}
        for word in tokens:
            tf[word] = tf.get(word, 0) + 1
        total = len(tokens) or 1
        vec = {w: (c / total) * idf.get(w, 0) for w, c in tf.items()}
        vectors.append(vec)

    return vectors, idf


def cosine_similarity(vec_a, vec_b):
    keys = set(vec_a) & set(vec_b)
    if not keys:
        return 0.0
    dot    = sum(vec_a[k] * vec_b[k] for k in keys)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def retrieve(query, chunks, vectors, idf, k=TOP_K):
    q_tokens = tokenize(query)
    total = len(q_tokens) or 1
    q_vec = {w: (q_tokens.count(w) / total) * idf.get(w, 0)
             for w in set(q_tokens)}
    scores = [(cosine_similarity(q_vec, vec), chunk)
              for vec, chunk in zip(vectors, chunks)]
    scores.sort(key=lambda x: x[0], reverse=True)
    return [chunk for score, chunk in scores[:k] if score > 0]


# ── OLLAMA AI ANSWER ──────────────────────────────────────────────────────────

def check_ollama():
    """Check if Ollama is running before we start."""
    try:
        req = urllib.request.Request("http://localhost:11434")
        urllib.request.urlopen(req, timeout=3)
        return True
    except Exception:
        return False


def ask_ollama(question, context_chunks):
    """
    Send the question + relevant policy sections to the local AI model.
    The model is instructed to answer ONLY from the provided context
    so it doesn't make things up.
    """
    context = "\n\n---\n\n".join(context_chunks)

    prompt = f"""You are a formal HR assistant for Nexagen Biosciences.
STRICT RULES YOU MUST FOLLOW:
- Answer using ONLY the HR policy excerpts provided. Do not use outside knowledge.
- NEVER address the user by any name or term. Do not say "king", "buddy", "friend", "sure", or any greeting.
- Start your answer directly with the information. No preamble.
- If the answer is not in the excerpts, say: "That information is not covered in the HR policy."
- Maximum 3-4 sentences.


HR POLICY EXCERPTS:
{context}

EMPLOYEE QUESTION:
{question}

ANSWER:"""

    payload = json.dumps({
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            response = data.get("response", "").strip()
            for word in ["king", "buddy", "friend", "bro", "mate", "chief", "boss"]:
                response = re.sub(rf"\b{word}\b", "", response, flags=re.IGNORECASE)
            response = re.sub(r"\s{2,}", " ", response).strip(" ,!")
            return response
    except urllib.error.URLError:
        return ("⚠️  Could not reach Ollama. Make sure it is running:\n"
                "     1. Open a new Command Prompt\n"
                "     2. Run: ollama serve")
    except Exception as e:
        return f"⚠️  Error: {e}"


# ── UI ─────────────────────────────────────────────────────────────────────────

def print_banner():
    print()
    print("╔" + "═" * 58 + "╗")
    print("║   🤖  NEXAGEN BIOSCIENCES — AI HR CHATBOT            ║")
    print("║   Powered by Llama 3.2 running locally on your PC.  ║")
    print("║   Ask me anything about company HR policies.         ║")
    print("║   Type 'quit' to exit.                               ║")
    print("╚" + "═" * 58 + "╝")
    print()


def main():
    # Check Ollama is running first
    print("\n⏳  Checking Ollama connection...")
    if not check_ollama():
        print("\n❌  Ollama is not running! Please:")
        print("    1. Install Ollama from https://ollama.com")
        print("    2. Open Command Prompt and run: ollama pull llama3.2")
        print("    3. Then run: ollama serve")
        print("    4. Come back and run this script again.\n")
        return

    print("✅  Ollama connected!")
    print("⏳  Loading HR policy and building search index...")

    text    = load_document(DOCUMENT_PATH)
    chunks  = split_into_chunks(text)
    vectors, idf = build_tfidf(chunks)

    print(f"✅  Ready! Indexed {len(chunks)} policy sections.\n")
    print_banner()

    examples = [
        "How many PTO days do I get in my first year?",
        "Can I work from home every day?",
        "What is the 401k match and when am I vested?",
        "How long is parental leave for a new dad?",
        "When are performance reviews and how do raises work?",
        "What do I need to do if I want to resign?",
    ]
    print("  💡  Try asking:")
    for ex in examples:
        print(f"      • {ex}")
    print()

    while True:
        try:
            query = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  👋  Goodbye!\n")
            break

        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            print("\n  👋  Goodbye!\n")
            break

        print("  🤖  Thinking...", end="\r", flush=True)
        print(" " * 30, end="\r", flush=True)

        # Step 1: find relevant sections
        relevant = retrieve(query, chunks, vectors, idf)

        if not relevant:
            print("  🤖  Bot:")
            print("      I couldn't find anything related to that in the HR policy.\n")
            print("  " + "─" * 56 + "\n")
            continue

        # Step 2: ask the AI
        answer = ask_ollama(query, relevant)

        wrapped = textwrap.fill(
            answer, width=56,
            initial_indent="     ",
            subsequent_indent="     ",
        )
        print(f"  🤖  Bot:\n{wrapped}\n")
        print("  " + "─" * 56 + "\n")


if __name__ == "__main__":
    main()
