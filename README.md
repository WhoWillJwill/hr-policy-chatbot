# 🤖 HR Policy Chatbot

An AI-powered Q&A chatbot that answers employee questions based strictly on the contents of a company HR policy document. Ask a question in plain English and get a natural, conversational answer — no digging through PDFs required.

---

## 💡 Project Overview

Many companies store critical HR information in long documents that employees rarely read. This chatbot makes that information instantly accessible through natural language questions.

This project demonstrates:
- **Document processing** — loading and chunking unstructured text
- **NLP-based retrieval** — using TF-IDF and cosine similarity to find relevant sections
- **Local AI integration** — connecting a retrieval pipeline to a local LLM (Llama 3.2 via Ollama)
- **Hallucination prevention** — the model is grounded strictly to the source document

---

## 🗂️ Project Structure

```
hr-policy-chatbot/
├── chatbot_ai.py     # AI-powered chatbot using Ollama + Llama 3.2
├── hr_policy.txt     # Sample HR policy document (Nexagen Biosciences)
├── requirements.txt  # Dependencies
└── README.md
```

---

## 🛠️ Setup Instructions

### Step 1 — Make sure Python is installed

Open Command Prompt and run:
```
python --version
```
You should see Python 3.8 or higher. If not, download it from [https://python.org](https://python.org).

---

### Step 2 — Install Ollama

Ollama lets you run AI models locally on your computer for free.

1. Go to [https://ollama.com](https://ollama.com)
2. Click **Download for Windows** and run the installer
3. Once installed, **close and reopen Command Prompt**
4. Verify it worked:
```
ollama --version
```

---

### Step 3 — Download the AI model

This downloads Llama 3.2 to your computer (~2GB, one time only):
```
ollama pull llama3.2
```
Wait for it to finish before moving on.

---

### Step 4 — Clone or download this repository

**Option A — Clone with Git:**
```
git clone https://github.com/YOURUSERNAME/hr-policy-chatbot.git
cd hr-policy-chatbot
```

**Option B — Download manually:**
- Click the green **Code** button on this page
- Select **Download ZIP**
- Extract the folder and open it

---

### Step 5 — Run the chatbot

You need **two** Command Prompt windows open at the same time.

**Window 1 — Start Ollama** (keep this open the whole time):
```
ollama serve
```

**Window 2 — Run the chatbot:**

Navigate to your project folder first. The easiest way on Windows:
1. Open the folder in File Explorer
2. Click the address bar at the top
3. Type `cmd` and press Enter

Then run:
```
python chatbot_ai.py
```

---

## 💬 Example Interaction

```
╔══════════════════════════════════════════════════════════╗
║   🤖  NEXAGEN BIOSCIENCES — AI HR CHATBOT            ║
║   Powered by Llama 3.2 running locally on your PC.  ║
║   Ask me anything about company HR policies.         ║
║   Type 'quit' to exit.                               ║
╚══════════════════════════════════════════════════════════╝

  You: How many PTO days do I get in my first year?

  🤖  Bot:
     In your first two years at Nexagen, you accrue PTO
     at a rate of 15 days per year (1.25 days per month).
     You can also roll over up to 5 unused days into the
     next year, and cash out up to 3 days each November.

  ────────────────────────────────────────────────────────

  You: What is the 401k match?

  🤖  Bot:
     Nexagen matches up to 4% of your salary in the 401k.
     Keep in mind you won't be fully vested in those
     company contributions until you've completed 3 years
     of service.

  ────────────────────────────────────────────────────────

  You: Can my manager take away my work from home days?

  🤖  Bot:
     Yes, remote work privileges can be revoked by your
     manager with 2 weeks written notice if performance
     or collaboration standards are not being met.
```

---

## ⚙️ How It Works

### Step 1 — Document Chunking
The HR policy text is split into sections. Each section covers a specific topic (PTO, remote work, benefits, etc.).

### Step 2 — TF-IDF Retrieval
When you ask a question, the chatbot:
1. Converts your question into a TF-IDF vector
2. Computes cosine similarity between your question and every section
3. Returns the top 3 most relevant sections

> **TF-IDF** (Term Frequency–Inverse Document Frequency) scores how important a word is to a specific section vs. the whole document. Words that appear in one section but not others score highest — making retrieval precise.

### Step 3 — AI Answer Generation
The retrieved sections and your question are passed to **Llama 3.2**, which generates a natural, conversational answer grounded only in the document.

---

## 🔒 Why "Grounded" Matters

The AI is explicitly instructed to answer **only from the provided document excerpts** — not from its general training knowledge. This prevents hallucination, which is a critical requirement for real-world enterprise AI applications.

---

## ❗ Troubleshooting

**`ollama` is not recognized**
- Ollama is not installed or needs a fresh Command Prompt after install
- Close all Command Prompt windows, reopen, and try again

**`python` is not recognized**
- Python is not installed — download from [https://python.org](https://python.org)
- During install, check the box that says **"Add Python to PATH"**

**Chatbot says it can't reach Ollama**
- Make sure `ollama serve` is running in a separate Command Prompt window
- Do not close that window while using the chatbot

**Model takes a long time to respond**
- This is normal on first use — Llama 3.2 is loading into memory
- Responses get faster after the first question

---

## 🛠️ Built With

- **Python** (standard library only — no pip installs required)
- **TF-IDF + Cosine Similarity** for document retrieval
- **Ollama** for local LLM inference
- **Llama 3.2** as the language model

---

## 📌 Real-World Applications

This architecture — retrieval + generation — is the foundation of **RAG (Retrieval-Augmented Generation)**, the most widely used pattern for enterprise AI chatbots today. Real-world use cases include:

- Internal HR or IT helpdesks
- Customer support bots grounded in product documentation
- Compliance Q&A tools in regulated industries like healthcare, finance, and pharma

---

## 👤 Jadon Wilson


[LinkedIn](www.linkedin.com/in/jadon-wilson-0b1307188) • [GitHub](https://github.com/WhoWillJwill)
