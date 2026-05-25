import os
import json
import time
from groq import Groq
from dotenv import load_dotenv
from prompts import (
    build_qa_prompt,
    build_summarization_prompt,
    build_classification_prompt,
    build_cot_prompt,
    is_in_scope,
    SYSTEM_PROMPTS,
    PromptResult
)

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"


def run_prompt(prompt: PromptResult, max_tokens: int = 500) -> dict:
    start = time.time()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system",  "content": prompt.system},
            {"role": "user",    "content": prompt.user}
        ],
        max_tokens=max_tokens,
        temperature=0.1
    )
    latency = round(time.time() - start, 2)
    return {
        "task":     prompt.task,
        "output":   response.choices[0].message.content,
        "tokens":   response.usage.total_tokens,
        "latency":  latency
    }


def answer_question(
    question: str,
    context:  str,
    source:   str = "document",
    domain_keywords: list[str] = None
) -> dict:
    if domain_keywords and not is_in_scope(question, domain_keywords):
        return {
            "task":    "refusal",
            "output":  f"This question is outside my current knowledge scope. I can only answer questions about {', '.join(domain_keywords)}.",
            "tokens":  0,
            "latency": 0
        }
    prompt = build_qa_prompt(question, context, source)
    return run_prompt(prompt)


def summarize(text: str, title: str = "document") -> dict:
    prompt = build_summarization_prompt(text, title)
    return run_prompt(prompt, max_tokens=300)


def classify(text: str, categories: list[str]) -> dict:
    prompt = build_classification_prompt(text, categories)
    result = run_prompt(prompt, max_tokens=150)

    try:
        parsed = json.loads(result["output"])
        result["label"]      = parsed.get("label")
        result["confidence"] = parsed.get("confidence")
        result["reasoning"]  = parsed.get("reasoning")
    except json.JSONDecodeError:
        result["label"]      = None
        result["confidence"] = None
        result["reasoning"]  = "Parse error"

    return result


def reason(question: str) -> dict:
    prompt = build_cot_prompt(question)
    return run_prompt(prompt, max_tokens=400)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing prompt executor")
    print("=" * 60)

    # Test 1 — QA with source citation
    print("\n[Test 1] QA with citation")
    result = answer_question(
        question="What is the coffee automaton?",
        context="The coffee automaton is a two-dimensional cellular automaton simulating the mixing of coffee and cream particles.",
        source="Quanitify.pdf",
        domain_keywords=["coffee", "automaton", "complexity", "entropy"]
    )
    print(f"Answer  : {result['output']}")
    print(f"Latency : {result['latency']}s | Tokens: {result['tokens']}")

    # Test 2 — Out-of-scope refusal
    print("\n[Test 2] Out-of-scope refusal")
    result = answer_question(
        question="Who won the FIFA World Cup?",
        context="The coffee automaton simulates mixing.",
        domain_keywords=["coffee", "automaton", "complexity", "entropy"]
    )
    print(f"Answer  : {result['output']}")

    # Test 3 — Classification
    print("\n[Test 3] Classification")
    result = classify(
        text="The app crashes every time I try to upload a file larger than 10MB.",
        categories=["bug_report", "feature_request", "support_query"]
    )
    print(f"Label      : {result['label']}")
    print(f"Confidence : {result['confidence']}")
    print(f"Reasoning  : {result['reasoning']}")

    # Test 4 — Summarization
    print("\n[Test 4] Summarization")
    result = summarize(
        text="The coffee automaton models complexity in closed systems. It uses a two-dimensional grid where cream and coffee particles mix over time. Complexity rises then falls as the system reaches equilibrium.",
        title="Coffee Automaton Paper"
    )
    print(f"Summary : {result['output']}")

    # Test 5 — Chain of thought
    print("\n[Test 5] Chain of thought")
    result = reason("If a RAG pipeline processes 500 documents each split into 10 chunks, how many embeddings are needed?")
    print(f"Reasoning : {result['output']}")

    print("\nAll executor tests complete.")