from dataclasses import dataclass
from typing import Optional


# ── Prompt templates ─────────────────────────────────────

SYSTEM_PROMPTS = {
    "qa": """You are a precise question-answering assistant.
Answer ONLY based on the provided context.
Always cite the source document at the end of your answer as [Source: <filename>].
If the answer is not in the context, respond with: "I cannot find this in the provided documents."
Never make assumptions or use outside knowledge.""",

    "summarization": """You are an expert summarization assistant.
Produce concise, factual summaries that preserve key information.
Structure your summary with: one sentence overview, then 3-5 bullet points of key facts.
Do not include opinions or information not present in the source text.
Always end with: [Summary of: <document title>]""",

    "classification": """You are a text classification assistant.
Classify the input text into exactly one of the provided categories.
Respond in this exact JSON format:
{"label": "<category>", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}
Do not add any text outside the JSON.""",

    "refusal": """You are a helpful assistant with strict scope boundaries.
You ONLY answer questions related to the provided documents and context.
For any question outside this scope, respond with:
"This question is outside my current knowledge scope. I can only answer questions about [DOMAIN]."
Never attempt to answer out-of-scope questions even if you know the answer."""
}


# ── Few-shot examples ─────────────────────────────────────

FEW_SHOT_QA = [
    {
        "context": "The Eiffel Tower was completed in 1889 and stands 330 meters tall.",
        "question": "How tall is the Eiffel Tower?",
        "answer": "The Eiffel Tower stands 330 meters tall. [Source: eiffel_facts.pdf]"
    },
    {
        "context": "Python was created by Guido van Rossum and first released in 1991.",
        "question": "Who created Python?",
        "answer": "Python was created by Guido van Rossum. [Source: python_history.pdf]"
    }
]

FEW_SHOT_CLASSIFICATION = [
    {
        "text": "The server is returning 500 errors on all API calls.",
        "label": "bug_report",
        "confidence": 0.97,
        "reasoning": "Describes a technical system failure requiring investigation."
    },
    {
        "text": "Can you add a dark mode to the dashboard?",
        "label": "feature_request",
        "confidence": 0.95,
        "reasoning": "User is requesting a new UI capability."
    },
    {
        "text": "How do I reset my password?",
        "label": "support_query",
        "confidence": 0.99,
        "reasoning": "User needs help with an existing feature."
    }
]

FEW_SHOT_COT = [
    {
        "question": "A company has 3 offices each with 12 employees. How many total employees?",
        "chain_of_thought": """Let me think through this step by step:
1. Number of offices: 3
2. Employees per office: 12
3. Total = 3 × 12 = 36
Therefore, the company has 36 total employees.""",
        "answer": "36 employees"
    }
]


# ── Prompt builders ───────────────────────────────────────

@dataclass
class PromptResult:
    system:   str
    user:     str
    task:     str
    few_shot: bool


def build_qa_prompt(
    question: str,
    context:  str,
    source:   str = "document",
    use_few_shot: bool = True
) -> PromptResult:
    examples = ""
    if use_few_shot:
        for ex in FEW_SHOT_QA:
            examples += f"\nContext: {ex['context']}\nQ: {ex['question']}\nA: {ex['answer']}\n"

    user_msg = f"{examples}\nContext: {context}\nSource: {source}\nQ: {question}\nA:"
    return PromptResult(
        system=SYSTEM_PROMPTS["qa"],
        user=user_msg,
        task="qa",
        few_shot=use_few_shot
    )


def build_summarization_prompt(
    text:  str,
    title: str = "document"
) -> PromptResult:
    user_msg = f"Document title: {title}\n\nText to summarize:\n{text}"
    return PromptResult(
        system=SYSTEM_PROMPTS["summarization"],
        user=user_msg,
        task="summarization",
        few_shot=False
    )


def build_classification_prompt(
    text:       str,
    categories: list[str],
    use_few_shot: bool = True
) -> PromptResult:
    cat_str = ", ".join(categories)
    examples = ""
    if use_few_shot:
        valid = [e for e in FEW_SHOT_CLASSIFICATION if e["label"] in categories]
        for ex in valid[:2]:
            examples += f'\nText: "{ex["text"]}"\n'
            examples += f'{{"label": "{ex["label"]}", "confidence": {ex["confidence"]}, "reasoning": "{ex["reasoning"]}"}}\n'

    user_msg = f"Categories: {cat_str}\n{examples}\nText: \"{text}\"\n"
    return PromptResult(
        system=SYSTEM_PROMPTS["classification"],
        user=user_msg,
        task="classification",
        few_shot=use_few_shot
    )


def build_cot_prompt(
    question: str,
    use_few_shot: bool = True
) -> PromptResult:
    system = """You are a careful reasoning assistant.
Always think step by step before giving your final answer.
Show your reasoning explicitly before stating the conclusion."""

    examples = ""
    if use_few_shot:
        for ex in FEW_SHOT_COT:
            examples += f"\nQ: {ex['question']}\n{ex['chain_of_thought']}\nAnswer: {ex['answer']}\n"

    user_msg = f"{examples}\nQ: {question}\nLet me think through this step by step:"
    return PromptResult(
        system=system,
        user=user_msg,
        task="chain_of_thought",
        few_shot=use_few_shot
    )


def is_in_scope(question: str, domain_keywords: list[str]) -> bool:
    question_lower = question.lower()
    return any(kw.lower() in question_lower for kw in domain_keywords)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing prompt builders")
    print("=" * 60)

    # Test 1 — QA prompt
    qa = build_qa_prompt(
        question="What is the coffee automaton?",
        context="The coffee automaton is a two-dimensional cellular automaton simulating mixing of coffee and cream.",
        source="Quanitify.pdf"
    )
    print(f"\n[QA Prompt]")
    print(f"Task     : {qa.task}")
    print(f"Few-shot : {qa.few_shot}")
    print(f"System   : {qa.system[:80]}...")
    print(f"User     : {qa.user[:200]}...")

    # Test 2 — Classification prompt
    clf = build_classification_prompt(
        text="The app crashes every time I open it on iOS 17.",
        categories=["bug_report", "feature_request", "support_query"]
    )
    print(f"\n[Classification Prompt]")
    print(f"Task     : {clf.task}")
    print(f"User msg : {clf.user[:200]}...")

    # Test 3 — Scope check
    domain = ["coffee", "automaton", "complexity", "entropy", "cellular"]
    print(f"\n[Scope Check]")
    print(f"'What is entropy?' in scope: {is_in_scope('What is entropy?', domain)}")
    print(f"'Who won the World Cup?' in scope: {is_in_scope('Who won the World Cup?', domain)}")

    print("\nAll prompt builders working.")