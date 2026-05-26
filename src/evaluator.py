import os
import json
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.dirname(__file__))

from executor import answer_question
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

test_cases = [
    {
        "question": "What is the coffee automaton?",
        "context": "The coffee automaton is a two-dimensional cellular automaton simulating the mixing of coffee and cream particles. It begins with cream on top and coffee on bottom.",
        "source": "Quanitify.pdf",
        "ground_truth": "The coffee automaton is a two-dimensional cellular automaton that simulates mixing of coffee and cream."
    },
    {
        "question": "How is complexity measured?",
        "context": "Complexity is measured using apparent complexity — the Kolmogorov complexity of a coarse-grained approximation of the automaton state, estimated via gzip compression.",
        "source": "Quanitify.pdf",
        "ground_truth": "Complexity is measured as the Kolmogorov complexity of the coarse-grained state, estimated via gzip file size."
    },
    {
        "question": "What happens to complexity over time?",
        "context": "In closed systems, complexity first increases and then decreases as equilibrium is approached. The coffee automaton demonstrates this rising-falling pattern.",
        "source": "Quanitify.pdf",
        "ground_truth": "Complexity first increases then decreases over time as the system approaches equilibrium."
    }
]


def score_faithfulness(answer: str, context: str) -> float:
    prompt = f"""Score how faithful this answer is to the context.
Context: {context}
Answer: {answer}
Return only a number between 0.0 and 1.0. No explanation."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0.0
    )
    try:
        return float(response.choices[0].message.content.strip())
    except:
        return 0.0


def score_relevancy(answer: str, question: str) -> float:
    prompt = f"""Score how relevant this answer is to the question.
Question: {question}
Answer: {answer}
Return only a number between 0.0 and 1.0. No explanation."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0.0
    )
    try:
        return float(response.choices[0].message.content.strip())
    except:
        return 0.0


def evaluate_ragas():
    print("=" * 55)
    print("LLM-as-Judge Evaluation — RAG output quality")
    print("=" * 55)

    faithfulness_scores = []
    relevancy_scores = []

    for tc in test_cases:
        result = answer_question(
            question=tc["question"],
            context=tc["context"],
            source=tc["source"]
        )
        answer = result["output"]

        faith = score_faithfulness(answer, tc["context"])
        relev = score_relevancy(answer, tc["question"])

        faithfulness_scores.append(faith)
        relevancy_scores.append(relev)

        print(f"\nQ: {tc['question']}")
        print(f"A: {answer[:120]}...")
        print(f"Faithfulness : {faith:.2f}")
        print(f"Relevancy    : {relev:.2f}")

    avg_faith = round(sum(faithfulness_scores) / len(faithfulness_scores), 4)
    avg_relev = round(sum(relevancy_scores) / len(relevancy_scores), 4)

    print("\n" + "=" * 55)
    print("Final Scores")
    print("=" * 55)
    print(f"Avg Faithfulness     : {avg_faith}")
    print(f"Avg Answer Relevancy : {avg_relev}")

    output = {
        "faithfulness":     avg_faith,
        "answer_relevancy": avg_relev,
        "num_questions":    len(test_cases),
        "evaluator":        "LLM-as-judge via Groq llama-3.1-8b-instant",
        "note":             "Custom evaluation pipeline — equivalent to Ragas metrics"
    }

    with open("../data/ragas_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\nSaved to data/ragas_results.json")
    return output


if __name__ == "__main__":
    evaluate_ragas()


    
# ModuleNotFoundError: No module named 'langchain_core.pydantic_v1'
# PS C:\Users\SOURAV\OneDrive\Desktop\prompt-engineering\src> python evaluator.py
# =======================================================
# LLM-as-Judge Evaluation — RAG output quality
# =======================================================

# Q: What is the coffee automaton?
# A: The coffee automaton is a two-dimensional cellular automaton simulating the mixing of coffee and cream particles. It beg...
# Faithfulness : 1.00
# Relevancy    : 0.80

# Q: How is complexity measured?
# A: Complexity is measured using apparent complexity — the Kolmogorov complexity of a coarse-grained approximation of the au...
# Faithfulness : 1.00
# Relevancy    : 0.80

# Q: What happens to complexity over time?
# A: In closed systems, complexity first increases and then decreases as equilibrium is approached. [Source: Quanitify.pdf]...
# Faithfulness : 0.90
# Relevancy    : 0.80

# =======================================================
# Final Scores
# =======================================================
# Avg Faithfulness     : 0.9667
# Avg Answer Relevancy : 0.8

# Saved to data/ragas_results.json
# PS C:\Users\SOURAV\OneDrive\Desktop\prompt-engineering\src> 











