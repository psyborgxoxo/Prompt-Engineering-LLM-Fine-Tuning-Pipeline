import os
import sys
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import asyncio

load_dotenv()
sys.path.append(os.path.dirname(__file__))

from executor import classify, answer_question, summarize
from evaluator import evaluate_ragas

app = FastAPI(
    title="Prompt Engineering & LLM Evaluation API",
    description="Classification, Q&A, summarization and RAG evaluation endpoints",
    version="1.0.0"
)


# ── Request/Response models ──────────────────────────────

class ClassifyRequest(BaseModel):
    text: str
    categories: list[str] = ["bug_report", "feature_request", "support_query"]

class ClassifyResponse(BaseModel):
    text:       str
    label:      str
    confidence: float
    reasoning:  str
    latency:    float

class QARequest(BaseModel):
    question:        str
    context:         str
    source:          str = "document"
    domain_keywords: list[str] = []

class QAResponse(BaseModel):
    question: str
    answer:   str
    tokens:   int
    latency:  float

class SummarizeRequest(BaseModel):
    text:  str
    title: str = "document"

class EvaluateResponse(BaseModel):
    faithfulness:     float
    answer_relevancy: float
    num_questions:    int
    evaluator:        str


# ── Health ───────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "Prompt Engineering API"}


# ── Classify endpoint ────────────────────────────────────

@app.post("/classify", response_model=ClassifyResponse)
async def classify_text(request: ClassifyRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    try:
        result = classify(request.text, request.categories)
        return ClassifyResponse(
            text=request.text,
            label=result["label"] or "unknown",
            confidence=result["confidence"] or 0.0,
            reasoning=result["reasoning"] or "",
            latency=result["latency"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Q&A endpoint ─────────────────────────────────────────

@app.post("/qa", response_model=QAResponse)
async def question_answer(request: QARequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        result = answer_question(
            question=request.question,
            context=request.context,
            source=request.source,
            domain_keywords=request.domain_keywords or None
        )
        return QAResponse(
            question=request.question,
            answer=result["output"],
            tokens=result["tokens"],
            latency=result["latency"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Summarize with SSE streaming ─────────────────────────

@app.post("/summarize/stream")
async def summarize_stream(request: SummarizeRequest):
    async def generate():
        result = summarize(request.text, request.title)
        words = result["output"].split(" ")
        for word in words:
            yield f"data: {word} \n\n"
            await asyncio.sleep(0.05)
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── Evaluate endpoint ─────────────────────────────────────

@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate():
    try:
        results = evaluate_ragas()
        return EvaluateResponse(
            faithfulness=results["faithfulness"],
            answer_relevancy=results["answer_relevancy"],
            num_questions=results["num_questions"],
            evaluator=results["evaluator"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)