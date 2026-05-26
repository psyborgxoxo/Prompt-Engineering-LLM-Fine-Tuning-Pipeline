Prompt Engineering and LLM Fine-Tuning Pipeline
A production-grade prompt engineering and LLM evaluation system built with Groq LLaMA, LoRA/PEFT fine-tuning, and FastAPI. Covers system prompt design, few-shot learning, chain-of-thought reasoning, classification with guardrails, and LLM-as-judge evaluation.
Tech Stack: Python, Groq LLaMA-3.1-8B, DistilBERT, LoRA/PEFT, HuggingFace Transformers, FastAPI, Docker
Project Structure:

src/prompts.py — system prompts, few-shot exemplars, chain-of-thought templates
src/executor.py — prompt runner for QA, classification, summarization, CoT
src/evaluator.py — LLM-as-judge evaluation pipeline
src/main.py — FastAPI REST API
data/fine_tuning_results.json — LoRA fine-tuning results
data/adapter_config.json — LoRA adapter config (r=16, alpha=32)
data/ragas_results.json — evaluation scores

Setup: clone the repo, create a venv, pip install -r requirements.txt, add GROQ_API_KEY to .env, run uvicorn src.main:app --port 8001
API Endpoints:

GET /health
POST /classify — intent classification with confidence score and reasoning
POST /qa — grounded Q&A with source citation and scope refusal
POST /summarize/stream — SSE streaming summarization
POST /evaluate — LLM-as-judge faithfulness and relevancy scoring

Fine-tuning: LoRA applied to DistilBERT for 3-class text classification (bug report, feature request, support query) using r=16, lora_alpha=32, target modules q_lin and v_lin. Only 1.3% of parameters trained. With 3K instruction pairs F1 reaches 0.91 vs GPT-3.5-turbo zero-shot baseline of 0.87.
Evaluation results: faithfulness 0.97, answer relevancy 0.80, evaluated across 3 test cases using LLM-as-judge pipeline.
Author: Sourav,  Automation Engineer / SDET. Built as part of an AI Engineering portfolio project.