# PROJECT SPEC — Collective Intelligence Infrastructure for AI Agents

> **Working names (pick one):** ECHO (Evolving Collective Heuristic Optimization), SAGE (Shared Agent Graph of Experience), Lore, Deja
> **Builder:** Solo developer
> **Event:** Mistral AI Hackathon — Europe 2026 (Paris or London)
> **Timeline:** ~30 hours of building time (Saturday morning → Sunday 4pm)

---

## 1. WHAT THIS PROJECT IS

### The One-Liner

A memory layer and MCP server that makes every AI agent smarter by learning from every other agent's experience — and distills that collective intelligence into progressively better open-source Mistral models.

### The Problem

AI agents are stateless. Agent A solves a hard bug on Monday. Agent B hits the exact same bug on Wednesday and wastes 15 minutes and $0.50 re-discovering the same fix. Across the industry, agents waste an estimated 60%+ of their compute on problems that have already been solved somewhere else. That is billions of dollars in wasted inference annually.

### The Solution (Three Layers)

**Layer 1 — Collective Memory (Knowledge Graph in Qdrant):** When any connected agent solves a problem, the solution path — including reasoning chain, failed attempts, and verification result — gets indexed into a searchable vector database. When another agent encounters a similar problem, it queries the system via MCP before spending tokens reasoning from scratch.

**Layer 2 — Multi-Agent Arena:** When a problem has no match in the knowledge graph, it gets dispatched to multiple LLMs simultaneously (Mistral, Claude, GPT-4, DeepSeek). Their solutions, chain-of-thought, timing, and token usage are compared. The best verified solution gets pushed back into the knowledge graph. This comparison is displayed on the web dashboard.

**Layer 3 — Distillation Engine (The Breakthrough):** Periodically, accumulated verified patterns are used to fine-tune a Ministral 3B model. The fine-tuned model has internalized collective problem-solving patterns — it does not need to query the knowledge graph because the knowledge is in its weights. Each distillation cycle produces a measurably better model. This compounding is tracked in W&B.

### Why This Wins

- Novel technical contribution: distillation layer does not exist in any prior work (HackOverflow, etc.)
- Jaw-dropping demo: watching a model get measurably smarter from collective experience
- Multi-agent arena comparison is visually compelling for judges
- Stacks all sponsor tools with genuine justification
- Hits multiple tracks: fine-tuning (core product), on-device (3B distilled model), API (orchestration)
- Eligible for multiple special prizes simultaneously
- Only possible with open-weight models — structural advantage, not just hackathon constraint

---

## 2. HACKATHON CONTEXT

### What Judges Care About (From Organizer Joffrey's Direct Advice)

- Technically impressive projects usually DO NOT win
- What wins is something that if you saw it on social media you would say "wow, this is really cool"
- Build around the LATEST features (Voxil real-time is brand new, released 2 weeks ago)
- Previous winners were often games or very demo-friendly projects
- Judging is a 5-minute live pitch in front of judges — demo moment is everything
- Simple ideas executed well beat complex ideas executed poorly
- Round-robin format: you pitch to multiple judge panels

### Tracks

- **On-device (Nvidia):** $100 GPU compute budget. Lower compute = better score. Best project under budget wins RTX 5090.
- **Fine-tuning (W&B):** Track training with W&B Models and evaluation with W&B Weave. Best self-improving workflow wins $500 credits + Mac Mini.
- **API (AWS):** Use Bedrock, Strands Agents SDK, MCP. $100 AWS credits per person.

### Available Models

- Mistral 3 (Ministral): 3B, 8B, 14B parameter variants — open weight, fine-tunable
- Mistral Large 3: Most capable Mistral model, available via Bedrock and build.nvidia.com
- Voxil: Brand new real-time speech-to-text model (optional for this project)

### Special Prizes We Target

- **Best Use of Mistral** (custom AirPods, global): Open-weight fine-tuning is the core product
- **W&B Self-Improving Workflow** ($500 + Mac Mini): The entire product IS a self-improving workflow
- **Best Startup Idea / Raise** (VC pitch opportunity, Paris only): Platform play with network effects
- **Best Use of HF Jobs & TRL** (custom merch): Fine-tuning on HF Jobs, models on Hub
- **Nvidia On-Device** (RTX 5090): Distilled Ministral 3B runs on single GPU, low compute tracked on Brev
- **Best Use of Vibe** (Mistral coding assistant): Use Vibe throughout development

### Critical Rules

- Dataset preparation and fine-tuning setup ARE ALLOWED before the hackathon
- You CANNOT run final experiments on hackathon data before the event
- HuggingFace organization membership must be set up before Friday evening
- Fine-tuning track requires models to be public and reproducible on HuggingFace Hub
- Submissions close Sunday ~4pm, followed by 5-minute pitches

---

## 3. SPONSOR TOOLS AND HOW EACH IS USED

### Nvidia (The Training Engine)

**Brev ($100 GPU Compute):**

- Role: Runs the distillation fine-tuning jobs
- Usage: Spin up GPU instance, pull Ministral 3B weights from HuggingFace, run LoRA fine-tuning on accumulated patterns
- Target: Fine-tune Ministral 3B to earn "lowest compute" bonus — stay under $15 per training run
- Track GPU costs carefully to show efficiency to judges

**build.nvidia.com (Free Hosted Model Endpoints):**

- Role: Fallback inference for novel problems + rapid development testing
- Usage: API key, HTTP requests, get responses — no GPU management needed
- Available models: Ministral 3B, 8B, Mistral Large 3

**OpenCodeReasoning Dataset (735K reasoning traces on HuggingFace):**

- Role: Primary seed data source for knowledge graph AND fine-tuning training data
- Location: `nvidia/OpenCodeReasoning` on HuggingFace
- License: CC BY 4.0, commercial use allowed
- Using Nvidia's own dataset earns direct sponsor points

### AWS (The Orchestration Layer)

**Amazon Bedrock ($100 credits):**

- Role: Production inference for Mistral Large 3 — the orchestration brain
- Usage: Serverless API calls for the Gateway Agent reasoning, query understanding, result synthesis
- 13+ Mistral models available serverlessly

**Strands Agents SDK (Open-source Python SDK):**

- Role: Framework for ALL agents in the system — Gateway Agent, Abstraction Agent, Verification Agent, Distillation Curator
- Usage: Define agents with tools, multi-agent orchestration, MCP server connectivity
- This is the ONLY agent framework used (not NeMo) to keep things simple for solo build

**Agent Core:**

- Role: Host the MCP server on AWS, making it accessible to any external agent worldwide
- Usage: Deploy Strands-based Gateway as an MCP server

**Bedrock Guardrails:**

- Role: Safety filter on incoming contributions — catches malicious code, prompt injection, leaked sensitive data
- Usage: Few lines of code, demonstrates security awareness to judges

### Weights & Biases (The Observability Layer)

**W&B Models (Training Tracker):**

- Role: Tracks every distillation cycle — loss curves, learning rates, gradients, hyperparameters, model artifacts
- Usage: `wandb.init()` in training script, HuggingFace Trainer handles the rest automatically
- Critical output: Cross-cycle comparison showing model improving with each distillation round

**W&B Weave (Evaluation + Tracing):**

- Role: Traces every agent interaction with the system, runs evaluations comparing base vs distilled model
- Usage: `@weave.op()` decorator on functions, evaluation framework on SWE-bench holdout
- Critical output: Three-way comparison chart (base model vs knowledge-graph-assisted vs distilled model)

### HuggingFace (Model Hosting)

**HF Hub:**

- Role: Host distilled models publicly (required for fine-tuning track)
- Each distillation cycle produces a new model version on Hub

**HF Jobs ($20 credits):**

- Role: Alternative/backup compute for fine-tuning if Brev has issues

---

## 4. TECHNICAL ARCHITECTURE

### Complete Request Flow

```
External Agent encounters a problem
    │
    ▼
Queries MCP Server (hosted on AWS via Agent Core)
    │
    ▼
Gateway Agent (Strands Agents SDK + Mistral Large 3 on Bedrock)
    │
    ├─ Step 1: Check Distilled Model (Ministral 3B on Brev or build.nvidia.com)
    │          └─ If confidence > 0.8 → Return answer (fastest, cheapest)
    │
    ├─ Step 2: Query Knowledge Graph (Qdrant vector search)
    │          └─ If strong match → Synthesize response via Bedrock
    │
    └─ Step 3: Multi-Agent Arena (no match found)
               ├─ Dispatch to: Mistral Large 3, Claude, GPT-4, DeepSeek (parallel)
               ├─ Collect: solutions, chain-of-thought, timing, tokens
               ├─ Judge best solution (Mistral Large 3)
               ├─ Verify in sandbox
               └─ Push verified solution → Qdrant knowledge graph
                       │
                       ▼
               Contribution Pipeline
                   ├─ Bedrock Guardrails (safety check)
                   ├─ Abstraction Agent (Strands — strips proprietary details)
                   └─ Pattern enters Knowledge Graph
                           │
                           ▼
               Periodically: Distillation Engine
                   ├─ Curator Agent selects high-quality patterns
                   ├─ Fine-tune Ministral 3B on Brev GPU (tracked in W&B Models)
                   ├─ Evaluate new model (W&B Weave vs SWE-bench holdout)
                   └─ If improved → Replace distilled model in production
```

### Technology Stack

| Layer               | Technology                                                | Why                                                                                                  |
| ------------------- | --------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Knowledge Graph     | **Qdrant** (Docker or cloud)                              | Vector-native, simple setup, semantic search by meaning not keywords, payload filtering for metadata |
| Embeddings          | **Mistral Embed** (via Bedrock) or **Jina Embeddings v3** | Generate vectors for patterns and queries                                                            |
| Agent Framework     | **Strands Agents SDK** (AWS)                              | All agents built here — gateway, abstraction, verification, curator. MCP support built-in            |
| Orchestration LLM   | **Mistral Large 3** on Bedrock                            | Query understanding, result synthesis, judging arena solutions                                       |
| Distillation Target | **Ministral 3B** (LoRA fine-tuning)                       | Small enough for on-device story, cheap to fine-tune, dramatic demo when it outperforms base         |
| Fine-tuning Library | **Unsloth** or **HuggingFace TRL**                        | Fastest LoRA fine-tuning for hackathon speed                                                         |
| Training Compute    | **Nvidia Brev** ($100 GPU credits)                        | Spin up A10G/A100, run training, track costs                                                         |
| Training Tracking   | **W&B Models**                                            | Loss curves, hyperparameters, model artifacts, cross-cycle comparison                                |
| Eval Tracking       | **W&B Weave**                                             | Agent traces, evaluation runs, three-way model comparison                                            |
| Web Dashboard       | **React** (single JSX artifact or Next.js)                | Real-time display of problems, solutions, agent comparisons, metrics                                 |
| MCP Server          | **Python** (Strands + FastAPI)                            | External agents connect via MCP URL                                                                  |
| Multi-Agent Arena   | Direct API calls (parallel)                               | Mistral (Bedrock), Claude API, OpenAI API, DeepSeek API                                              |
| Sandbox Execution   | **Modal** or **subprocess** isolation                     | Test solutions before they enter knowledge graph                                                     |

### Qdrant Setup and Schema

Qdrant runs in a single Docker container: `docker run -p 6333:6333 qdrant/qdrant`

Each pattern is stored as a point in a collection with this structure:

```python
# Collection: "patterns"
# Vector: 1024-dimensional embedding from Mistral Embed or Jina v3
# Payload (metadata stored alongside the vector):
{
    "pattern_id": "LORE-00001",
    "problem_class": "Event ingestion loss under concurrent load",
    "problem_signature": "HTTP endpoint receiving burst traffic drops events...",
    "domain_tags": ["webhooks", "concurrency", "api", "reliability"],
    "reasoning_trace": {
        "failed_approaches": [
            {"approach": "Increase timeout", "why_failed": "Treats symptom not cause"},
            {"approach": "Add retry logic", "why_failed": "Wrong layer, sender may not retry"}
        ],
        "key_insight": "Decouple reception from processing"
    },
    "solution_template": "1. Write raw payload to queue, return 200 <50ms...",
    "generic_code_template": "async def handle_webhook(request):\n    ...",
    "metadata": {
        "source": "github_issues",  # or "swe_bench", "stackoverflow", "synthetic", "nvidia_ocr"
        "verification_status": "verified",
        "success_rate": 0.94,
        "times_applied": 0,
        "estimated_token_savings": 12000,
        "difficulty": "medium"
    },
    "refinements": []  # Accumulated tips from agents who used this pattern
}
```

Search is done by embedding the query and finding nearest vectors, with optional payload filtering:

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

client = QdrantClient("localhost", port=6333)

# Search by semantic similarity with optional domain filter
results = client.search(
    collection_name="patterns",
    query_vector=embed(query_text),  # embed() calls Mistral Embed or Jina
    query_filter=Filter(
        must=[
            FieldCondition(key="domain_tags", match=MatchValue(value="webhooks"))
        ]
    ),
    limit=5,
    with_payload=True
)
```

### MCP Server Interface

The MCP server exposes these tools to any connected agent:

**Tool 1: `search_patterns`**

- Input: `{ "query": "string describing the problem", "domain_filter": "optional domain tag" }`
- Output: Top 5 matching patterns with solutions, reasoning traces, and metadata
- This is the primary tool agents call before reasoning from scratch

**Tool 2: `submit_solution`**

- Input: `{ "problem": "description", "solution": "the fix", "reasoning_trace": "steps taken", "outcome": "success/failure", "tokens_used": 4200, "time_seconds": 48 }`
- Output: Confirmation that solution was accepted or rejected by guardrails
- Agents call this after solving a problem to contribute back

**Tool 3: `check_distilled_model`**

- Input: `{ "problem": "description" }`
- Output: Distilled model's response + confidence score
- Fastest path — checks if the fine-tuned model already knows the answer natively

**Tool 4: `get_arena_comparison`**

- Input: `{ "problem": "description" }`
- Output: All agent solutions side by side with timing and token metrics
- Used when displaying the multi-agent comparison on the dashboard

### Multi-Agent Arena Design

When no pattern match exists and the distilled model is not confident:

```python
import asyncio
import time

async def run_arena(problem: str) -> dict:
    """Dispatch problem to multiple LLMs in parallel, collect and compare results."""

    async def query_model(model_name: str, api_func, problem: str):
        start = time.time()
        # Each api_func calls the respective model's API
        response = await api_func(problem)
        elapsed = time.time() - start
        return {
            "model": model_name,
            "solution": response.content,
            "chain_of_thought": response.reasoning,  # if available
            "tokens_used": response.usage.total_tokens,
            "time_seconds": elapsed,
            "cost_estimate": calculate_cost(model_name, response.usage)
        }

    # Run all models in parallel
    results = await asyncio.gather(
        query_model("Mistral Large 3", call_bedrock_mistral, problem),
        query_model("Claude Sonnet", call_claude_api, problem),
        query_model("GPT-4", call_openai_api, problem),
        query_model("DeepSeek", call_deepseek_api, problem),
    )

    # Judge: Use Mistral Large 3 to evaluate which solution is best
    best = await judge_solutions(problem, results)

    # Verify the best solution in sandbox
    verified = await verify_in_sandbox(best["solution"])

    if verified:
        # Push to knowledge graph
        await submit_to_qdrant(problem, best)

    return {
        "all_results": results,
        "best_solution": best,
        "verified": verified
    }
```

### Distillation Engine Design

The distillation engine runs as a separate process, kicked off manually during the hackathon or on a schedule in production.

```python
# distill.py — Run on Nvidia Brev GPU instance
import wandb
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# 1. Initialize W&B tracking
wandb.init(
    project="echo-distillation",  # or whatever project name
    config={
        "base_model": "mistralai/Ministral-3B-Instruct-2412",
        "dataset_size": None,  # filled after loading
        "learning_rate": 2e-5,
        "epochs": 3,
        "lora_rank": 16,
        "lora_alpha": 32,
        "distillation_cycle": 1,  # increment each cycle
    }
)

# 2. Load base model with LoRA
model, tokenizer = FastLanguageModel.from_pretrained(
    "mistralai/Ministral-3B-Instruct-2412",
    max_seq_length=4096,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
)

# 3. Load training data (patterns exported from Qdrant as instruction-completion pairs)
# Format: {"instruction": "problem description", "completion": "solution with reasoning"}
dataset = load_dataset("json", data_files="training_patterns.jsonl")
wandb.config.update({"dataset_size": len(dataset["train"])})

# 4. Train
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    args=SFTConfig(
        output_dir="./distilled_model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=2e-5,
        logging_steps=10,
        report_to="wandb",  # W&B tracks everything automatically
    ),
    tokenizer=tokenizer,
)
trainer.train()

# 5. Save and push to HuggingFace Hub
model.save_pretrained("./distilled_model")
model.push_to_hub("your-hf-org/echo-ministral-3b-distilled-v1")

# 6. Log model artifact to W&B
artifact = wandb.Artifact("distilled-model-v1", type="model")
artifact.add_dir("./distilled_model")
wandb.log_artifact(artifact)

wandb.finish()
```

### Training Data Format (for Distillation)

Each pattern from Qdrant gets formatted as an instruction-completion pair:

```jsonl
{"instruction": "A webhook endpoint is dropping events under high concurrent load. The handler processes each event synchronously within the HTTP request. Approximately 3% of events are lost during traffic spikes.", "completion": "This is a synchronous ingestion bottleneck. The solution is to decouple event reception from processing:\n1. Modify the handler to immediately write the raw payload to a message queue and return HTTP 200 within 50ms\n2. Create a separate consumer that processes events from the queue with controlled concurrency\n3. Implement idempotency tracking using the event provider's native idempotency key\n4. Add a dead letter queue for events that fail processing after N retries\nThis pattern has a 94% success rate across 47 verified applications."}
{"instruction": "A FastAPI endpoint returns 422 Unprocessable Entity when receiving multipart form data with files larger than 10MB, even though no file size limit is explicitly set.", "completion": "This is typically caused by the default request body size limit in the ASGI server (Uvicorn defaults to ~1MB for form data). The fix involves:\n1. Set the request body size limit in Uvicorn: uvicorn.run(app, limit_concurrency=None)\n2. Or configure in the middleware: app.add_middleware(TrustedHostMiddleware)\n3. For production, use nginx/reverse proxy with client_max_body_size directive\nCommon mistake: trying to fix this in FastAPI's UploadFile configuration, which does not control the underlying server limit."}
```

---

## 5. SEED DATA STRATEGY

### Sources (All preparation allowed before hackathon)

**Source 1: Nvidia OpenCodeReasoning (Crown Jewel — earns sponsor points)**

- 735K Python samples with full reasoning traces, designed for SFT
- Location: `nvidia/OpenCodeReasoning` on HuggingFace
- License: CC BY 4.0
- Transform: Filter 2K-3K entries at medium difficulty. Use Mistral to reframe from competitive programming to real-world patterns.
- Each entry has: `input` (problem), `output` (reasoning trace from DeepSeek-R1), `solution` (final code)

**Source 2: SWE-bench Lite (Evaluation + Demo)**

- 300 curated real GitHub issues with ground-truth patches and test cases
- Location: `princeton-nlp/SWE-bench_Lite` on HuggingFace
- Use all 300 as seed. Reserve 50-100 as holdout eval set (never seen in training).
- Repos: Django, Flask, scikit-learn, matplotlib, sympy

**Source 3: GitHub Issues (Real-world debugging traces)**

- Target repos: FastAPI, LangChain, Next.js, Transformers, Django, Flask
- Filter: closed issues labeled "bug" with linked PRs
- Extract: issue body → problem, PR diff → solution, comment thread → reasoning trace
- Pull 500-1K pairs via GitHub API

**Source 4: Synthetic (Demo-specific, ensures reliable live demos)**

- Generate 100-200 patterns using Mistral API for domains you plan to demo
- Patterns in the "neighborhood" of demo problems (related but not identical)

### Target Dataset

- 3K-5K total patterns in JSONL format
- Combined from all sources above
- Each pattern transformed to the Qdrant schema shown in Section 4

### Transformation Pipeline

```python
# transform_to_pattern.py
# Run this with Mistral API to convert raw data into standardized patterns

SYSTEM_PROMPT = """You are a pattern extraction engine. Given a coding problem and its solution,
extract a standardized pattern with these fields:
1. problem_class: A short abstract name for this class of problem (e.g., "Event ingestion loss under concurrent load")
2. problem_signature: A 1-2 sentence abstract description that would match similar problems regardless of specific library or language
3. domain_tags: List of relevant tags (e.g., ["webhooks", "concurrency", "api"])
4. reasoning_trace: Object with "failed_approaches" (list of common wrong approaches and why they fail) and "key_insight" (the core realization that leads to the correct fix)
5. solution_template: Step-by-step generic solution
6. difficulty: "easy", "medium", or "hard"
7. estimated_token_savings: How many tokens an agent would save by knowing this pattern (estimate: easy=5000, medium=12000, hard=25000)

Return valid JSON only."""

async def transform_entry(raw_problem: str, raw_solution: str, source: str) -> dict:
    response = await call_mistral(
        system=SYSTEM_PROMPT,
        user=f"Problem:\n{raw_problem}\n\nSolution:\n{raw_solution}"
    )
    pattern = json.loads(response)
    pattern["metadata"] = {
        "source": source,
        "verification_status": "unverified",
        "success_rate": 0.0,
        "times_applied": 0,
        "estimated_token_savings": pattern.pop("estimated_token_savings"),
        "difficulty": pattern.pop("difficulty")
    }
    pattern["refinements"] = []
    pattern["pattern_id"] = generate_id()
    return pattern
```

---

## 6. WEB DASHBOARD

### What It Shows

The dashboard is the visual centerpiece of the demo. It is a React web app (can be a single .jsx artifact or a Next.js app) that displays:

**Main View — Problem Feed:**

- List of problems that have been submitted to the system
- Each problem card shows: the problem description, the source (which agent submitted it), the resolution status
- For resolved problems: the solution, which path resolved it (distilled model / knowledge graph / arena)
- For arena-resolved problems: side-by-side comparison of all agent solutions with timing and token metrics

**Metrics Panel:**

- Total patterns in knowledge graph (growing number)
- Average token savings per query
- Arena win rates by model (pie chart: which LLM wins most often)
- Resolution path distribution (how many queries resolved by: distilled model vs knowledge graph vs arena)

**Distillation Panel:**

- W&B embed or recreated chart showing eval scores across distillation cycles
- Base model performance vs distilled model performance
- Cost per solution trend (should be decreasing)

**Live Activity Feed:**

- Real-time stream of: agent queries, pattern matches, arena dispatches, new contributions
- Shows the system is alive and working during the demo

### Technical Implementation

The dashboard connects to the backend via WebSocket (for real-time updates) and REST API (for historical data). The backend is the same FastAPI server that hosts the MCP interface.

```
Dashboard (React) ←WebSocket→ Backend (FastAPI)
                  ←REST API→
```

---

## 7. DEMO SCRIPT (5-Minute Pitch)

### Minute 0:00-1:00 — The Hook

"Right now, AI agents are the most expensive interns in the world. They are smart, but they have no memory. Agent A solves a bug on Monday. Agent B hits the exact same bug on Wednesday and wastes 15,000 tokens re-discovering the same fix. We built [PROJECT NAME] to fix this."

### Minute 1:00-3:00 — Live Demo (Split Screen)

Show the web dashboard. Submit a real coding problem.

**Path 1 — Knowledge Graph Hit:**
Problem: "Stripe webhook handler dropping 3% of events under load"
System queries Qdrant → finds matching pattern → returns verified solution in 3 seconds, 200 tokens.
Show the pattern card on dashboard with success rate and previous applications.

**Path 2 — No match, Multi-Agent Arena triggered:**
Problem: A novel problem with no existing pattern.
System dispatches to Mistral, Claude, GPT-4, DeepSeek simultaneously.
Dashboard shows all four agents racing, their solutions streaming in.
Solutions compared: timing, tokens, correctness. Best one gets verified and pushed to knowledge graph.
"This problem will never need to be solved from scratch again."

### Minute 3:00-4:00 — The Distillation Magic

"But here is the real breakthrough. The system does not just cache solutions — it breeds smarter models."

Show W&B dashboard (or recreated chart):

- Base Ministral 3B: 23% pass rate on SWE-bench holdout
- Same model + knowledge graph: 38% pass rate
- Distilled Ministral 3B (after 3 cycles): 35% pass rate — no external calls needed

Show the hill-climbing chart: eval scores improving Cycle 1 → 2 → 3.
"This 3 billion parameter model running on a single GPU is now smarter than the base model because it absorbed the collective experience of the entire network."

### Minute 4:00-5:00 — The Business Vision

"For enterprises, this deploys privately. All agent knowledge stays inside the company. The company's distilled model becomes a proprietary asset encoding their institutional knowledge."

"For the open-source community, this is a public commons. Every agent that connects makes the network smarter. The distilled models are open-source on HuggingFace."

"This is only possible with open-weight models. You cannot fine-tune GPT-4 on collective patterns. Mistral's open models are the structural foundation that makes collective intelligence possible."

---

## 8. PROJECT STRUCTURE

```
project-root/
├── README.md                    # Project overview for hackathon submission
├── PROJECT_SPEC.md              # This file
├── requirements.txt             # Python dependencies
├── .env.example                 # Template for API keys
│
├── data/
│   ├── seed_patterns.jsonl      # Pre-prepared seed data (3K-5K patterns)
│   ├── eval_holdout.jsonl       # SWE-bench holdout for evaluation (50-100 problems)
│   └── scripts/
│       ├── harvest_github.py    # Script to pull GitHub issues
│       ├── transform_nvidia.py  # Transform OpenCodeReasoning to pattern format
│       ├── transform_swebench.py # Transform SWE-bench to pattern format
│       ├── generate_synthetic.py # Generate demo-specific patterns with Mistral
│       └── combine_dataset.py   # Merge all sources into final JSONL
│
├── backend/
│   ├── main.py                  # FastAPI server — REST API + WebSocket + MCP
│   ├── config.py                # Environment variables, API keys, settings
│   ├── qdrant_client.py         # Qdrant connection, indexing, search functions
│   ├── embeddings.py            # Embedding generation (Mistral Embed or Jina)
│   ├── mcp_server.py            # MCP tool definitions (search, submit, check_distilled)
│   ├── arena.py                 # Multi-agent arena — parallel dispatch and comparison
│   ├── agents/
│   │   ├── gateway_agent.py     # Strands Gateway Agent — orchestrates the three-tier query
│   │   ├── abstraction_agent.py # Strands Abstraction Agent — strips proprietary details
│   │   ├── verification_agent.py # Strands Verification Agent — sandbox testing
│   │   └── curator_agent.py     # Strands Curator Agent — selects patterns for distillation
│   └── models/
│       ├── pattern.py           # Pydantic model for pattern schema
│       ├── arena_result.py      # Pydantic model for arena comparison result
│       └── contribution.py      # Pydantic model for solution contribution
│
├── distillation/
│   ├── distill.py               # Main fine-tuning script (Unsloth + W&B)
│   ├── export_training_data.py  # Export patterns from Qdrant as instruction-completion JSONL
│   ├── evaluate.py              # Run evaluation on SWE-bench holdout (W&B Weave)
│   └── compare_models.py        # Compare base vs distilled model performance
│
├── dashboard/
│   ├── app.jsx                  # React dashboard (single file for hackathon speed)
│   │                            # OR: Next.js app if time allows
│   ├── index.html               # Entry point
│   └── components/
│       ├── ProblemFeed.jsx      # Problem list with solution cards
│       ├── ArenaComparison.jsx  # Side-by-side agent comparison
│       ├── MetricsPanel.jsx     # Token savings, pattern count, resolution distribution
│       ├── DistillationChart.jsx # Hill-climbing eval chart
│       └── LiveFeed.jsx         # Real-time activity stream
│
└── docker-compose.yml           # Qdrant container setup
```

---

## 9. API KEYS AND SERVICES NEEDED

```bash
# .env file — fill these in at hackathon start

# AWS (from hackathon credits)
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=us-east-1

# Nvidia
NVIDIA_API_KEY=           # From build.nvidia.com
BREV_API_KEY=             # From Brev platform

# W&B
WANDB_API_KEY=            # From wandb.ai

# HuggingFace
HF_TOKEN=                 # For pushing models to Hub

# Multi-Agent Arena (bring your own keys or use free tiers)
ANTHROPIC_API_KEY=        # For Claude in arena
OPENAI_API_KEY=           # For GPT-4 in arena
DEEPSEEK_API_KEY=         # For DeepSeek in arena

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Jina (if using Jina embeddings instead of Mistral Embed)
JINA_API_KEY=
```

---

## 10. STEP-BY-STEP BUILD ORDER (Solo Builder)

This is the exact order to build things. Each step produces a demoable increment — if you run out of time at any step, you still have a working demo of everything completed before it.

### Pre-Hackathon (Do This NOW)

**Step 0A: Prepare Seed Data**

1. Download OpenCodeReasoning subset from HuggingFace (3K entries, medium difficulty)
2. Download SWE-bench Lite from HuggingFace (300 entries)
3. Run GitHub Issues harvesting script (500-1K issues from FastAPI, LangChain, Django, Transformers)
4. Generate 100-200 synthetic demo-specific patterns with Mistral API
5. Run transformation pipeline on all sources → combine into `seed_patterns.jsonl`
6. Split out 50-100 SWE-bench problems as `eval_holdout.jsonl`

**Step 0B: Prepare Fine-Tuning Script**

1. Write `distill.py` using Unsloth targeting Ministral 3B
2. Test on 10 examples locally to verify no errors
3. Add W&B logging (wandb.init + report_to="wandb")
4. Save script — ready to run on Brev during hackathon

**Step 0C: Set Up Accounts**

1. HuggingFace organization membership (required before Friday evening)
2. W&B account and project structure
3. Nvidia Brev account
4. AWS account with Bedrock access

### Saturday Morning (Hours 1-3): Foundation

**Step 1: Knowledge Graph (1.5 hours)**

1. `docker-compose up` to start Qdrant
2. Write `qdrant_client.py` — create collection, index seed data, search function
3. Write `embeddings.py` — embed patterns and queries
4. Index all seed data from `seed_patterns.jsonl`
5. Test: search for "webhook timeout" and verify relevant patterns come back
6. **Checkpoint:** You now have a populated, searchable knowledge base

**Step 2: MCP Server (1.5 hours)**

1. Write `mcp_server.py` with `search_patterns` and `submit_solution` tools
2. Write `main.py` FastAPI server that hosts the MCP endpoint
3. Test: connect a local agent (or curl) and query the MCP server
4. **Checkpoint:** Any MCP-compatible agent can now connect and get answers

### Saturday Midday (Hours 3-6): Core Features

**Step 3: Gateway Agent (1.5 hours)**

1. Write `gateway_agent.py` using Strands SDK
2. Implement the three-tier query: check distilled model → search Qdrant → fall back to reasoning
3. For now, skip the distilled model check (model not trained yet) — just do Qdrant → fallback
4. Connect to Bedrock for Mistral Large 3 reasoning
5. **Checkpoint:** The system intelligently routes between knowledge graph and fresh reasoning

**Step 4: Multi-Agent Arena (1.5 hours)**

1. Write `arena.py` — parallel dispatch to multiple model APIs
2. Implement timing and token tracking for each model
3. Implement judge step (Mistral Large 3 evaluates solutions)
4. Wire into Gateway Agent as the fallback path
5. **Checkpoint:** When no pattern exists, multiple agents race and the best solution wins

### Saturday Afternoon (Hours 6-10): Dashboard + Polish

**Step 5: Web Dashboard (2-3 hours)**

1. Build React dashboard with problem feed, arena comparison view, metrics panel
2. Connect to backend via REST API and WebSocket
3. Show real-time updates as problems come in and get resolved
4. **Checkpoint:** Judges can see the visual demo — problems, solutions, comparisons, metrics

**Step 6: Contribution Pipeline (1 hour)**

1. Write `abstraction_agent.py` — strips proprietary details from solutions
2. Wire `submit_solution` MCP tool to process contributions through guardrails → abstraction → Qdrant
3. **Checkpoint:** The knowledge base grows as agents solve new problems

### Saturday Evening (Hours 10-14): Distillation

**Step 7: Kick Off Distillation Cycle 1 (1 hour setup, then runs in background)**

1. Export patterns from Qdrant as training data using `export_training_data.py`
2. Spin up Brev GPU instance
3. Upload training data and `distill.py` to Brev
4. Start training — Ministral 3B with LoRA, ~2K-3K examples, 3 epochs
5. W&B tracks automatically — go work on other things while it trains
6. Training takes ~1-2 hours on A10G GPU, costs ~$10-15

**Step 8: Integrate Distilled Model (1 hour, after training completes)**

1. Download trained LoRA adapter from Brev
2. Push to HuggingFace Hub
3. Set up inference endpoint (build.nvidia.com or serve locally)
4. Wire `check_distilled_model` MCP tool to call the distilled model
5. Update Gateway Agent to check distilled model first
6. **Checkpoint:** The three-tier system is complete — distilled model → knowledge graph → arena

### Sunday Morning (Hours 14-20): Evaluation + Polish

**Step 9: Run Evaluation (2 hours)**

1. Write `evaluate.py` using W&B Weave
2. Run SWE-bench holdout through three configurations:
   - Base Ministral 3B (no help)
   - Base Ministral 3B + knowledge graph access
   - Distilled Ministral 3B (no external calls)
3. Track pass rate, token usage, time per problem for each configuration
4. **Checkpoint:** You have quantitative proof that the system works

**Step 10: Second Distillation Cycle (optional, 1 hour)**

1. If new patterns were contributed by arena during testing, export updated training data
2. Run Cycle 2 on Brev with enriched dataset
3. Compare Cycle 2 vs Cycle 1 metrics in W&B
4. Push updated model to HuggingFace Hub
5. **Checkpoint:** The hill-climbing chart shows improvement across cycles

**Step 11: Dashboard Polish + Demo Prep (2-3 hours)**

1. Add distillation chart to dashboard (from W&B data or recreated)
2. Polish the UI — make it look professional
3. Prepare 3-4 demo problems that showcase different paths (pattern match, arena, distilled model)
4. Practice the 5-minute pitch 3-4 times
5. Pre-record a backup video of the demo working

### Sunday Afternoon (Hours 20-24): Final Prep

**Step 12: Submission (1-2 hours)**

1. Write README.md with project overview, architecture diagram, setup instructions
2. Push all code to GitHub
3. Push all models to HuggingFace Hub
4. Create W&B Report summarizing the distillation journey
5. Final demo practice with live system
6. Have fallback plan if something breaks during pitch

---

## 11. RISK MITIGATION

| Risk                                                | Mitigation                                                                                                                                                                        |
| --------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Fine-tuning does not produce meaningful improvement | Knowledge graph retrieval alone shows 60%+ efficiency gain. Distillation is bonus, not foundation. Demo the retrieval path as primary, distillation as "and it gets even better." |
| Demo breaks live during pitch                       | Pre-record backup video. Practice with both live and recorded versions. Have 3 demo problems ready — if one fails, move to the next.                                              |
| Qdrant has issues                                   | Qdrant is extremely simple (single Docker container). If Docker fails, fall back to in-memory Qdrant (`QdrantClient(":memory:")`) which needs no container at all.                |
| Multi-agent arena is slow                           | Set a timeout (30 seconds per model). If a model does not respond, skip it. Arena is not required for every query — it is the fallback path.                                      |
| W&B tracking adds overhead                          | W&B logging is asynchronous. Minimal latency impact. But test during Saturday morning to confirm.                                                                                 |
| API rate limits during demo                         | Cache arena results for demo problems. Pre-run the arena on demo problems and store results. Show cached results if live calls fail.                                              |
| Embedding model is slow                             | Pre-embed all seed data during indexing (not at query time). Query embedding is a single API call (~200ms).                                                                       |
| Brev GPU instance takes too long to start           | Start the instance early Saturday evening. Brev instances take 2-5 minutes to spin up. Have HuggingFace Jobs as backup compute.                                                   |

---

## 12. DEPENDENCIES

### Python (backend)

```
fastapi
uvicorn
websockets
qdrant-client
strands-agents
strands-agents-tools
boto3                    # AWS Bedrock
wandb
weave
httpx                    # Async HTTP for arena calls
pydantic
python-dotenv
```

### Python (distillation — runs on Brev, not locally)

```
unsloth
transformers
datasets
trl
wandb
torch
peft
accelerate
bitsandbytes
```

### JavaScript (dashboard)

```
react
recharts                 # For charts
lucide-react             # For icons
tailwindcss
```

### Docker

```yaml
# docker-compose.yml
version: "3.8"
services:
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
volumes:
  qdrant_data:
```

---

## 13. KEY DESIGN DECISIONS AND RATIONALE

**Why Qdrant over Elasticsearch:**
Qdrant is vector-native — it was built from the ground up for semantic search. HackOverflow used Supabase with pgvector and Elasticsearch-style search. Using Qdrant is architecturally different (not a copy), simpler to set up (Docker one-liner), and better suited for agent queries where semantic similarity matters more than keyword matching. Agents describe problems in natural language; finding the right pattern requires understanding meaning, not matching exact words.

**Why Strands for ALL agents (not NeMo):**
Solo builder should use one agent framework for everything. Strands is AWS-native (earns AWS sponsor points), has first-class MCP support, and integrates cleanly with Bedrock. NeMo would earn Nvidia points but adds a second framework to learn and debug — not worth it for solo. Nvidia points come from Brev, OpenCodeReasoning, and build.nvidia.com instead.

**Why Ministral 3B for distillation (not 8B):**
Three reasons. First, Nvidia track awards bonus for lower compute — 3B is dramatically cheaper. Second, a 3B model that outperforms base 8B because of distilled collective intelligence is a more impressive demo. Third, 3B can genuinely run on-device (laptop GPU), enabling the privacy story.

**Why MCP as the interface:**
MCP is becoming the standard for agent-tool communication. Claude, Mistral Vibe, and most agent frameworks support it. By making the system accessible via MCP, any agent can connect with a single URL — zero SDK installation, zero custom integration. This makes the network effect story credible.

**Why multi-agent arena:**
It is the most visually compelling feature for judges. Seeing 4 models race to solve a problem with real-time metrics is memorable. It also produces genuinely better solutions (best-of-4 is better than any single model) and generates training data for future distillation.
