# Portfolio Project Analysis
## Comprehensive Evaluation for MLOps/AI Engineer Roles

**Author:** Ayush
**Date:** December 2024
**Goal:** Select and build a "Hero Project" that differentiates from typical applicants

---

## Table of Contents

1. [Current Portfolio Assessment](#current-portfolio-assessment)
2. [Project Option 1: SAGE (Self-Adaptive Generative Engine)](#project-option-1-sage)
3. [Project Option 2: Sentinel (LLM Observability Platform)](#project-option-2-sentinel)
4. [Project Option 3: Forge (Evaluation-Driven Prompt Engineering)](#project-option-3-forge)
5. [Project Option 4: Mirror (Production Data Replay)](#project-option-4-mirror)
6. [Comparative Analysis](#comparative-analysis)
7. [Hiring Manager Perspective](#hiring-manager-perspective)
8. [Final Recommendation](#final-recommendation)

---

## Current Portfolio Assessment

### Existing Projects

| Project | Category | What It Demonstrates |
|---------|----------|---------------------|
| **CodePilot** | AI Application | LLM integration, code generation, user-facing product |
| **VerbaQuery** | RAG/NLP | Retrieval systems, embeddings, document processing |
| **10 Open Source PRs** | Collaboration | Code review, large codebase navigation, community contribution |

### Gap Analysis

```
What You Have:
✓ AI/LLM application development
✓ RAG implementation experience
✓ Open source collaboration

What's Missing:
✗ Production operations (MLOps)
✗ Infrastructure-as-Code
✗ Monitoring and observability
✗ CI/CD for ML systems
✗ Safe deployment strategies
✗ Systematic evaluation
```

### The Problem

Most Master's graduates have:
- Titanic/MNIST classification projects
- Basic "Chat with PDF" RAG demos
- Jupyter notebooks with no deployment story

**Your goal:** Show you can take a model from prototype → production with proper operations.

---

## Project Option 1: SAGE

### Self-Adaptive Generative Engine

**One-liner:** A production RAG system that improves itself based on user feedback, with automated retraining and canary deployments.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                            │
│                   (Streamlit Chat + Feedback)                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API LAYER (FastAPI)                         │
│               /query   /feedback   /health   /metrics            │
└────────────────────────────┬────────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────────┐
│  RAG Core   │    │  Feedback   │    │  Canary Router  │
│ (LangChain) │    │  Collector  │    │  (A/B routing)  │
└──────┬──────┘    └──────┬──────┘    └────────┬────────┘
       │                  │                    │
       ▼                  ▼                    ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────────┐
│   Qdrant    │    │ PostgreSQL  │    │     MLflow      │
│ (Vectors)   │    │ (Feedback)  │    │   (Registry)    │
└─────────────┘    └─────────────┘    └─────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PIPELINE ORCHESTRATION (Mage.ai)               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ Ingest   │→ │ Retrain  │→ │  Deploy  │→ │ Validate │        │
│  │ Feedback │  │ Embeds   │  │  Canary  │  │  Canary  │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     OBSERVABILITY STACK                          │
│          Prometheus  →  Grafana  →  Evidently AI                 │
└─────────────────────────────────────────────────────────────────┘
```

### Tech Stack

| Layer | Technology | Justification |
|-------|------------|---------------|
| UI | Streamlit | Fast to build, good for demos |
| API | FastAPI | Industry standard, async, auto-docs |
| RAG | LangChain + OpenAI | Flexible, well-documented |
| Vector DB | Qdrant | Open-source, good performance |
| Feedback Storage | PostgreSQL | Reliable, queryable |
| Orchestration | Mage.ai | Modern alternative to Airflow |
| Experiment Tracking | MLflow | Industry standard |
| Monitoring | Prometheus + Grafana | Universal observability |
| Drift Detection | Evidently AI | Purpose-built for ML |
| Containers | Docker Compose | Simple local development |
| IaC | Terraform (documented) | Shows production thinking |

### Implementation Timeline

| Week | Milestone | Deliverables |
|------|-----------|--------------|
| 1 | Foundation | FastAPI + RAG + Qdrant working |
| 2 | Feedback System | Collection UI + PostgreSQL + Analytics |
| 3 | Pipelines | Mage.ai + Retrain pipeline + MLflow |
| 4 | Canary + Observability | A/B routing + Prometheus + Grafana |
| 5 | Polish | Tests + Documentation + Demo video |

### Key Features

**1. Feedback Collection**
```python
@router.post("/feedback")
async def submit_feedback(feedback: FeedbackCreate):
    db.store(feedback)

    if feedback.rating == "negative":
        flag_document(feedback.source_doc_id)

    satisfaction = calc_satisfaction_rate(hours=24)
    if satisfaction < 0.80:
        trigger_retrain_pipeline()
```

**2. Canary Routing**
```python
class CanaryRouter:
    def route(self, request_id: str) -> str:
        # Consistent hashing for user stickiness
        if hash(request_id) % 100 < 10:  # 10% canary
            return "canary"
        return "stable"
```

**3. Automated Retrain Pipeline**
```
Trigger: satisfaction_rate < 0.80 OR manual
Steps:
  1. Load negative feedback from PostgreSQL
  2. Identify flagged documents
  3. Re-embed with updated content
  4. Update Qdrant collection
  5. Register new version in MLflow
  6. Deploy as canary (10% traffic)
  7. Monitor for 24 hours
  8. Auto-promote if metrics improve
```

### Strengths

- Shows end-to-end MLOps thinking
- Demonstrates understanding of production concerns
- Unique angle (feedback-driven improvement)
- Strong interview talking points

### Weaknesses

- High complexity for solo developer
- Canary validation logic is tricky
- Many moving parts to maintain
- Risk of "tutorial code stitched together" perception

### Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Canary logic incomplete | High | Medium | Simplify to time-window A/B |
| Too many tools, shallow knowledge | Medium | High | Reduce stack, go deep |
| Demo fails during interview | Medium | High | Record backup video |
| Can't explain trade-offs | Low | Very High | Write decision docs |

---

## Project Option 2: Sentinel

### LLM Observability Platform

**One-liner:** An open-source observability layer that captures every LLM call, tracks cost/latency/tokens, and provides debugging dashboards.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    YOUR LLM APPLICATION                          │
│                  (Any app using OpenAI, etc.)                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SENTINEL SDK (Python)                         │
│     sentinel.wrap(openai_client)  # Instruments all calls       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   COLLECTOR API (FastAPI)                        │
│                    POST /v1/traces                               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  STORAGE (PostgreSQL + TimescaleDB)              │
│   traces, spans, metrics, costs, latencies, token_counts        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   VISUALIZATION (Grafana)                        │
│   - Cost per user/day                                           │
│   - Latency P50/P95/P99                                         │
│   - Token usage trends                                          │
│   - Error rate monitoring                                       │
│   - Trace explorer (request → response debugging)               │
└─────────────────────────────────────────────────────────────────┘
```

### Tech Stack

| Component | Technology | Justification |
|-----------|------------|---------------|
| SDK | Python package | Easy integration |
| Collector | FastAPI | High throughput, async |
| Storage | PostgreSQL + TimescaleDB | Time-series optimized |
| Visualization | Grafana | Industry standard |
| Tracing | OpenTelemetry compatible | Interoperability |
| Deployment | Docker Compose | Simple setup |

### Implementation Timeline

| Week | Milestone | Deliverables |
|------|-----------|--------------|
| 1 | SDK + Collector | Python wrapper, FastAPI endpoint |
| 2 | Storage + Queries | TimescaleDB schema, aggregation queries |
| 3 | Dashboards | Grafana dashboards, alerting rules |
| 4 | Polish | Documentation, example app, demo |

### Key Features

**1. SDK Instrumentation**
```python
from sentinel import Sentinel

sentinel = Sentinel(api_key="...")
client = sentinel.wrap(OpenAI())

# All calls are now automatically traced
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}]
)

# Sentinel captures:
# - Request/response
# - Latency
# - Token counts
# - Cost
# - Model used
# - User ID (if provided)
```

**2. Cost Tracking Dashboard**
```
┌─────────────────────────────────────────────────┐
│  Daily LLM Spend                                │
│  ─────────────────                              │
│  Today: $47.23                                  │
│  This Week: $312.87                             │
│  This Month: $1,247.33                          │
│                                                 │
│  Top Users:                                     │
│  1. user_123: $89.44                           │
│  2. user_456: $67.21                           │
│  3. user_789: $45.12                           │
└─────────────────────────────────────────────────┘
```

**3. Latency Monitoring**
```
┌─────────────────────────────────────────────────┐
│  Response Latency (last 24h)                    │
│  ────────────────────────────                   │
│  P50: 1.2s                                      │
│  P95: 3.4s                                      │
│  P99: 8.7s                                      │
│                                                 │
│  [Alert] P99 exceeded 5s threshold at 14:32    │
└─────────────────────────────────────────────────┘
```

### Strengths

- Solves immediate pain point (cost/debugging)
- Lower complexity than SAGE
- Clear value proposition
- Applicable to any LLM app

### Weaknesses

- Similar tools exist (Langfuse, LangSmith)
- Less "novel" than feedback loop concept
- More "infrastructure" than "AI" work

### Differentiation from Langfuse

| Feature | Langfuse | Sentinel (Yours) |
|---------|----------|------------------|
| Hosting | Cloud or self-host | Self-host only (simpler) |
| Pricing | Freemium | Open source |
| Focus | Full platform | Lightweight, focused |
| Setup | Complex | Docker Compose, 5 min |

**Your angle:** "Langfuse for teams that just want observability without the platform complexity."

---

## Project Option 3: Forge

### Evaluation-Driven Prompt Engineering

**One-liner:** A prompt versioning and evaluation system that tracks prompt performance over time and ensures you never deploy a regression.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      FORGE WEB UI (Streamlit)                    │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   Prompts   │  │  Test Cases │  │   Results   │              │
│  │   Editor    │  │   Manager   │  │  Dashboard  │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       FORGE API (FastAPI)                        │
│                                                                  │
│   POST /prompts              - Create/update prompts            │
│   POST /test-cases           - Define test cases                │
│   POST /evaluate             - Run evaluation                   │
│   GET  /results/{prompt_id}  - Get eval history                 │
│   POST /promote              - Promote prompt to production     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EVALUATION ENGINE                             │
│                                                                  │
│   For each (prompt_version, test_case):                         │
│     1. Run prompt against LLM                                   │
│     2. Compare output to expected (exact, semantic, LLM-judge)  │
│     3. Calculate score                                          │
│     4. Store result                                             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STORAGE (PostgreSQL)                          │
│                                                                  │
│   prompts: id, version, content, created_at, is_production     │
│   test_cases: id, input, expected_output, eval_method          │
│   eval_runs: id, prompt_id, timestamp, overall_score           │
│   eval_results: id, run_id, test_case_id, output, score        │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CI/CD INTEGRATION                             │
│                                                                  │
│   GitHub Actions:                                               │
│     - On PR: Run eval, post score as comment                   │
│     - Nightly: Run full eval suite, alert on regression        │
│     - On merge: Auto-promote if score >= threshold             │
└─────────────────────────────────────────────────────────────────┘
```

### Tech Stack

| Component | Technology | Justification |
|-----------|------------|---------------|
| UI | Streamlit | Quick to build |
| API | FastAPI | Simple, fast |
| Database | PostgreSQL | Reliable, queryable |
| Eval Methods | Custom + RAGAS | Flexible scoring |
| CI/CD | GitHub Actions | Standard, free |
| LLM | OpenAI/Anthropic | Your choice |

### Implementation Timeline

| Week | Milestone | Deliverables |
|------|-----------|--------------|
| 1 | Core System | Prompt CRUD, test case CRUD, basic eval |
| 2 | Eval Engine | Multiple eval methods, scoring, storage |
| 3 | CI/CD | GitHub Actions integration, PR comments |
| 4 | Polish | UI improvements, documentation, demo |

### Key Features

**1. Prompt Versioning**
```python
# Create a new prompt version
prompt = forge.create_prompt(
    name="summarizer",
    version="v3",
    content="""
    Summarize the following text in 2-3 sentences.
    Focus on the main argument and key evidence.

    Text: {text}
    """,
)
```

**2. Test Case Definition**
```python
# Define test cases with expected outputs
forge.add_test_case(
    prompt_name="summarizer",
    input={"text": "Long article about climate change..."},
    expected_output="Climate change is accelerating...",
    eval_method="semantic_similarity",  # or "exact", "llm_judge"
    threshold=0.85,
)
```

**3. Evaluation Run**
```python
# Run evaluation
results = forge.evaluate(prompt_name="summarizer", version="v3")

print(results)
# {
#   "prompt": "summarizer",
#   "version": "v3",
#   "overall_score": 0.87,
#   "test_cases": [
#     {"id": 1, "score": 0.92, "passed": True},
#     {"id": 2, "score": 0.81, "passed": False},
#     {"id": 3, "score": 0.88, "passed": True},
#   ],
#   "recommendation": "PROMOTE"  # or "REJECT"
# }
```

**4. GitHub Actions Integration**
```yaml
# .github/workflows/prompt-eval.yml
name: Prompt Evaluation

on:
  pull_request:
    paths:
      - 'prompts/**'

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Run Forge Evaluation
        run: |
          forge evaluate --all --output results.json

      - name: Post Results to PR
        uses: actions/github-script@v6
        with:
          script: |
            const results = require('./results.json');
            const body = `## Prompt Evaluation Results

            | Prompt | Version | Score | Status |
            |--------|---------|-------|--------|
            ${results.map(r =>
              `| ${r.name} | ${r.version} | ${r.score} | ${r.passed ? '✅' : '❌'} |`
            ).join('\n')}
            `;
            github.rest.issues.createComment({
              owner: context.repo.owner,
              repo: context.repo.repo,
              issue_number: context.issue.number,
              body: body
            });
```

### Strengths

- Lowest complexity of all options
- Finishable in 2-3 weeks
- Highly demo-able
- Directly relevant to LLM jobs
- CI/CD integration is impressive

### Weaknesses

- Less "MLOps infrastructure" focused
- Doesn't show distributed systems skills
- Some similar tools exist (promptfoo)

### Differentiation

Your angle: **"Prompt engineering with CI/CD discipline."**

Most prompt tools are notebooks or playgrounds. Forge treats prompts like code:
- Version control
- Automated testing
- Deployment gates
- Regression prevention

---

## Project Option 4: Mirror

### Production Data Replay for ML

**One-liner:** Capture production inference requests and replay them against new model versions to detect regressions before deployment.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRODUCTION ML SERVICE                         │
│                    (Your existing API)                           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   MIRROR MIDDLEWARE (FastAPI)                    │
│                                                                  │
│   1. Intercept request                                          │
│   2. Forward to model                                           │
│   3. Capture request + response                                 │
│   4. Store in replay buffer                                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   REPLAY STORAGE (S3/MinIO)                      │
│                                                                  │
│   replay_buffer/                                                │
│   ├── 2024-12-28/                                              │
│   │   ├── request_001.json                                     │
│   │   ├── request_002.json                                     │
│   │   └── ...                                                  │
│   └── 2024-12-27/                                              │
│       └── ...                                                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    REPLAY ENGINE (CLI)                           │
│                                                                  │
│   mirror replay \                                               │
│     --model-v1 http://localhost:8001 \                         │
│     --model-v2 http://localhost:8002 \                         │
│     --date 2024-12-28 \                                        │
│     --output diff_report.html                                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DIFF REPORT (HTML)                            │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Replay Summary                                         │   │
│   │  ─────────────                                          │   │
│   │  Total requests: 10,000                                 │   │
│   │  Identical outputs: 9,650 (96.5%)                       │   │
│   │  Minor diff: 280 (2.8%)                                 │   │
│   │  Major diff: 70 (0.7%)  ← INVESTIGATE                  │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   [Expandable list of major diffs with side-by-side view]      │
└─────────────────────────────────────────────────────────────────┘
```

### Tech Stack

| Component | Technology | Justification |
|-----------|------------|---------------|
| Middleware | FastAPI | Easy to integrate |
| Storage | MinIO (S3-compatible) | Simple, local-friendly |
| Replay Engine | Python CLI (Click) | Scriptable |
| Diff Visualization | HTML report | No deps needed |
| Model Versioning | MLflow | Industry standard |

### Implementation Timeline

| Week | Milestone | Deliverables |
|------|-----------|--------------|
| 1 | Capture Middleware | Request interception, storage |
| 2 | Replay Engine | CLI tool, parallel replay |
| 3 | Diff Analysis | Comparison logic, HTML report |
| 4 | Integration | MLflow integration, docs, demo |

### Key Features

**1. Request Capture**
```python
from mirror import MirrorMiddleware

app = FastAPI()
app.add_middleware(
    MirrorMiddleware,
    storage_path="s3://mirror-bucket/replay",
    sample_rate=1.0,  # Capture 100% of requests
)

@app.post("/predict")
async def predict(request: PredictRequest):
    # Your normal prediction logic
    return model.predict(request)
```

**2. Replay CLI**
```bash
# Replay yesterday's traffic against new model
mirror replay \
  --source s3://mirror-bucket/replay/2024-12-27 \
  --baseline http://model-v1:8000/predict \
  --candidate http://model-v2:8000/predict \
  --output report.html \
  --parallel 10

# Output:
# Replaying 10,000 requests...
# [████████████████████████] 100%
#
# Results:
#   Identical: 9,650 (96.5%)
#   Minor diff: 280 (2.8%)
#   Major diff: 70 (0.7%)
#
# Report saved to: report.html
```

**3. Diff Report**
```
┌─────────────────────────────────────────────────────────────────┐
│  Request #4521 - MAJOR DIFF                                     │
├─────────────────────────────────────────────────────────────────┤
│  Input:                                                         │
│  {"text": "The bank was closed on Sunday..."}                  │
│                                                                 │
│  Baseline (v1):                                                 │
│  {"label": "financial", "confidence": 0.89}                    │
│                                                                 │
│  Candidate (v2):                                                │
│  {"label": "geography", "confidence": 0.72}  ← WRONG           │
│                                                                 │
│  Analysis: Model v2 misclassified "bank" (river bank vs        │
│  financial institution). Regression in disambiguation.         │
└─────────────────────────────────────────────────────────────────┘
```

### Strengths

- Novel concept (not many open-source tools do this)
- Shows deep understanding of ML deployment risks
- Applicable to any ML model (not just LLMs)
- "Safety engineering" is a strong interview story

### Weaknesses

- Requires having a model to test against
- Storage costs for high-traffic systems
- More niche use case

---

## Comparative Analysis

### Complexity vs. Impact Matrix

```
                    HIGH IMPACT
                         │
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          │   MIRROR     │    SAGE      │
          │              │              │
          │              │              │
LOW ──────┼──────────────┼──────────────┼────── HIGH
COMPLEXITY│              │              │     COMPLEXITY
          │              │              │
          │   FORGE      │  SENTINEL    │
          │              │              │
          └──────────────┼──────────────┘
                         │
                         │
                    LOW IMPACT
```

### Feature Comparison

| Feature | SAGE | Sentinel | Forge | Mirror |
|---------|------|----------|-------|--------|
| Shows MLOps skills | ✅✅✅ | ✅✅ | ✅ | ✅✅ |
| Shows AI/LLM skills | ✅✅ | ✅ | ✅✅✅ | ✅ |
| Shows infra skills | ✅✅ | ✅✅ | ✅ | ✅✅ |
| Finishable in 4 weeks | ⚠️ | ✅ | ✅✅ | ✅ |
| Demo-able in 2 min | ✅ | ✅✅ | ✅✅✅ | ✅ |
| Novel/unique | ✅✅ | ✅ | ✅ | ✅✅✅ |
| Interview talking points | ✅✅✅ | ✅✅ | ✅✅ | ✅✅ |

### Risk Assessment

| Project | Biggest Risk | Mitigation |
|---------|--------------|------------|
| SAGE | Canary logic too complex | Simplify to time-window A/B |
| Sentinel | "It's just Langfuse clone" | Focus on simplicity angle |
| Forge | "Too simple for MLOps role" | Add MLflow, CI/CD, Grafana |
| Mirror | "Niche use case" | Demo with real model comparison |

### Time Estimate (Realistic)

| Project | Minimum Viable | Polished | With Docs + Demo |
|---------|----------------|----------|------------------|
| SAGE | 4 weeks | 6 weeks | 7 weeks |
| Sentinel | 2 weeks | 3 weeks | 4 weeks |
| Forge | 2 weeks | 3 weeks | 3 weeks |
| Mirror | 3 weeks | 4 weeks | 5 weeks |

---

## Hiring Manager Perspective

### What Actually Matters

Based on reviewing hundreds of ML/AI engineering candidates:

**Top 3 things that get attention:**
1. **Can you explain trade-offs?** Why X over Y?
2. **Is the code clean?** Or is it tutorial spaghetti?
3. **Is there a demo?** 2-minute Loom > 10-page README

**Red flags that hurt candidates:**
- Too many buzzwords, can't explain any deeply
- No tests
- README is AI-generated fluff
- "Tutorial code stitched together" feel

### Interview Questions You Should Be Able to Answer

**For SAGE:**
- "Why Mage.ai over Airflow?"
- "How do you handle user session stickiness in canary routing?"
- "What happens if your retrain pipeline fails mid-way?"

**For Sentinel:**
- "Why not just use Langfuse?"
- "How do you handle high-throughput collection without dropping events?"
- "How does your cost calculation work for different models?"

**For Forge:**
- "How do you handle non-deterministic LLM outputs in testing?"
- "What's your threshold for auto-promoting a prompt?"
- "How do you prevent prompt injection in your eval?"

**For Mirror:**
- "How much storage does this need at scale?"
- "How do you define 'major' vs 'minor' diff?"
- "What about latency-sensitive production systems?"

---

## Final Recommendation

### For Maximum Impact with Manageable Risk

**Build Forge first (3 weeks), then layer on MLOps (2 weeks).**

```
Week 1-2: Forge Core
  - Prompt versioning
  - Test case management
  - Basic evaluation engine
  - PostgreSQL storage

Week 3: CI/CD + UI
  - GitHub Actions integration
  - PR comment bot
  - Streamlit dashboard

Week 4: MLOps Layer
  - MLflow for "experiment tracking" of prompts
  - Grafana dashboard for eval metrics over time
  - Scheduled nightly evals

Week 5: Polish
  - Documentation
  - Architecture decision records
  - 2-minute demo video
  - Clean up code, add tests
```

### Why This Approach

1. **You have a working project by week 3.** If time runs out, you still have something polished.

2. **It compounds your existing portfolio.**
   - CodePilot (AI app)
   - VerbaQuery (RAG)
   - Forge (evaluation/MLOps)
   = "Full-stack AI engineer who knows how to test"

3. **Low risk of failure.** No distributed systems, no complex orchestration.

4. **Easy to extend.** If an interviewer asks "what would you add?", you can describe SAGE features as "future work."

5. **Unique angle.** "CI/CD for prompts" is catchy and understandable.

### Alternative: If You Want Pure MLOps Credibility

Build a **simplified SAGE**:

```
Week 1-2: RAG + Feedback Collection
  - FastAPI + LangChain + Qdrant
  - Feedback UI (thumbs up/down)
  - PostgreSQL storage

Week 3: Observability
  - Prometheus metrics
  - Grafana dashboards
  - Satisfaction rate tracking

Week 4: Pipeline (Simplified)
  - Mage.ai pipeline that re-indexes documents
  - Manual trigger (not auto-trigger)
  - MLflow for version tracking

Week 5: Polish
  - Docker Compose full stack
  - Documentation
  - Demo video
```

**What you skip:** Canary deployments, auto-promotion, drift detection.

**What you say in interviews:** "V1 has manual promotion. V2 would add canary routing with statistical validation."

This shows you understand the full picture but were pragmatic about scope.

---

## Decision Framework

Answer these questions:

1. **How much time do you realistically have?**
   - 2-3 weeks → Forge
   - 4-5 weeks → Simplified SAGE or Mirror
   - 6+ weeks → Full SAGE

2. **What role are you targeting?**
   - "AI Engineer" → Forge (shows eval discipline)
   - "MLOps Engineer" → SAGE or Sentinel (shows infra)
   - "Full-stack ML" → Forge + MLflow + Grafana (best of both)

3. **What's your risk tolerance?**
   - Low → Forge (almost guaranteed to finish)
   - Medium → Sentinel or Mirror
   - High → Full SAGE

4. **What complements your existing portfolio best?**
   - You have RAG (VerbaQuery) → Don't build another RAG. Build Forge or Mirror.
   - You have AI apps → Show operations side with SAGE or Sentinel.

---

## Next Steps

1. **Make a decision** on which project to build
2. **Create the repository** (or repurpose ml-monitor)
3. **Set up project structure** and basic scaffolding
4. **Build in public** - commit daily, show progress
5. **Record demo video** at the end
6. **Write honest "Limitations" section** in README

---

*Document created: December 2024*
*Purpose: Portfolio project selection for MLOps/AI Engineer job search*
