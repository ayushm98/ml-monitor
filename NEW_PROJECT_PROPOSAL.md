# New Project Proposal: LLM Cost Optimizer

## The Idea: "Cascade" - Intelligent LLM Request Router

**One-liner:** A proxy layer that automatically routes LLM requests to the cheapest model capable of handling the task, with semantic caching to eliminate redundant calls.

---

## Why This Beats All 4 Previous Options

### The Problem (It's Real and Expensive)

```
Every company using LLMs faces this:

┌─────────────────────────────────────────────────────────────┐
│  Current State (Dumb Routing)                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  "What's 2+2?"  ──────────────────────►  GPT-4  ($0.03)    │
│  "Summarize this legal doc"  ─────────►  GPT-4  ($0.03)    │
│  "What's 2+2?" (again)  ──────────────►  GPT-4  ($0.03)    │
│  "Hello"  ────────────────────────────►  GPT-4  ($0.03)    │
│                                                             │
│  Monthly bill: $15,000                                      │
│  Wasted spend: ~60%                                         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  With Cascade (Smart Routing)                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  "What's 2+2?"  ──────────────────────►  GPT-3.5  ($0.001) │
│  "Summarize this legal doc"  ─────────►  GPT-4    ($0.03)  │
│  "What's 2+2?" (again)  ──────────────►  CACHE    ($0.00)  │
│  "Hello"  ────────────────────────────►  GPT-3.5  ($0.001) │
│                                                             │
│  Monthly bill: $6,200                                       │
│  Savings: 59%                                               │
└─────────────────────────────────────────────────────────────┘
```

**This is a $10K/month problem at mid-size companies.**

---

## Hiring Committee Pre-Review

Let's score this against the rubric BEFORE building:

| Criteria | Expected Score | Why |
|----------|----------------|-----|
| Engineering Maturity | 8/10 | Proxy architecture, caching layer, async handling, rate limiting |
| Data Reality | 8/10 | Use REAL query distributions from public datasets (MS MARCO, Natural Questions, ShareGPT) |
| "So What?" Factor | 9/10 | **Direct dollar savings. CFOs understand this.** |

### Why This Wins

1. **Solves a REAL problem** - Every company with LLM spend has this pain
2. **Measurable in DOLLARS** - Not accuracy, not F1, actual money saved
3. **Uses REAL data** - Public conversation datasets have realistic query distributions
4. **Novel enough** - Few good open-source solutions (Martian, Portkey are closed/paid)
5. **Shows production thinking** - Caching, fallbacks, latency budgets
6. **Finishable in 4 weeks** - Focused scope

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     YOUR APPLICATION                             │
│                  (Drop-in OpenAI replacement)                    │
│                                                                  │
│   from cascade import CascadeClient                              │
│   client = CascadeClient()  # Same interface as OpenAI          │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CASCADE PROXY (FastAPI)                       │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Semantic   │  │   Request    │  │   Cost       │          │
│  │   Cache      │  │   Classifier │  │   Tracker    │          │
│  │   (Redis +   │  │   (DistilBERT│  │   (Per-user  │          │
│  │   Embeddings)│  │   fine-tuned)│  │   reporting) │          │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘          │
│         │                 │                                      │
│         ▼                 ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   ROUTING ENGINE                         │    │
│  │                                                          │    │
│  │   if cache_hit:                                          │    │
│  │       return cached_response  # $0.00                    │    │
│  │                                                          │    │
│  │   complexity = classifier.predict(query)                 │    │
│  │                                                          │    │
│  │   if complexity == "simple":                             │    │
│  │       return gpt35(query)      # $0.001                  │    │
│  │   elif complexity == "medium":                           │    │
│  │       return gpt4_mini(query)  # $0.01                   │    │
│  │   else:                                                  │    │
│  │       return gpt4(query)       # $0.03                   │    │
│  │                                                          │    │
│  └─────────────────────────────────────────────────────────┘    │
└────────────────────────────────┬────────────────────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│   OpenAI API    │   │  Anthropic API  │   │   Local LLM     │
│   (GPT-3.5/4)   │   │  (Claude)       │   │   (Ollama)      │
└─────────────────┘   └─────────────────┘   └─────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                     OBSERVABILITY                                │
│                                                                  │
│   Prometheus Metrics:                                            │
│   - cascade_requests_total{model="gpt4", cache="miss"}          │
│   - cascade_cost_dollars{model="gpt35"}                         │
│   - cascade_latency_seconds{model="gpt4"}                       │
│   - cascade_cache_hit_rate                                       │
│                                                                  │
│   Grafana Dashboard:                                             │
│   - Cost savings this month: $4,230                             │
│   - Cache hit rate: 34%                                          │
│   - Model distribution: 60% GPT-3.5, 25% GPT-4-mini, 15% GPT-4  │
└─────────────────────────────────────────────────────────────────┘
```

---

## The Three Core Components

### 1. Semantic Cache (The Money Saver)

**Problem:** Users ask the same questions repeatedly. Why pay twice?

```python
class SemanticCache:
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.redis = Redis()
        self.qdrant = QdrantClient()

    def get(self, query: str, threshold: float = 0.92) -> Optional[str]:
        """Return cached response if semantically similar query exists."""
        embedding = self.embedder.encode(query)

        results = self.qdrant.search(
            collection="query_cache",
            query_vector=embedding,
            limit=1,
            score_threshold=threshold
        )

        if results:
            cache_key = results[0].payload["cache_key"]
            return self.redis.get(cache_key)

        return None

    def set(self, query: str, response: str, ttl: int = 3600):
        """Cache response with semantic indexing."""
        embedding = self.embedder.encode(query)
        cache_key = hashlib.sha256(query.encode()).hexdigest()

        self.redis.setex(cache_key, ttl, response)
        self.qdrant.upsert(
            collection="query_cache",
            points=[{
                "id": cache_key,
                "vector": embedding,
                "payload": {"cache_key": cache_key, "query": query}
            }]
        )
```

**Why this impresses hiring managers:**
- Shows you understand embeddings beyond "RAG tutorial"
- Cache invalidation strategy (TTL) shows production thinking
- Similarity threshold tuning is a real engineering decision

---

### 2. Complexity Classifier (The Router Brain)

**Problem:** How do you know if a query needs GPT-4 or GPT-3.5?

```python
class ComplexityClassifier:
    """
    Fine-tuned DistilBERT that classifies query complexity.

    Labels:
    - simple: Factual, short answers, basic math, greetings
    - medium: Summarization, translation, structured extraction
    - complex: Reasoning, analysis, creative writing, code generation
    """

    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "./models/complexity-classifier"
        )
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def predict(self, query: str) -> str:
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True)
        outputs = self.model(**inputs)
        predicted_class = outputs.logits.argmax().item()
        return ["simple", "medium", "complex"][predicted_class]
```

**Training Data (This is the KEY differentiator):**

You DON'T use toy data. You use:

| Dataset | Size | Why It's Real |
|---------|------|---------------|
| ShareGPT | 90K conversations | Real user queries to ChatGPT |
| LMSYS-Chat-1M | 1M conversations | Real multi-turn conversations |
| WildChat | 1M conversations | Uncensored real user queries |
| MS MARCO | 1M queries | Real search queries |

**Labeling Strategy:**
```python
# Heuristic labeling based on response characteristics
def label_complexity(query: str, response: str) -> str:
    # Simple: Short responses, factual
    if len(response.split()) < 50 and not any(
        kw in response.lower() for kw in ["however", "therefore", "analysis"]
    ):
        return "simple"

    # Complex: Long responses with reasoning markers
    if len(response.split()) > 200 or any(
        kw in response.lower() for kw in ["step 1", "first,", "let me", "```"]
    ):
        return "complex"

    return "medium"
```

Then you VALIDATE by running samples through GPT-3.5 vs GPT-4 and measuring quality delta.

---

### 3. Cost Tracker (The Business Case)

```python
class CostTracker:
    """Tracks costs per user, per model, per feature."""

    PRICING = {
        "gpt-4": {"input": 0.03, "output": 0.06},      # per 1K tokens
        "gpt-4-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "cache": {"input": 0, "output": 0},
    }

    def record(self, user_id: str, model: str, input_tokens: int, output_tokens: int):
        cost = (
            self.PRICING[model]["input"] * input_tokens / 1000 +
            self.PRICING[model]["output"] * output_tokens / 1000
        )

        # What it WOULD have cost with GPT-4
        baseline_cost = (
            self.PRICING["gpt-4"]["input"] * input_tokens / 1000 +
            self.PRICING["gpt-4"]["output"] * output_tokens / 1000
        )

        savings = baseline_cost - cost

        # Store in TimescaleDB for time-series analysis
        self.db.insert(
            user_id=user_id,
            model=model,
            actual_cost=cost,
            baseline_cost=baseline_cost,
            savings=savings,
            timestamp=datetime.utcnow()
        )
```

**Dashboard Output:**
```
┌─────────────────────────────────────────────────────────────┐
│  COST SAVINGS REPORT - December 2024                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Total Requests:        45,231                              │
│  Cache Hits:            15,892 (35.1%)                      │
│                                                             │
│  Model Distribution:                                        │
│  ████████████████████░░░░░  GPT-3.5    58%                 │
│  ██████░░░░░░░░░░░░░░░░░░░  GPT-4-mini 24%                 │
│  ████░░░░░░░░░░░░░░░░░░░░░  GPT-4      18%                 │
│                                                             │
│  Baseline Cost (all GPT-4):    $1,847.23                   │
│  Actual Cost:                    $612.44                   │
│  ─────────────────────────────────────────                 │
│  SAVINGS:                      $1,234.79  (66.8%)          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Component | Technology | Why |
|-----------|------------|-----|
| Proxy | FastAPI | Async, high throughput, OpenAI-compatible |
| Semantic Cache | Redis + Qdrant | Fast KV + vector search |
| Classifier | DistilBERT (fine-tuned) | Small, fast, accurate enough |
| Embeddings | all-MiniLM-L6-v2 | Fast, good quality |
| Cost Storage | TimescaleDB | Time-series optimized |
| Monitoring | Prometheus + Grafana | Industry standard |
| Deployment | Docker Compose | Simple, reproducible |

---

## Implementation Timeline

| Week | Milestone | Deliverables |
|------|-----------|--------------|
| 1 | Core Proxy | OpenAI-compatible proxy, basic routing |
| 2 | Semantic Cache | Qdrant integration, cache hit/miss logic |
| 3 | Classifier | Fine-tune DistilBERT on ShareGPT, integrate |
| 4 | Observability | Prometheus, Grafana, cost dashboard |
| 5 | Polish | Load testing, docs, demo video |

---

## The Interview Story

**When they ask "Tell me about a project":**

> "I built Cascade, an LLM request router that reduced API costs by 67% in my testing.
>
> The core insight was that most production LLM traffic is simple queries that don't need GPT-4. So I built a classifier trained on 100K real user conversations from ShareGPT to predict query complexity, and routed simple queries to GPT-3.5.
>
> I also added semantic caching - if someone asks 'What's the capital of France?' and someone else asks 'What is France's capital?', we return the cached response. That alone saved 35% of requests.
>
> The hardest part was tuning the similarity threshold. Too high (0.95) and you miss obvious duplicates. Too low (0.85) and you return wrong answers. I built an evaluation harness with 1,000 query pairs to find the optimal threshold of 0.92."

**Follow-up questions you can ACTUALLY answer:**

- "How did you handle cache invalidation?" → TTL + version tags for model updates
- "What about streaming responses?" → Cache only completed responses, stream directly for cache misses
- "How do you measure classifier accuracy?" → Held-out test set + A/B test comparing quality ratings
- "What if the classifier is wrong?" → Fallback logic + user feedback loop for retraining

---

## Why This Beats Mirror/SAGE/Forge/Sentinel

| Criteria | Cascade | Mirror | SAGE | Forge | Sentinel |
|----------|---------|--------|------|-------|----------|
| Real Problem | ✅ Cost optimization | ✅ Regression | ⚠️ Vague | ⚠️ Vague | ❌ Solved |
| Measurable Impact | ✅ Dollars saved | ✅ Regressions caught | ⚠️ Satisfaction? | ⚠️ Accuracy? | ❌ Nothing unique |
| Real Data | ✅ ShareGPT/LMSYS | ⚠️ Need your own | ❌ What docs? | ❌ Manual test cases | ❌ Your own calls |
| Novel | ✅ Few OSS options | ✅ Few OSS options | ⚠️ RAG is everywhere | ❌ Promptfoo exists | ❌ Langfuse exists |
| Finishable | ✅ 4-5 weeks | ✅ 4 weeks | ❌ 6+ weeks | ✅ 3 weeks | ✅ 3 weeks |

---

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Classifier accuracy too low | Medium | Start with rule-based routing, add classifier as enhancement |
| Cache returning wrong answers | Medium | Conservative threshold (0.92+), add confidence scores |
| "It's just a proxy" perception | Low | Emphasize the ML classifier and real data training |
| Scope creep | Medium | Hard stop at Week 5, polish over features |

---

## Bonus: Open Source Potential

This could actually be useful to others. Name it something catchy:

- **Cascade** - LLM requests flow down to the cheapest capable model
- **Penny** - Pinching pennies on your LLM spend
- **Arbiter** - Decides which model handles each request

If it gets GitHub stars, that's social proof for your resume.

---

## Next Steps

1. **Confirm this is the project you want to build**
2. **Set up the repository structure**
3. **Download ShareGPT dataset for classifier training**
4. **Build the proxy skeleton in Week 1**

---

*This proposal addresses every weakness identified in the hiring committee review: real data, real problem, measurable impact, production awareness, and finishable scope.*
