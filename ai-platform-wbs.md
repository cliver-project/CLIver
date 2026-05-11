# WBS: AI Training Platform (Internal Company Tool)

**Project:** `ai-platform`
**Objective:** Enable Data Scientists & ML Engineers to ingest training materials (PDFs, notebooks, Markdown) and spin up Fine-Tuning, RAG, and AI Agent skills with minimal operational overhead.
**Target Users:** Data Scientists, ML Engineers
**Timeline:** MVP in 2–3 months

---

## Phase 0: Foundation & Setup (Weeks 1–2)

### 0.1 Infrastructure & Environment
- [ ] 0.1.1 Set up cloud / on-prem Kubernetes cluster (EKS / GKE / self-managed)
- [ ] 0.1.2 Provision GPU nodes (NVIDIA A10G / A100) with CUDA drivers
- [ ] 0.1.3 Set up container registry (ECR / GCR / Harbor)
- [ ] 0.1.4 Configure CI/CD pipeline (GitHub Actions / GitLab CI)
- [ ] 0.1.5 Set up monitoring & logging stack (Prometheus + Grafana / ELK)
- [ ] 0.1.6 Configure secrets management (Vault / AWS Secrets Manager)

### 0.2 Repo & Code Standards
- [ ] 0.2.1 Create monorepo structure (backend, frontend, workflows, docs)
- [ ] 0.2.2 Set up Python environment (Poetry / uv) with linting (Ruff, mypy)
- [ ] 0.2.3 Set up React / Next.js frontend scaffold
- [ ] 0.2.4 Define API contract (OpenAPI 3.0 spec)
- [ ] 0.2.5 Set up testing framework (pytest, Playwright for e2e)

### 0.3 Authentication & Authorization
- [ ] 0.3.1 Integrate SSO (Okta / Azure AD / Google Workspace)
- [ ] 0.3.2 Define roles: Admin, Power User, Viewer
- [ ] 0.3.3 Implement RBAC at API level

---

## Phase 1: Material Ingestion Pipeline (Weeks 3–5)

### 1.1 Document Parsing & Storage
- [ ] 1.1.1 Build PDF parser (PyMuPDF / Unstructured.io) with table extraction
- [ ] 1.1.2 Build Jupyter notebook parser (extract code cells + markdown + outputs)
- [ ] 1.1.3 Build Markdown parser (frontmatter, headings, code blocks)
- [ ] 1.1.4 Normalize all parsed content into a unified `DocumentChunk` schema
- [ ] 1.1.5 Store raw files in object storage (S3 / MinIO)
- [ ] 1.1.6 Store parsed chunks in vector DB (Chroma / Pinecone / Qdrant)

### 1.2 Material Catalog & Search
- [ ] 1.2.1 Build "Materials" UI — upload, list, search, delete
- [ ] 1.2.2 Implement drag-and-drop batch upload (resumable)
- [ ] 1.2.3 Show parsing status (pending / processing / done / failed)
- [ ] 1.2.4 Full-text + semantic search over ingested materials
- [ ] 1.2.5 Tagging & metadata system (author, topic, difficulty, format)

### 1.3 Chunking & Embedding Strategy
- [ ] 1.3.1 Implement configurable chunking (semantic / recursive / fixed-size)
- [ ] 1.3.2 Select embedding model (e.g., `text-embedding-3-small` or `bge-large`)
- [ ] 1.3.3 Build embedding pipeline with batch processing
- [ ] 1.3.4 Store embeddings with metadata for retrieval

---

## Phase 2: Fine-Tuning Skill (Weeks 4–7)

### 2.1 Dataset Preparation
- [ ] 2.1.1 Convert parsed materials into fine-tuning formats (Alpaca / ShareGPT / OpenAI)
- [ ] 2.1.2 Allow user to select / filter chunks to include in training set
- [ ] 2.1.3 Build dataset preview & quality checks (dedup, length outliers, label balance)
- [ ] 2.1.4 Support train / eval / test split with configurable ratios

### 2.2 Fine-Tuning Orchestration
- [ ] 2.2.1 Integrate with training frameworks (Axolotl / Unsloth / Lit-GPT)
- [ ] 2.2.2 Define LoRA / QLoRA config presets (4-bit, rank, alpha)
- [ ] 2.2.3 Build job submission — user picks base model, dataset, hyperparams
- [ ] 2.2.4 Queue training jobs on GPU nodes (Kubernetes Job / Argo Workflows)
- [ ] 2.2.5 Stream training logs to user in real-time (WebSocket / Server-Sent Events)

### 2.3 Model Registry & Deployment
- [ ] 2.3.1 Save fine-tuned LoRA adapters + merged checkpoints to model registry (MLflow / Hugging Face)
- [ ] 2.3.2 Auto-register completed model with version tags
- [ ] 2.3.3 Deploy model as inference endpoint (TGI / vLLM / SGLang)
- [ ] 2.3.4 Run A/B evaluation against base model (automated eval harness)

### 2.4 Skill Packaging
- [ ] 2.4.1 Package fine-tuned model as a "Fine-Tune Skill" — name, description, version
- [ ] 2.4.2 Show inference playground in UI (chat or completion interface)
- [ ] 2.4.3 Allow sharing skill with team / role

---

## Phase 3: RAG Skill (Weeks 5–8)

### 3.1 Knowledge Base Builder
- [ ] 3.1.1 UI to create a "Knowledge Base" from selected materials
- [ ] 3.1.2 Configure chunking strategy per knowledge base (override defaults)
- [ ] 3.1.3 Choose embedding model per knowledge base
- [ ] 3.1.4 Add incremental sync — when source material updates, re-chunk & re-embed

### 3.2 Retrieval Pipeline
- [ ] 3.2.1 Implement hybrid search (dense + sparse via BM25 / SPLADE)
- [ ] 3.2.2 Build re-ranking step (Cohere rerank / BGE-reranker / cross-encoder)
- [ ] 3.2.3 Add query rewriting / decomposition for complex questions
- [ ] 3.2.4 Configurable retrieval parameters (top-k, similarity threshold, diversity)

### 3.3 RAG Chat Interface
- [ ] 3.3.1 Build chat UI with source citations (which doc, which chunk)
- [ ] 3.3.2 Support multi-turn conversation with conversation history
- [ ] 3.3.3 Show retrieval provenance — "why this answer" panel
- [ ] 3.3.4 Let user select base LLM (GPT-4o, Llama 3, Qwen) + knowledge base

### 3.4 RAG Evaluation
- [ ] 3.4.1 Collect user feedback (thumbs up/down per answer)
- [ ] 3.4.2 Build evaluation set from chat history
- [ ] 3.4.3 Run automated eval (faithfulness, answer relevancy, context precision)
- [ ] 3.4.4 Dashboard: retrieval latency, answer quality trends

### 3.5 Skill Packaging
- [ ] 3.5.1 Package as "RAG Skill" — linked knowledge base, model, config
- [ ] 3.5.2 Expose via API for external consumption (REST / MCP server)
- [ ] 3.5.3 Embeddable widget for internal portals (iframe / React component)

---

## Phase 4: AI Agent Skill (Weeks 7–10)

### 4.1 Agent Engine
- [ ] 4.1.1 Choose agent framework (LangGraph / CrewAI / AutoGen)
- [ ] 4.1.2 Implement ReAct loop with tool-calling support
- [ ] 4.1.3 Build tool registry — define, version, and share tools
- [ ] 4.1.4 Built-in tools: web search, code execution (sandbox), file I/O, SQL query

### 4.2 Skill Authoring
- [ ] 4.2.1 UI to create an "Agent Skill" from training materials
- [ ] 4.2.2 Auto-extract tools & patterns from Jupyter notebooks (function definitions, API calls)
- [ ] 4.2.3 Let user define system prompt, available tools, guardrails
- [ ] 4.2.4 Support multi-agent topologies (supervisor + workers, sequential, parallel)
- [ ] 4.2.5 Add human-in-the-loop approval steps

### 4.3 Tool Integration from Materials
- [ ] 4.3.1 Parse Python function signatures from notebooks → convert to tool definitions
- [ ] 4.3.2 Parse REST API examples from Markdown → convert to OpenAPI tool specs
- [ ] 4.3.3 Allow user to manually register custom tools (any REST / gRPC endpoint)
- [ ] 4.3.4 Sandboxed code execution environment (gVisor / Firecracker / Docker)

### 4.4 Agent Testing & Observability
- [ ] 4.4.1 Agent playground — chat with agent, see full reasoning traces
- [ ] 4.4.2 Trace viewer (LangSmith / LangFuse / custom): tool calls, tokens, latency
- [ ] 4.4.3 Run batch evaluation against test cases
- [ ] 4.4.4 Cost tracking per agent run (LLM tokens + tool execution)

### 4.5 Skill Packaging
- [ ] 4.5.1 Package as "Agent Skill" — agent config, tools, guardrails
- [ ] 4.5.2 Deploy as persistent agent endpoint (WebSocket / SSE)
- [ ] 4.5.3 Schedule recurring agent runs (cron / event-driven)

---

## Phase 5: Platform Integration & UX (Weeks 8–11)

### 5.1 Central Dashboard
- [ ] 5.1.1 Build unified dashboard — list all skills (Fine-Tune, RAG, Agent)
- [ ] 5.1.2 Skill marketplace view — browse, search, filter skills
- [ ] 5.1.3 One-click "deploy" or "use" for any skill
- [ ] 5.1.4 Usage metrics per skill (calls, latency, cost, user feedback)

### 5.2 User Management
- [ ] 5.2.1 Team management — invite, remove, role assignment
- [ ] 5.2.2 Usage quotas per user / team
- [ ] 5.2.3 Audit log — who created/deployed/modified what

### 5.3 Cost & Resource Tracking
- [ ] 5.3.1 Track GPU hours per fine-tuning job
- [ ] 5.3.2 Track inference costs per skill (RAG, Agent, fine-tuned model)
- [ ] 5.3.3 Show cost breakdown in dashboard per team / per project

### 5.4 Notifications & Alerts
- [ ] 5.4.1 Notify user on job completion (fine-tuning done, indexing done)
- [ ] 5.4.2 Alert on anomalies (high error rate, cost spike, failed jobs)
- [ ] 5.4.3 Integration with Slack / Teams / Email

---

## Phase 6: Testing, Docs & Launch (Weeks 11–12)

### 6.1 Testing
- [ ] 6.1.1 Unit tests for parsers, chunkers, retrieval pipeline
- [ ] 6.1.2 Integration tests for end-to-end skill creation flows
- [ ] 6.1.3 Performance tests (ingestion throughput, retrieval latency, GPU utilization)
- [ ] 6.1.4 Security audit (SSO, RBAC, sandboxed code execution, data isolation)

### 6.2 Documentation
- [ ] 6.2.1 User guide — how to create each skill type
- [ ] 6.2.2 Admin guide — managing users, resources, costs
- [ ] 6.2.3 API reference for consuming skills programmatically
- [ ] 6.2.4 Quickstart templates (sample materials → skill in 5 min)

### 6.3 Launch
- [ ] 6.3.1 Onboard pilot team (3–5 users), collect feedback
- [ ] 6.3.2 Bug bash & polish
- [ ] 6.3.3 Company-wide rollout with training session
- [ ] 6.3.4 Set up ongoing support & maintenance rotation

---

## Summary

| Phase | Focus | Timeline |
|---|---|---|
| Phase 0 | Foundation & Setup | Weeks 1–2 |
| Phase 1 | Material Ingestion Pipeline | Weeks 3–5 |
| Phase 2 | Fine-Tuning Skill | Weeks 4–7 |
| Phase 3 | RAG Skill | Weeks 5–8 |
| Phase 4 | AI Agent Skill | Weeks 7–10 |
| Phase 5 | Platform Integration & UX | Weeks 8–11 |
| Phase 6 | Testing, Docs & Launch | Weeks 11–12 |
