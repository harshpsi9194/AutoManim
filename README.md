# RAG-Based Manim Code Generator (`manim-rag`)

A modular Retrieval-Augmented Generation (RAG) system for generating, validating, and automatically repairing **Manim Community Edition (ManimCE)** animation scripts.

## Architecture

1. **ManimCE Knowledge Base & AST Chunker (`manim_rag.chunker`)**
   - `ManimASTChunker` parses ManimCE Python source trees with `ast` into structured `CodeChunk` records (classes, methods, functions, async functions, decorators, base classes, full signatures with annotations/defaults, docstrings, source slices, and line ranges).
   - Handles nested classes, overloaded definitions, private-symbol filtering, and broken files (`on_syntax_error="skip" | "raise"`).
   - `ManimKnowledgeBase` provides symbol lookup, metadata filtering, and JSON persistence.

2. **Hybrid Search Retriever (`manim_rag.retrieval`)**
   - Combines **vector similarity search** (`ChromaVectorStore` for ChromaDB persistent/ephemeral collections or `InMemoryVectorStore` for zero-download local testing) with **BM25 keyword search** (`BM25Index` with code-aware camelCase/snake_case tokenization).
   - Supports **Reciprocal Rank Fusion (RRF)** and **min-max weighted score fusion**, plus a LangChain `BaseRetriever` adapter (`HybridRetriever.as_langchain_retriever()`).

3. **LangChain Generation Pipeline with Llama 3 70B (`manim_rag.generation`)**
   - `ManimGenerationPipeline` composes LangChain LCEL chains (`ChatPromptTemplate | llm | StrOutputParser | extract_python_code`) for initial generation and error-driven regeneration.
   - `LLMConfig` supports Llama 3 70B across **Groq**, **Together AI**, **Ollama**, **HuggingFace**, and any **OpenAI-compatible** endpoint.

4. **Sandboxed Docker Execution & Failure Classification / Retry Loop (`manim_rag.execution`, `manim_rag.orchestrator`)**
   - `DockerSandboxExecutor` executes generated scripts inside isolated Docker containers (`--network=none`, `--read-only` rootfs with bounded `/tmp`, `--cap-drop=ALL`, `--security-opt=no-new-privileges:true`, CPU/memory/PID limits, and timeout-triggered `docker rm -f` cleanup).
   - `FailureClassifier` parses tracebacks into:
     - `api_error` (`AttributeError`, `ImportError`, `ModuleNotFoundError`, `NameError`, `UnboundLocalError`)
     - `argument_error` (`TypeError`, `ValueError`, signature mismatches)
     - `syntax_error` (`SyntaxError`, `IndentationError`, `TabError`)
     - `runtime_error` / `timeout_error` / `unknown_error`
   - `ManimRAGOrchestrator` extracts offending symbols from the failure, performs targeted hybrid retrieval for corrected ManimCE APIs, and regenerates code up to `max_retries`.

## Installation

```bash
cd /home/ice/projects/manim-rag
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
# Optional ChromaDB vector backend:
# uv pip install -e ".[chromadb,dev]"
```

## CLI Usage

```bash
# 1. Build a knowledge base from ManimCE source (or the bundled example API)
manim-rag index examples/sample_manim_source -o manim_kb.json

# 2. Query the hybrid retriever (RRF or weighted fusion)
manim-rag search "Create circle animation" --kb manim_kb.json --fusion rrf --top-k 5

# 3. Dry-run retrieval for a prompt
manim-rag generate "Plot sin(x) on axes and create a blue circle" --kb manim_kb.json --dry-run

# 4. Run full generation + Docker sandbox execution (requires LLM API key + Docker image)
export GROQ_API_KEY=...
docker build -t manim-rag-sandbox:latest -f docker/Dockerfile .
manim-rag generate "Draw a red circle and transform it into a square" --kb manim_kb.json --provider groq
```

## Running Tests

```bash
source .venv/bin/activate
pytest
```
