"""Command-line interface for manim-rag indexing, hybrid search, and code generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from manim_rag.chunker import ManimASTChunker, ManimKnowledgeBase
from manim_rag.config import DockerConfig, LLMConfig, LLMProvider, PipelineConfig, RetrievalConfig
from manim_rag.execution import DockerSandboxExecutor, FailureClassifier
from manim_rag.generation import ManimGenerationPipeline
from manim_rag.orchestrator import ManimRAGOrchestrator
from manim_rag.retrieval import ChromaVectorStore, HybridRetriever, InMemoryVectorStore


def _build_retriever(
    kb: ManimKnowledgeBase,
    *,
    backend: str,
    chroma_dir: str | None,
    config: RetrievalConfig,
) -> HybridRetriever:
    if backend == "chroma":
        vector_store = ChromaVectorStore(
            collection_name=config.collection_name,
            persist_directory=chroma_dir,
        )
    else:
        vector_store = InMemoryVectorStore()
    return HybridRetriever(kb, vector_store=vector_store, config=config)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="manim-rag",
        description="RAG-Based ManimCE Code Generator using AST chunking, Hybrid Search, Llama 3 70B, and Docker sandbox retry.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # index
    idx = sub.add_parser("index", help="AST-chunk a ManimCE directory and save the knowledge base.")
    idx.add_argument("source_dir", help="Path to ManimCE source directory.")
    idx.add_argument(
        "-o",
        "--output",
        default="manim_kb.json",
        help="Path to output JSON knowledge base (default: manim_kb.json).",
    )
    idx.add_argument(
        "--include-private",
        action="store_true",
        help="Include private (_prefixed) functions and classes.",
    )

    # search
    srch = sub.add_parser("search", help="Run hybrid BM25 + vector search over a knowledge base.")
    srch.add_argument("query", help="Search query.")
    srch.add_argument("--kb", required=True, help="Path to saved JSON knowledge base.")
    srch.add_argument("--top-k", type=int, default=5, help="Number of chunks to return.")
    srch.add_argument(
        "--fusion",
        choices=["rrf", "weighted"],
        default="rrf",
        help="Hybrid fusion strategy.",
    )
    srch.add_argument(
        "--backend",
        choices=["memory", "chroma"],
        default="memory",
        help="Vector store backend.",
    )
    srch.add_argument("--chroma-dir", default=None, help="Optional ChromaDB persistence path.")

    # generate
    gen = sub.add_parser("generate", help="Generate and execute ManimCE code with retry loop.")
    gen.add_argument("prompt", help="Animation prompt.")
    gen.add_argument("--kb", required=True, help="Path to saved JSON knowledge base.")
    gen.add_argument(
        "--provider",
        choices=[p.value for p in LLMProvider],
        default=LLMProvider.GROQ.value,
        help="Llama 3 70B provider.",
    )
    gen.add_argument("--model", default=None, help="Override Llama 3 70B model ID.")
    gen.add_argument("--base-url", default=None, help="Override OpenAI-compatible base URL.")
    gen.add_argument("--max-retries", type=int, default=3, help="Max repair retries.")
    gen.add_argument("--docker-image", default="manim-rag-sandbox:latest", help="Sandbox Docker image.")
    gen.add_argument("--workdir", default=None, help="Optional output workspace directory.")
    gen.add_argument(
        "--dry-run",
        action="store_true",
        help="Only perform hybrid retrieval and print formatted context without calling LLM/Docker.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "index":
        chunker = ManimASTChunker(include_private=args.include_private)
        kb, result = ManimKnowledgeBase.from_directory(args.source_dir, chunker=chunker)
        out_path = kb.save(args.output)
        print(
            json.dumps(
                {
                    "output": str(out_path),
                    "files_processed": result.files_processed,
                    "chunks_indexed": len(kb),
                    "syntax_errors": len(result.errors),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "search":
        kb = ManimKnowledgeBase.load(args.kb)
        cfg = RetrievalConfig(top_k=args.top_k, fusion_method=args.fusion)
        retriever = _build_retriever(
            kb,
            backend=args.backend,
            chroma_dir=args.chroma_dir,
            config=cfg,
        )
        hits = retriever.retrieve(args.query, top_k=args.top_k)
        payload = [
            {
                "chunk_id": h.chunk.chunk_id,
                "qualified_name": h.chunk.qualified_name,
                "chunk_type": h.chunk.chunk_type.value,
                "score": round(h.score, 6),
                "signature": h.chunk.signature,
            }
            for h in hits
        ]
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "generate":
        kb = ManimKnowledgeBase.load(args.kb)
        ret_cfg = RetrievalConfig()
        retriever = _build_retriever(kb, backend="memory", chroma_dir=None, config=ret_cfg)

        if args.dry_run:
            hits = retriever.retrieve(args.prompt, top_k=ret_cfg.top_k)
            print(
                json.dumps(
                    {
                        "mode": "dry-run",
                        "prompt": args.prompt,
                        "retrieved": [
                            {
                                "chunk_id": h.chunk.chunk_id,
                                "qualified_name": h.chunk.qualified_name,
                                "score": round(h.score, 6),
                            }
                            for h in hits
                        ],
                    },
                    indent=2,
                )
            )
            return 0

        pipe_cfg = PipelineConfig(
            llm=LLMConfig(
                provider=LLMProvider(args.provider),
                model_name=args.model,
                base_url=args.base_url,
            ),
            retrieval=ret_cfg,
            docker=DockerConfig(image=args.docker_image),
            max_retries=args.max_retries,
        )
        orchestrator = ManimRAGOrchestrator(
            retriever=retriever,
            generator=ManimGenerationPipeline(config=pipe_cfg.llm),
            executor=DockerSandboxExecutor(config=pipe_cfg.docker),
            classifier=FailureClassifier(),
            config=pipe_cfg,
        )
        result = orchestrator.run(args.prompt, workdir=Path(args.workdir) if args.workdir else None)
        print(result.model_dump_json(indent=2))
        return 0 if result.success else 1

    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
