"""
Agentic RAG loop — query decomposition + iterative retrieval + self-check.

Flow per run():
  1. Decompose query → sub-questions (LLM)
  2. Hybrid-search each sub-question, merge & deduplicate results
  3. Generate answer from merged context (LLM)
  4. Self-check answer for completeness (LLM)
  5. If gaps remain and iterations are left: use gaps as new queries → goto 2
"""

import logging
import time

from app.config import settings
from app.core.generation.answer_generator import generate_answer
from app.core.generation.query_decomposer import decompose_query
from app.core.generation.self_check import self_check
from app.prompts.devops_prompts import build_devops_prompt

logger = logging.getLogger(__name__)


class AgenticRAGLoop:
    def __init__(self, rag_pipeline, max_iterations: int = 2):
        self.rag = rag_pipeline
        self.max_iterations = max(1, min(max_iterations, 3))

    def run(self, query: str, top_k: int | None = None, use_hybrid: bool = True) -> dict:
        top_k = top_k or settings.TOP_K_RESULTS
        t0 = time.perf_counter()

        # ── Step 1: Decompose ──────────────────────────────────────────────
        sub_questions = decompose_query(query, self.rag.llm_client)
        logger.info(f"[Agentic] sub-questions: {sub_questions}")

        # Accumulate chunks across all iterations; key deduplicates by content
        pool: dict[str, dict] = {}
        iterations_log = []
        answer = ""
        check: dict = {"complete": False, "confidence": 0.0, "gaps": []}
        current_queries = sub_questions

        for iteration in range(self.max_iterations):
            logger.info(
                f"[Agentic] iteration {iteration + 1}/{self.max_iterations} "
                f"queries={current_queries}"
            )

            # ── Step 2: Retrieve ───────────────────────────────────────────
            per_q_k = max(3, top_k // max(len(current_queries), 1) + 2)
            for sq in current_queries:
                results = self.rag.hybrid_search(sq, top_k=per_q_k)
                for r in results:
                    # Deduplicate by first 80 chars of content (handles near-dupes)
                    key = r.get("content", "")[:80]
                    if key not in pool or r["score"] > pool[key]["score"]:
                        pool[key] = r

            # Rank merged pool, keep top_k
            ranked = sorted(pool.values(), key=lambda x: x["score"], reverse=True)[:top_k]
            context_chunks = [r["content"] for r in ranked]

            if not context_chunks:
                logger.warning("[Agentic] no context retrieved — stopping")
                break

            # ── Step 3: Generate ───────────────────────────────────────────
            use_devops = getattr(self.rag, "use_devops_prompts", False)
            answer = generate_answer(
                query=query,
                context_chunks=context_chunks,
                llm_client=self.rag.llm_client,
                use_devops_prompts=use_devops,
                prompt_builder=build_devops_prompt if use_devops else None,
            )

            # ── Step 4: Self-check ─────────────────────────────────────────
            check = self_check(query, answer, self.rag.llm_client)
            iterations_log.append(
                {
                    "iteration": iteration + 1,
                    "queries": current_queries,
                    "chunks_used": len(ranked),
                    "complete": check["complete"],
                    "confidence": check["confidence"],
                    "gaps": check["gaps"],
                }
            )
            logger.info(
                f"[Agentic] iteration {iteration + 1} done — "
                f"complete={check['complete']} confidence={check['confidence']:.2f} "
                f"gaps={check['gaps']}"
            )

            # ── Step 5: Loop or stop ───────────────────────────────────────
            if check["complete"] or not check["gaps"]:
                break

            if iteration + 1 < self.max_iterations:
                # Next round: search specifically for identified gaps
                current_queries = check["gaps"][:3]

        return {
            "answer": answer,
            "sources": ranked if context_chunks else [],
            "sub_questions": sub_questions,
            "iterations": iterations_log,
            "final_complete": check["complete"],
            "final_confidence": check["confidence"],
            "generation_time": time.perf_counter() - t0,
        }
