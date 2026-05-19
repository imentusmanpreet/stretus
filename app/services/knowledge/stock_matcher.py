"""
LLM-assisted supported stock matching.

Flow:
1. Query ChromaDB over stock-universe rows indexed from CSV/XLSX KB files.
2. Shortlist semantic candidates by company name.
3. Ask the configured LLM to choose the best supported company.
"""
from __future__ import annotations

import json
import re
from typing import Any

from app.core.errors import AppError
from app.kb import kb
from app.kb.compat import AMBIGUOUS_STOCK_VALIDATION_CODE, ambiguous_stock_validation_facts
from app.kb.schemas import Stock
from app.services.ai.llm import LLMService

def _normalise_company_key(company_name: str) -> str:
    return "".join(ch.lower() for ch in str(company_name or "") if ch.isalnum())


def _direct_stock_payload(stock: Stock) -> dict[str, Any]:
    return {
        "company_name": stock.display_name,
        "symbol": stock.symbol.split(".", 1)[0],
        "exchange": stock.exchange,
        "sector": stock.sector,
        "confidence": 1.0,
        "reason": "kb_stock_match",
    }


def _direct_supported_stock_match(query: str) -> dict[str, Any] | None:
    if not str(query or "").strip():
        return None

    resolution = kb.resolve_stock_query(query)
    if resolution.is_ambiguous:
        facts = ambiguous_stock_validation_facts(query, resolution.ambiguous_matches)
        return {
            "ambiguous": True,
            "validation_code": AMBIGUOUS_STOCK_VALIDATION_CODE,
            "validation_facts": facts,
            "matches": facts["stock_options"],
            "confidence": 1.0,
            "reason": "ambiguous_stock_prefix",
        }

    if resolution.stock is None:
        return None
    return _direct_stock_payload(resolution.stock)


def _dedupe_candidates(
    documents: list[str],
    metadatas: list[dict[str, Any]],
    distances: list[float] | None = None,
    max_candidates: int = 8,
) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    distances = distances or []

    for index, metadata in enumerate(metadatas):
        company_name = str(metadata.get("company_name") or "").strip()
        symbol = str(metadata.get("symbol") or "").strip()
        if not company_name:
            continue

        candidate = {
            "candidate_id": f"c{len(by_key) + 1}",
            "company_name": company_name,
            "symbol": symbol,
            "source_index": str(metadata.get("source_index") or "").strip(),
            "source_file": str(metadata.get("source") or metadata.get("source_file") or "").strip(),
            "distance": float(distances[index]) if index < len(distances) else 999.0,
            "document": documents[index] if index < len(documents) else "",
        }

        dedupe_key = _normalise_company_key(company_name) or symbol or company_name.lower()
        current = by_key.get(dedupe_key)
        if current is None:
            by_key[dedupe_key] = candidate
            continue

        current_rank = (0 if current["symbol"] else 1, current["distance"])
        candidate_rank = (0 if candidate["symbol"] else 1, candidate["distance"])
        if candidate_rank < current_rank:
            candidate["candidate_id"] = current["candidate_id"]
            by_key[dedupe_key] = candidate

    ranked = sorted(
        by_key.values(),
        key=lambda item: (0 if item["symbol"] else 1, item["distance"], item["company_name"].lower()),
    )

    for idx, candidate in enumerate(ranked[:max_candidates], start=1):
        candidate["candidate_id"] = f"c{idx}"
    return ranked[:max_candidates]


def search_supported_stock_candidates(query: str, n_results: int = 12) -> list[dict[str, Any]]:
    cleaned_query = " ".join(str(query or "").split())
    if not cleaned_query:
        return []

    try:
        from app.services.knowledge.embedder import (
            COLLECTION_NAME,
            STOCK_UNIVERSE_RECORD_TYPE,
            get_chroma_client,
        )
        from app.services.knowledge.embeddings import embed_texts

        client = get_chroma_client()
        collection = client.get_collection(COLLECTION_NAME)
        if collection.count() == 0:
            return []

        query_embedding = embed_texts([cleaned_query])[0]
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(max(n_results, 1), collection.count()),
            where={"record_type": STOCK_UNIVERSE_RECORD_TYPE},
            include=["documents", "metadatas", "distances"],
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        return _dedupe_candidates(documents, metadatas, distances)
    except Exception as exc:
        print(f"[Stock Match] candidate search failed: {exc}")
        return []


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = str(text or "").strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.IGNORECASE | re.DOTALL).strip()

    try:
        return json.loads(stripped)
    except Exception:
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None


async def resolve_supported_stock(query: str) -> dict[str, Any] | None:
    direct_match = _direct_supported_stock_match(query)
    if direct_match:
        return direct_match

    candidates = search_supported_stock_candidates(query)
    if not candidates:
        return None

    llm = LLMService()
    candidate_lines = []
    for candidate in candidates:
        candidate_lines.append(
            json.dumps(
                {
                    "candidate_id": candidate["candidate_id"],
                    "company_name": candidate["company_name"],
                    "symbol": candidate["symbol"],
                    "source_index": candidate["source_index"],
                },
                ensure_ascii=True,
            )
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You match a user's requested Indian stock/company name to one supported company candidate.\n"
                "Choose only when the candidate clearly refers to the same company or a very common short form.\n"
                "Prefer candidates with a non-empty symbol.\n"
                "If the requested company is not clearly present, return matched=false.\n"
                "Respond with JSON only using this shape:\n"
                '{"matched": true|false, "candidate_id": "c1" | null, "confidence": 0.0, "reason": "short"}'
            ),
        },
        {
            "role": "user",
            "content": (
                f"Requested company: {query}\n"
                "Supported candidates:\n"
                + "\n".join(candidate_lines)
            ),
        },
    ]

    try:
        response = await llm.chat(messages)
        payload = _extract_json_object(response)
        if not payload or not payload.get("matched"):
            return None

        candidate_id = str(payload.get("candidate_id") or "").strip()
        confidence = float(payload.get("confidence") or 0.0)
        if not candidate_id or confidence < 0.5:
            return None

        selected = next((item for item in candidates if item["candidate_id"] == candidate_id), None)
        if not selected or not selected.get("symbol"):
            return None
        selected["confidence"] = confidence
        selected["reason"] = str(payload.get("reason") or "").strip()
        return selected
    except AppError:
        raise
    except Exception as exc:
        print(f"[Stock Match] LLM resolution failed: {exc}")
        return None
