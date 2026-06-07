"""
kec_rag_retriever_local.py
────────────────────────────────────────────────────────────────────────────
KEC 비탈면 매뉴얼 RAG — 로컬 BM25 검색기 (Milvus 없이 동작)

사용법:
    from kec_rag_retriever_local import LocalRetriever

    retriever = LocalRetriever()           # manual_chunks_v2.jsonl 자동 로드
    chunks = retriever.retrieve("그라운드 앵커 설계 하중 기준", top_k=6)
    for c in chunks:
        print(c.chunk_type, c.manual_id, f"p.{c.page_start}", c.content[:80])

특징:
    - 순수 Python (외부 패키지 불필요)
    - BM25(Okapi) 알고리즘 + 한국어 2-gram 토크나이저
    - chunk_type(text/table/figure) 가중치 보정
    - method 필터링 지원 (예: "그라운드 앵커"만 검색)
    - Milvus 미연결 시 kec_rag_agent.py 자동 폴백 대상
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import gzip
import json
import math
import re
import logging
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("kec_local_retriever")

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
# .gz 압축 파일 우선, 없으면 원본 jsonl 사용
_GZ_FILE   = _HERE / "manual_chunks_v2.jsonl.gz"
_JSONL_FILE = _HERE / "manual_chunks_v2.jsonl"
CHUNKS_FILE = _GZ_FILE if _GZ_FILE.exists() else _JSONL_FILE

# BM25 하이퍼파라미터
BM25_K1 = 1.5
BM25_B  = 0.75

# chunk_type별 점수 가중치
TYPE_WEIGHT = {"text": 1.0, "table": 1.2, "figure": 0.9}


# =============================================================================
# Chunk 데이터클래스 (kec_rag_indexer.Chunk 호환)
# =============================================================================

@dataclass
class Chunk:
    """로컬 검색 결과 단위 (kec_rag_indexer.Chunk 인터페이스 호환)."""
    chunk_id:   str = ""
    manual_id:  str = ""          # doc_name
    source_path: str = ""         # source 파일명
    page_start: int = 0
    page_end:   int = 0
    chapter:    str = ""
    section:    str = ""
    figure_id:  str = ""          # 그림/표 캡션
    chunk_type: str = "text"      # text | table | figure
    content:    str = ""          # 검색에 사용된 전체 텍스트
    keywords:   list[str] = field(default_factory=list)
    # 임베딩 (로컬 모드에서는 빈 벡터)
    dense_vector:  list[float] = field(default_factory=list)
    sparse_vector: dict        = field(default_factory=dict)
    # 추가 메타
    method:     str = ""
    bm25_score: float = 0.0


# =============================================================================
# 한국어 토크나이저 (2-gram + 어절 복합)
# =============================================================================

def tokenize(text: str) -> list[str]:
    """
    한국어 텍스트를 토큰 리스트로 변환.
    전략: 어절(공백) + 2-gram 문자 조합 → 높은 재현율 달성
    """
    # 특수문자·괄호 제거, 소문자
    text = re.sub(r"[^\w\s가-힣]", " ", text.lower())
    # 어절 분리
    words = [w for w in text.split() if len(w) >= 2]
    # 2-gram 문자
    bigrams = []
    for w in words:
        if len(w) >= 4:
            bigrams.extend(w[i:i+2] for i in range(len(w)-1))
    return words + bigrams


# =============================================================================
# BM25 엔진
# =============================================================================

class BM25:
    """순수 Python Okapi BM25 구현."""

    def __init__(self, documents: list[str]):
        self.n_docs  = len(documents)
        self.avgdl   = 0.0
        self.tf      : list[dict[str, int]] = []
        self.df      : dict[str, int]       = defaultdict(int)
        self.idf     : dict[str, float]     = {}

        # TF 계산
        for doc in documents:
            tokens = tokenize(doc)
            tf     = Counter(tokens)
            self.tf.append(tf)
            for token in tf:
                self.df[token] += 1
            self.avgdl += len(tokens)

        self.avgdl /= max(self.n_docs, 1)

        # IDF 계산
        for token, df in self.df.items():
            self.idf[token] = math.log(
                (self.n_docs - df + 0.5) / (df + 0.5) + 1
            )

    def score(self, doc_idx: int, query_tokens: list[str]) -> float:
        tf_doc = self.tf[doc_idx]
        dl     = sum(tf_doc.values())
        score  = 0.0
        for token in query_tokens:
            if token not in tf_doc:
                continue
            tf_val = tf_doc[token]
            idf    = self.idf.get(token, 0.0)
            tf_norm = tf_val * (BM25_K1 + 1) / (
                tf_val + BM25_K1 * (1 - BM25_B + BM25_B * dl / max(self.avgdl, 1))
            )
            score += idf * tf_norm
        return score

    def search(
        self,
        query: str,
        top_k: int = 10,
        idx_subset: Optional[list[int]] = None,
    ) -> list[tuple[int, float]]:
        """
        query에 대한 BM25 점수를 계산하여 상위 top_k 문서 인덱스와 점수를 반환.
        idx_subset: 검색 대상 인덱스를 제한 (공법 필터 등에 활용)
        """
        q_tokens = tokenize(query)
        candidates = idx_subset if idx_subset is not None else range(self.n_docs)
        scores = [
            (idx, self.score(idx, q_tokens))
            for idx in candidates
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# =============================================================================
# 로컬 검색기
# =============================================================================

class LocalRetriever:
    """
    manual_chunks_v2.jsonl을 로드하고 BM25 기반 검색을 제공한다.
    kec_rag_agent.py에서 Milvus 폴백으로 사용한다.
    """

    def __init__(self, chunks_file: Path = CHUNKS_FILE):
        self.chunks_file = chunks_file
        self._raw: list[dict]  = []    # 원본 JSON 레코드
        self._chunks: list[Chunk] = [] # Chunk 객체 목록
        self._bm25: Optional[BM25] = None

        self._load()
        self._build_index()

    # ── 내부 메서드 ────────────────────────────────────────────────────────────

    def _load(self):
        """JSONL 파일을 읽어 raw 레코드와 Chunk 객체 생성."""
        if not self.chunks_file.exists():
            raise FileNotFoundError(
                f"청크 파일 없음: {self.chunks_file}\n"
                "extract_chunks_v2.py를 먼저 실행하세요."
            )
        # .gz 자동 감지
        opener = gzip.open if str(self.chunks_file).endswith(".gz") else open
        with opener(str(self.chunks_file), "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                self._raw.append(rec)
                self._chunks.append(self._to_chunk(rec))

        logger.info(
            "로컬 검색기 로드: %d 청크 (%s)",
            len(self._chunks),
            self.chunks_file.name,
        )

    def _to_chunk(self, rec: dict) -> Chunk:
        """JSON 레코드 → Chunk 데이터클래스 변환."""
        chunk_type = rec.get("type", "text")
        figure_id  = rec.get("figure_caption", "") or rec.get("table_title", "")
        # 장·절 추출 (텍스트 첫 줄에서 '제N장', 'N.N' 패턴 탐지)
        content    = rec.get("text", "")
        chapter, section = _extract_chapter_section(content)

        return Chunk(
            chunk_id    = rec.get("chunk_id", ""),
            manual_id   = rec.get("doc_name", ""),
            source_path = rec.get("source", ""),
            page_start  = rec.get("page", 0),
            page_end    = rec.get("page", 0),
            chapter     = chapter,
            section     = section,
            figure_id   = figure_id,
            chunk_type  = chunk_type,
            content     = content,
            method      = rec.get("method", ""),
        )

    def _build_index(self):
        """BM25 인덱스 빌드."""
        docs = [c.content for c in self._chunks]
        self._bm25 = BM25(docs)
        logger.info("BM25 인덱스 빌드 완료 (문서 수=%d)", len(docs))

    # ── 공개 API ───────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 6,
        method_filter: Optional[str] = None,
        type_filter: Optional[list[str]] = None,
        boost_tables: bool = True,
    ) -> list[Chunk]:
        """
        BM25로 상위 chunk를 검색하여 반환한다.

        Args:
            query:         검색 쿼리
            top_k:         반환할 청크 수
            method_filter: 공법명 필터 (예: "그라운드 앵커")
                           None이면 전체 검색
            type_filter:   청크 타입 필터 (예: ["text", "table"])
                           None이면 전체 타입
            boost_tables:  표 청크의 점수를 가중치 보정할지 여부
        """
        # 필터링된 인덱스 목록
        idx_subset = None
        if method_filter or type_filter:
            idx_subset = []
            for i, c in enumerate(self._chunks):
                method_ok = (not method_filter) or (method_filter in c.method)
                type_ok   = (not type_filter) or (c.chunk_type in type_filter)
                if method_ok and type_ok:
                    idx_subset.append(i)

        # BM25 검색 (여유분 확보 후 재정렬)
        raw_results = self._bm25.search(query, top_k=top_k * 3, idx_subset=idx_subset)

        # 타입 가중치 보정
        weighted = []
        for idx, score in raw_results:
            chunk  = self._chunks[idx]
            weight = TYPE_WEIGHT.get(chunk.chunk_type, 1.0) if boost_tables else 1.0
            weighted.append((chunk, score * weight))

        # 정렬 후 상위 top_k
        weighted.sort(key=lambda x: x[1], reverse=True)
        results = []
        for chunk, score in weighted[:top_k]:
            chunk.bm25_score = round(score, 4)
            results.append(chunk)

        return results

    def get_all_methods(self) -> list[str]:
        """인덱스에 포함된 모든 공법명 반환."""
        return sorted(set(c.method for c in self._chunks if c.method))

    def stats(self) -> dict:
        """청크 통계."""
        by_type   = Counter(c.chunk_type for c in self._chunks)
        by_method = Counter(c.method for c in self._chunks)
        return {
            "total":     len(self._chunks),
            "by_type":   dict(by_type),
            "by_method": dict(by_method),
        }


# =============================================================================
# 유틸 함수
# =============================================================================

def _extract_chapter_section(text: str) -> tuple[str, str]:
    """
    청크 텍스트 첫 부분에서 장·절 정보를 추출한다.
    예: "제3장 그라운드 앵커의 설계" → chapter="제3장 그라운드 앵커의 설계"
    """
    chapter, section = "", ""
    lines = [l.strip() for l in text.split("\n") if l.strip()][:5]

    for line in lines:
        # 장 패턴: 제N장 또는 Chapter N
        if re.match(r"^제\s*\d+\s*장", line):
            chapter = line[:30]
            break
        # 절 패턴: N.N 또는 N.N.N
        m = re.match(r"^(\d+\.\d+(?:\.\d+)?)\s+(.{2,30})", line)
        if m:
            if not section:
                section = m.group(0)[:30]

    return chapter, section


# =============================================================================
# kec_rag_agent.py 폴백 호환 함수 (동일 시그니처)
# =============================================================================

_default_retriever: Optional[LocalRetriever] = None

def retrieve_relevant_chunks(
    query: str,
    embedder=None,        # Milvus 버전 인터페이스 호환 (무시)
    top_k: int = 6,
    partition: Optional[str] = None,
) -> list[Chunk]:
    """
    kec_rag_indexer.retrieve_relevant_chunks()와 동일한 시그니처.
    kec_rag_agent.py에서 Milvus 대신 이 함수를 호출할 수 있다.
    """
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = LocalRetriever()

    # partition → method_filter 변환
    method_filter = None
    if partition:
        # 파티션명과 공법명이 일치하지 않으므로 키워드 매핑
        PARTITION_TO_METHODS = {
            "사면설계": ["그라운드 앵커", "억지말뚝", "숏크리트", "네일공", "록볼트"],
            "비탈면유지관리": ["낙석방지망", "낙석방지울타리"],
        }
        # 파티션에 매핑된 공법 중 쿼리에 언급된 것 우선
        mapped = PARTITION_TO_METHODS.get(partition, [])
        for m in mapped:
            if m in query:
                method_filter = m
                break

    return _default_retriever.retrieve(query, top_k=top_k, method_filter=method_filter)


# =============================================================================
# CLI 테스트
# =============================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    retriever = LocalRetriever()
    print("\n=== 청크 통계 ===")
    st = retriever.stats()
    print(f"  총 청크: {st['total']}")
    print(f"  유형별: {st['by_type']}")
    print(f"  공법별: {st['by_method']}\n")

    # 테스트 쿼리 목록
    queries = [
        "그라운드 앵커 자유장 길이 설계 기준",
        "비탈면 점검 주기 및 방법",
        "숏크리트 두께 기준",
        "억지말뚝 시공 허용오차",
        "낙석방지망 설치 방법",
    ]
    if len(sys.argv) > 1:
        queries = [" ".join(sys.argv[1:])]

    for q in queries:
        print(f"\n{'='*60}")
        print(f"질의: {q}")
        print("-"*60)
        results = retriever.retrieve(q, top_k=4)
        for i, c in enumerate(results, 1):
            print(
                f"[{i}] {c.chunk_type:6s} | {c.method:<12} | p.{c.page_start:3d} "
                f"| score={c.bm25_score:.3f}"
            )
            print(f"     {c.content[:120].replace(chr(10),' ')}...")
