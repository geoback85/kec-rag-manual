"""
kec_rag_agent_v2.py
────────────────────────────────────────────────────────────────────────────
KEC 비탈면 매뉴얼 RAG 에이전트 v2

변경사항 (v1 → v2):
  1. 검색 백엔드 자동 전환
       Milvus 연결 가능 → kec_rag_indexer.retrieve_relevant_chunks()
       Milvus 연결 불가 → kec_rag_retriever_local.LocalRetriever (BM25)
  2. chunk_type(text/table/figure) 인식 컨텍스트 빌더
       표 청크:  "[표 N] 제목 | 내용" 형식으로 LLM에 전달
       그림 청크: "[그림 N] 캡션 + 문맥" 형식으로 전달
  3. method 자동 필터링 강화
       쿼리에서 공법명 감지 → 해당 공법 청크 우선 검색
  4. kec_rag_mobile_app.py, kec_rag_eval.py와 동일 인터페이스 유지

사용법:
    from kec_rag_agent_v2 import KecRagAgent

    agent = KecRagAgent()                        # 자동으로 백엔드 감지
    resp  = agent.ask("그라운드 앵커 자유장 설계 기준은?")
    print(resp.format_for_display())

환경변수 (.env 또는 시스템):
    ANTHROPIC_API_KEY=sk-ant-...
    LLM_PROVIDER=claude   (claude | openai | local)
    LLM_MODEL=claude-sonnet-4-6
    MILVUS_HOST=localhost
    MILVUS_PORT=19530
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import re
import json
import logging
import textwrap
import time
from dataclasses import dataclass, field
from typing import Optional, Literal
from pathlib import Path

# ── 환경변수 ──────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("kec_rag_agent_v2")


# =============================================================================
# 검색 백엔드 자동 감지
# =============================================================================

def _try_milvus_import():
    """Milvus 연결 가능 여부 확인. 가능하면 (embedder, retrieve_fn) 반환."""
    try:
        from pymilvus import connections
        host = os.getenv("MILVUS_HOST", "localhost")
        port = int(os.getenv("MILVUS_PORT", "19530"))
        connections.connect(host=host, port=port, timeout=3)

        from kec_rag_indexer import BGE_M3_Embedder, retrieve_relevant_chunks
        embedder = BGE_M3_Embedder()
        logger.info("검색 백엔드: Milvus (%s:%d)", host, port)
        return embedder, retrieve_relevant_chunks
    except Exception as e:
        logger.info("Milvus 미연결 (%s) → 로컬 BM25 폴백", e)
        return None, None


def _load_local_backend():
    """로컬 BM25 검색기 로드."""
    from kec_rag_retriever_local import LocalRetriever, retrieve_relevant_chunks
    retriever = LocalRetriever()
    return retriever, retrieve_relevant_chunks


# =============================================================================
# 에이전트 설정
# =============================================================================

class AgentConfig:
    LLM_PROVIDER:    str   = os.getenv("LLM_PROVIDER", "claude")
    LLM_MODEL:       str   = os.getenv("LLM_MODEL",    "claude-sonnet-4-6")
    MAX_TOKENS:      int   = 4096
    TEMPERATURE:     float = 0.1
    CONTEXT_TOP_K:   int   = 6
    STREAMING:       bool  = False
    LOCAL_LLM_BASE_URL: str = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1")
    LOCAL_LLM_MODEL:    str = os.getenv("LOCAL_LLM_MODEL", "llama3")

acfg = AgentConfig()


# =============================================================================
# 데이터클래스
# =============================================================================

@dataclass
class Citation:
    manual_id:  str = ""
    chapter:    str = ""
    section:    str = ""
    page_start: int = 0
    page_end:   int = 0
    figure_id:  str = ""
    chunk_type: str = "text"

    def to_label(self) -> str:
        parts = [self.manual_id]
        if self.chapter: parts.append(self.chapter)
        if self.section: parts.append(self.section)
        parts.append(f"p.{self.page_start}" if self.page_start == self.page_end
                     else f"p.{self.page_start}~{self.page_end}")
        if self.figure_id: parts.append(self.figure_id)
        return " > ".join(parts)


@dataclass
class AgentResponse:
    query:            str  = ""
    answer:           str  = ""
    citations:        list[Citation] = field(default_factory=list)
    has_evidence:     bool = True
    confidence:       Literal["높음","보통","낮음","없음"] = "보통"
    template_used:    str  = "general_qa"
    retrieved_chunks: list = field(default_factory=list)
    elapsed_ms:       int  = 0
    backend:          str  = "local"   # "milvus" | "local"

    def format_for_display(self) -> str:
        lines = [
            "=" * 60,
            f"【질의】 {self.query}",
            f"【검색 백엔드】 {self.backend}",
            "=" * 60,
            "",
            "【답변】",
            self.answer,
            "",
        ]
        if self.citations:
            lines.append("【인용 출처】")
            for i, c in enumerate(self.citations, 1):
                lines.append(f"  [{i}] {c.to_label()}")
            lines.append("")
        lines.append(f"【신뢰도】 {self.confidence}   "
                     f"{'근거 있음' if self.has_evidence else '⚠ 근거 없음/불충분'}")
        lines.append(f"【응답 시간】 {self.elapsed_ms}ms")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "query":          self.query,
            "answer":         self.answer,
            "citations":      [vars(c) for c in self.citations],
            "has_evidence":   self.has_evidence,
            "confidence":     self.confidence,
            "template_used":  self.template_used,
            "elapsed_ms":     self.elapsed_ms,
            "backend":        self.backend,
        }


# =============================================================================
# 도메인 태그 & 공법명 감지
# =============================================================================

DOMAIN_TAG_MAP = {
    "[비탈면]":   "비탈면유지관리",
    "[사면]":     "비탈면유지관리",
    "[옹벽]":     "사면설계",
    "[앵커]":     "사면설계",
    "[억지말뚝]": "사면설계",
    "[숏크리트]": "사면설계",
    "[낙석]":     "비탈면유지관리",
    "[배수]":     "배수설계",
    "[계측]":     "비탈면유지관리",
    "[설계기준]": "사면설계",
}

AUTO_TAG_RULES = [
    (r"비탈면|절토사면|성토사면|사면붕괴",  "[비탈면]"),
    (r"배수|침투|지하수",                 "[배수]"),
    (r"옹벽|보강토|중력식",              "[옹벽]"),
    (r"앵커|어스앵커|그라운드앵커",       "[앵커]"),
    (r"억지말뚝|강관말뚝",               "[억지말뚝]"),
    (r"숏크리트|shotcrete|뿜어붙이기",   "[숏크리트]"),
    (r"낙석|낙반",                       "[낙석]"),
    (r"계측|경사계|침하계",              "[계측]"),
]

# 공법명 감지 패턴 (쿼리에서 method_filter 결정용)
METHOD_PATTERNS = [
    (r"그라운드\s*앵커|어스앵커",        "그라운드 앵커"),
    (r"억지말뚝|강관말뚝",               "억지말뚝"),
    (r"숏크리트|뿜어붙이기",             "숏크리트"),
    (r"네일공|쏘일네일",                 "네일공"),
    (r"록볼트",                         "록볼트"),
    (r"낙석방지망",                      "낙석방지망"),
    (r"낙석방지울타리",                  "낙석방지울타리"),
    (r"돌망태",                         "돌망태 옹벽"),
    (r"콘크리트\s*옹벽",                 "콘크리트 옹벽"),
]


def preprocess_query(query: str) -> dict:
    tags = []
    for tag in re.findall(r"\[[가-힣a-zA-Z]+\]", query):
        if tag in DOMAIN_TAG_MAP:
            tags.append(tag)
    for pattern, tag in AUTO_TAG_RULES:
        if re.search(pattern, query) and tag not in tags:
            tags.append(tag)

    cleaned    = re.sub(r"\[[가-힣a-zA-Z]+\]", "", query).strip() or query
    partitions = list(set(DOMAIN_TAG_MAP.get(t, "") for t in tags) - {""})
    partition  = partitions[0] if len(partitions) == 1 else None

    # 공법명 감지
    method_filter = None
    for pattern, method in METHOD_PATTERNS:
        if re.search(pattern, query):
            method_filter = method
            break

    return {
        "cleaned_query": cleaned,
        "tags":          tags,
        "partition":     partition,
        "method_filter": method_filter,
    }


# =============================================================================
# LLM 클라이언트
# =============================================================================

class LLMClient:
    def __init__(self):
        self.provider = acfg.LLM_PROVIDER
        self.model    = acfg.LLM_MODEL
        self._client  = self._init_client()

    def _init_client(self):
        if self.provider == "claude":
            import anthropic
            return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        elif self.provider in ("openai", "local"):
            from openai import OpenAI
            kwargs = {"api_key": os.getenv("OPENAI_API_KEY", "ollama")}
            if self.provider == "local":
                kwargs["base_url"] = acfg.LOCAL_LLM_BASE_URL
            return OpenAI(**kwargs)
        raise ValueError(f"지원하지 않는 LLM_PROVIDER: {self.provider}")

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        if self.provider == "claude":
            msg = self._client.messages.create(
                model=self.model, max_tokens=acfg.MAX_TOKENS,
                temperature=acfg.TEMPERATURE, system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return msg.content[0].text
        else:
            resp = self._client.chat.completions.create(
                model=self.model, max_tokens=acfg.MAX_TOKENS,
                temperature=acfg.TEMPERATURE,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user",   "content": user_prompt}],
            )
            return resp.choices[0].message.content


# =============================================================================
# 프롬프트 템플릿
# =============================================================================

SYSTEM_PROMPT_BASE = textwrap.dedent("""
    당신은 한국도로공사(KEC) 비탈면·사면 유지관리 및 설계 전문가 AI입니다.
    반드시 [참고 문서]의 내용만을 근거로 답변하십시오.
    [참고 문서]에 근거가 없을 때는 "자료상 근거 없음"이라고 명시하십시오.
    안전 관련 사안은 보수적으로 안내하고, 현장 전문가 확인을 권고하십시오.
    답변 마지막에 반드시 [인용 출처] 블록을 포함하십시오.
    한국어로 답변하되 기술 용어는 영어 병기를 허용합니다.
""").strip()

TEMPLATE_GENERAL_QA = textwrap.dedent("""
    [참고 문서]
    {context}

    [질의]
    {query}

    위 참고 문서를 바탕으로 다음 형식으로 답변하십시오.

    ## 답변 요약
    (2~4문장 핵심 요약)

    ## 상세 설명
    (수치·기준을 포함한 구체적 설명)

    ## 유의사항
    (현장 적용 시 주의사항, 예외 조건)

    ## [인용 출처]
    [출처1] 매뉴얼명 > 장 > 절 > 페이지 > 표/그림번호
    [출처2] ...
""").strip()

TEMPLATE_CHECKLIST_REVIEW = textwrap.dedent("""
    [참고 문서 — 매뉴얼 기준]
    {context}

    [검토 대상]
    {query}

    ## 체크리스트 평가 결과

    | 항목 | 매뉴얼 기준 | 검토 대상 값/상태 | 판정 | 비고 |
    |------|------------|-----------------|------|------|

    ## 종합 의견

    ## 보완 권고사항

    ## [인용 출처]
    [출처1] 매뉴얼명 > 장 > 절 > 페이지

    ※ 판정: ✅ 적합 | ⚠ 주의 | ❌ 부적합 | 🔍 확인필요
""").strip()

TEMPLATE_INCIDENT_REPORT = textwrap.dedent("""
    [참고 문서]
    {context}

    [사고 개요]
    {query}

    # 비탈면 사고 원인 분석 및 복구 대책 보고서 (초안)

    ## 1. 사고 개요
    ## 2. 현황 분석
    ### 2.1 지형·지질 조건
    ### 2.2 강우·수문 조건
    ### 2.3 구조물 상태
    ## 3. 붕괴 원인 분석
    ### 3.1 주요 원인
    ### 3.2 기여 요인
    ## 4. 복구 대책
    ### 4.1 긴급 조치
    ### 4.2 항구 복구 방안
    ### 4.3 재발 방지
    ## 5. 향후 유지관리 계획
    ## [인용 출처]

    ※ AI 생성 초안 — 현장 조사 및 전문가 검토 필수
""").strip()

TEMPLATE_REGISTRY = {
    "general_qa":        TEMPLATE_GENERAL_QA,
    "checklist_review":  TEMPLATE_CHECKLIST_REVIEW,
    "incident_report":   TEMPLATE_INCIDENT_REPORT,
}


# =============================================================================
# 컨텍스트 조립 (chunk_type 인식)
# =============================================================================

def build_context_block(chunks: list) -> str:
    """
    chunk_type을 고려해 LLM 컨텍스트 블록을 조립한다.
    표/그림 청크는 시각적으로 구분되는 헤더를 추가한다.
    """
    lines = []
    for i, c in enumerate(chunks, 1):
        # 헤더
        if c.chunk_type == "table":
            type_label = f"[표 청크{i}]"
            figure_label = f" ({c.figure_id})" if c.figure_id else ""
        elif c.chunk_type == "figure":
            type_label = f"[그림 청크{i}]"
            figure_label = f" ({c.figure_id})" if c.figure_id else ""
        else:
            type_label = f"[텍스트 청크{i}]"
            figure_label = ""

        manual = getattr(c, "manual_id", "") or getattr(c, "doc_name", "")
        page   = getattr(c, "page_start", 0)
        method = getattr(c, "method", "")

        header = (
            f"{type_label}{figure_label} "
            f"출처: {manual} | {method} | p.{page}"
        )
        lines.append(header)
        lines.append(c.content)
        lines.append("")
    return "\n".join(lines)


def parse_citations(chunks: list) -> list[Citation]:
    seen, result = set(), []
    for c in chunks:
        manual = getattr(c, "manual_id", "") or getattr(c, "doc_name", "")
        page   = getattr(c, "page_start", 0)
        key    = (manual, page, getattr(c, "figure_id", ""))
        if key in seen:
            continue
        seen.add(key)
        result.append(Citation(
            manual_id  = manual,
            chapter    = getattr(c, "chapter", ""),
            section    = getattr(c, "section", ""),
            page_start = page,
            page_end   = getattr(c, "page_end", page),
            figure_id  = getattr(c, "figure_id", ""),
            chunk_type = getattr(c, "chunk_type", "text"),
        ))
    return result


def assess_evidence(answer: str, chunks: list) -> tuple[bool, str]:
    no_evidence = ["자료상 근거 없음", "자료상 근거 불충분",
                   "확인되지 않", "명시되어 있지 않"]
    if any(sig in answer for sig in no_evidence):
        return False, "없음"
    if len(chunks) >= 4:
        return True, "높음"
    elif len(chunks) >= 2:
        return True, "보통"
    return True, "낮음"


# =============================================================================
# 메인 에이전트
# =============================================================================

class KecRagAgent:
    """
    KEC 비탈면 매뉴얼 RAG 에이전트 v2.

    Milvus 연결 가능 → bge-m3 + Milvus 하이브리드 검색
    Milvus 연결 불가 → 로컬 BM25 검색 (즉시 사용 가능)

    사용 예:
        agent = KecRagAgent()
        resp = agent.ask("그라운드 앵커 자유장 최소 길이는?")
        print(resp.format_for_display())

        # 체크리스트 검토
        resp = agent.ask("사면 경사 1:1.2, 소단 폭 1.0m", template="checklist_review")

        # 사고 보고서
        resp = agent.ask("고속도로 XX km 지점 절토사면 붕괴...", template="incident_report")
    """

    def __init__(self, force_local: bool = False):
        logger.info("KecRagAgent v2 초기화 중...")

        self._embedder       = None
        self._retrieve_fn    = None
        self._local_retriever = None
        self.backend         = "local"

        if not force_local:
            self._embedder, self._retrieve_fn = _try_milvus_import()
            if self._retrieve_fn:
                self.backend = "milvus"

        if self.backend == "local":
            self._local_retriever, self._retrieve_fn = _load_local_backend()

        self.llm = LLMClient()
        logger.info("초기화 완료 — 백엔드: %s | LLM: %s/%s",
                    self.backend, acfg.LLM_PROVIDER, acfg.LLM_MODEL)

    # ── 공개 API ───────────────────────────────────────────────────────────────

    def ask(
        self,
        query: str,
        template: Literal["general_qa","checklist_review","incident_report"] = "general_qa",
        top_k: int = acfg.CONTEXT_TOP_K,
        partition_override: Optional[str] = None,
        method_override: Optional[str] = None,
    ) -> AgentResponse:
        t0 = time.time()

        # 1. 질의 전처리
        meta   = preprocess_query(query)
        q_clean = meta["cleaned_query"]
        partition = partition_override or meta["partition"]
        method    = method_override    or meta["method_filter"]

        logger.info("질의: '%s' | 공법필터: %s | 파티션: %s", q_clean, method, partition)

        # 2. 검색
        if self.backend == "milvus":
            chunks = self._retrieve_fn(
                q_clean, self._embedder, top_k=top_k, partition=partition
            )
        else:
            chunks = self._local_retriever.retrieve(
                q_clean, top_k=top_k, method_filter=method
            )

        # 3. 컨텍스트 조립
        context = build_context_block(chunks)

        # 4. 프롬프트 조립
        user_tmpl = TEMPLATE_REGISTRY.get(template, TEMPLATE_GENERAL_QA)
        user_prompt = user_tmpl.format(context=context, query=query)

        # 5. LLM 호출
        answer = self.llm.chat(SYSTEM_PROMPT_BASE, user_prompt)

        # 6. 포맷팅
        citations = parse_citations(chunks)
        has_ev, confidence = assess_evidence(answer, chunks)
        elapsed = int((time.time() - t0) * 1000)

        return AgentResponse(
            query            = query,
            answer           = answer,
            citations        = citations,
            has_evidence     = has_ev,
            confidence       = confidence,
            template_used    = template,
            retrieved_chunks = chunks,
            elapsed_ms       = elapsed,
            backend          = self.backend,
        )

    def ask_stream(
        self,
        query: str,
        template: str = "general_qa",
        top_k: int = acfg.CONTEXT_TOP_K,
    ):
        """
        스트리밍 응답 제너레이터 (Claude 전용, kec_rag_mobile_app.py 연동).
        Yields: str 청크
        """
        meta    = preprocess_query(query)
        q_clean = meta["cleaned_query"]
        method  = meta["method_filter"]

        if self.backend == "milvus":
            chunks = self._retrieve_fn(q_clean, self._embedder, top_k=top_k)
        else:
            chunks = self._local_retriever.retrieve(q_clean, top_k=top_k, method_filter=method)

        context    = build_context_block(chunks)
        user_tmpl  = TEMPLATE_REGISTRY.get(template, TEMPLATE_GENERAL_QA)
        user_prompt = user_tmpl.format(context=context, query=query)

        if acfg.LLM_PROVIDER == "claude":
            import anthropic
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            with client.messages.stream(
                model=acfg.LLM_MODEL,
                max_tokens=acfg.MAX_TOKENS,
                temperature=acfg.TEMPERATURE,
                system=SYSTEM_PROMPT_BASE,
                messages=[{"role": "user", "content": user_prompt}],
            ) as stream:
                for text in stream.text_stream:
                    yield text
        else:
            # 스트리밍 미지원 LLM은 전체 응답 한번에 반환
            answer = self.llm.chat(SYSTEM_PROMPT_BASE, user_prompt)
            yield answer

    def search_only(
        self,
        query: str,
        top_k: int = 10,
        method_filter: Optional[str] = None,
    ) -> list:
        """LLM 없이 청크 검색만 수행 (평가·디버그용)."""
        meta   = preprocess_query(query)
        method = method_filter or meta["method_filter"]
        if self.backend == "milvus":
            return self._retrieve_fn(meta["cleaned_query"], self._embedder, top_k=top_k)
        return self._local_retriever.retrieve(
            meta["cleaned_query"], top_k=top_k, method_filter=method
        )


# =============================================================================
# CLI 인터랙티브 모드
# =============================================================================

if __name__ == "__main__":
    import sys

    print("KEC RAG 에이전트 v2 — 대화형 모드")
    print("종료: Ctrl+C 또는 'quit'\n")

    agent = KecRagAgent()
    print(f"백엔드: {agent.backend}\n")

    while True:
        try:
            query = input("질의 > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n종료합니다.")
            break

        if not query or query.lower() in ("quit", "exit", "q"):
            break

        # 템플릿 선택 힌트
        tmpl = "general_qa"
        if any(kw in query for kw in ["체크", "검토", "검토대상", "설계안"]):
            tmpl = "checklist_review"
        elif any(kw in query for kw in ["붕괴", "사고", "긴급", "보고서"]):
            tmpl = "incident_report"

        resp = agent.ask(query, template=tmpl)
        print("\n" + resp.format_for_display() + "\n")
