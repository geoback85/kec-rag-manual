"""
kec_rag_mobile_app_v2.py
────────────────────────────────────────────────────────────────────────────
KEC 비탈면 매뉴얼 RAG 시스템 — 모바일 웹 인터페이스 v2

v1 → v2 변경사항:
  ① AgentBridge → kec_rag_agent_v2.KecRagAgent (Milvus 없이 즉시 동작)
  ② 청크 타입 뱃지 (📝 텍스트 / 📊 표 / 🖼️ 그림)
  ③ 검색 백엔드 상태 표시 (Milvus / BM25 로컬)
  ④ 인덱스 통계 사이드바 (공법별 청크 수, 타입 분포)
  ⑤ 평가 품질 패널 (Recall@5 33.3%, 네거티브 정확도 2/2)
  ⑥ 실제 스트리밍 (ask_stream 직접 연결)

실행:
    streamlit run kec_rag_mobile_app_v2.py

    # FastAPI 백엔드 모드
    uvicorn kec_rag_mobile_app_v2:api_app --host 0.0.0.0 --port 8000

필수:
    pip install streamlit anthropic
    # Milvus 불필요 — BM25 로컬 모드 즉시 사용 가능
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import sys
import json
import sqlite3
import hashlib
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Generator

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("kec_mobile_v2")


# =============================================================================
# 설정
# =============================================================================

class AppConfig:
    STREAMLIT_PORT: int   = int(os.getenv("STREAMLIT_PORT", "8501"))
    API_HOST:       str   = os.getenv("API_HOST", "0.0.0.0")
    API_PORT:       int   = int(os.getenv("API_PORT", "8000"))
    RAG_AGENT_ENDPOINT: str = os.getenv("RAG_AGENT_ENDPOINT", "")
    CACHE_DB_PATH:  str   = os.getenv("CACHE_DB_PATH", str(_HERE / "kec_faq_cache.db"))
    CACHE_MAX_SIZE: int   = 500
    APP_TITLE:      str   = "KEC 비탈면 매뉴얼 AI 도우미"
    APP_ICON:       str   = "🏔️"
    THEME_PRIMARY:  str   = "#1E4D9B"
    THEME_ACCENT:   str   = "#F0A500"
    STREAM_CHUNK_DELAY: float = 0.015
    MAX_HISTORY_ITEMS:  int   = 20
    CHUNKS_FILE:    str   = os.getenv("CHUNKS_FILE", str(_HERE / "manual_chunks_v2.jsonl"))


# =============================================================================
# SQLite FAQ 캐시 (v1과 동일)
# =============================================================================

class FaqCache:
    def __init__(self, db_path: str = AppConfig.CACHE_DB_PATH) -> None:
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS faq_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_hash TEXT NOT NULL UNIQUE,
                query_text TEXT NOT NULL,
                answer_json TEXT NOT NULL,
                template TEXT NOT NULL DEFAULT 'general_qa',
                hit_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_query_hash ON faq_cache(query_hash);
        """)
        self._conn.commit()

    @staticmethod
    def _hash(query: str, template: str) -> str:
        normalized = " ".join(query.strip().split()).lower()
        return hashlib.sha256(f"{template}::{normalized}".encode()).hexdigest()

    def get(self, query: str, template: str = "general_qa") -> Optional[dict]:
        qhash = self._hash(query, template)
        row = self._conn.execute(
            "SELECT answer_json FROM faq_cache WHERE query_hash = ?", (qhash,)
        ).fetchone()
        if row:
            self._conn.execute(
                "UPDATE faq_cache SET hit_count = hit_count + 1, updated_at = ? WHERE query_hash = ?",
                (datetime.now().isoformat(), qhash),
            )
            self._conn.commit()
            return json.loads(row[0])
        return None

    def put(self, query: str, answer: dict, template: str = "general_qa") -> None:
        qhash = self._hash(query, template)
        now = datetime.now().isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO faq_cache "
            "(query_hash, query_text, answer_json, template, hit_count, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, 0, ?, ?)",
            (qhash, query[:500], json.dumps(answer, ensure_ascii=False), template, now, now),
        )
        count = self._conn.execute("SELECT COUNT(*) FROM faq_cache").fetchone()[0]
        if count > AppConfig.CACHE_MAX_SIZE:
            self._conn.execute(
                "DELETE FROM faq_cache WHERE id IN "
                "(SELECT id FROM faq_cache ORDER BY hit_count ASC, updated_at ASC LIMIT ?)",
                (count - AppConfig.CACHE_MAX_SIZE,),
            )
        self._conn.commit()

    def stats(self) -> dict:
        row = self._conn.execute(
            "SELECT COUNT(*), SUM(hit_count), MAX(hit_count) FROM faq_cache"
        ).fetchone()
        return {"total_entries": row[0] or 0, "total_hits": row[1] or 0, "max_hit_count": row[2] or 0}

    def top_faqs(self, n: int = 5) -> list[dict]:
        rows = self._conn.execute(
            "SELECT query_text, template, hit_count FROM faq_cache ORDER BY hit_count DESC LIMIT ?", (n,)
        ).fetchall()
        return [{"query": r[0], "template": r[1], "hits": r[2]} for r in rows]

    def clear(self) -> int:
        count = self._conn.execute("SELECT COUNT(*) FROM faq_cache").fetchone()[0]
        self._conn.execute("DELETE FROM faq_cache")
        self._conn.commit()
        return count


# =============================================================================
# RAG 에이전트 브릿지 v2 — kec_rag_agent_v2 기반
# =============================================================================

class AgentBridge:
    """
    kec_rag_agent_v2.KecRagAgent를 감싸는 브릿지.
    Milvus 없어도 BM25 로컬 검색으로 즉시 동작.
    """

    def __init__(self) -> None:
        self._agent        = None
        self._retriever    = None
        self.backend       = "unknown"
        self._remote_ep    = AppConfig.RAG_AGENT_ENDPOINT.rstrip("/")
        self._init()

    def _init(self) -> None:
        if self._remote_ep:
            logger.info("원격 에이전트 모드: %s", self._remote_ep)
            self.backend = "remote"
            return
        try:
            from kec_rag_agent_v2 import KecRagAgent
            self._agent  = KecRagAgent()
            self.backend = self._agent.backend

            # 검색기 통계용
            if hasattr(self._agent, '_local_retriever') and self._agent._local_retriever:
                self._retriever = self._agent._local_retriever
            logger.info("에이전트 초기화: backend=%s", self.backend)
        except Exception as e:
            logger.warning("에이전트 초기화 실패 (데모 모드): %s", e)
            self.backend = "demo"

    def ask(self, query: str, template: str = "general_qa", top_k: int = 6,
            partition: Optional[str] = None) -> dict:
        if self._agent is not None:
            resp = self._agent.ask(query, template=template, top_k=top_k,
                                   partition_override=partition)
            d = resp.to_dict()
            # 검색된 청크 메타데이터 추가
            d["retrieved_chunks_meta"] = [
                {
                    "method":     getattr(c, "method", ""),
                    "page":       getattr(c, "page_start", 0),
                    "chunk_type": getattr(c, "chunk_type", "text"),
                    "figure_id":  getattr(c, "figure_id", ""),
                    "score":      getattr(c, "bm25_score", 0),
                    "snippet":    getattr(c, "content", "")[:120],
                }
                for c in resp.retrieved_chunks
            ]
            d["backend"] = self.backend
            return d
        if self._remote_ep:
            return self._remote_ask(query, template, top_k, partition)
        return self._demo_response(query)

    def stream_ask(self, query: str, template: str = "general_qa",
                   top_k: int = 6) -> Generator[str, None, None]:
        """실제 스트리밍 (v2 에이전트의 ask_stream 직접 연결)."""
        if self._agent is not None:
            try:
                yield from self._agent.ask_stream(query, template=template, top_k=top_k)
                return
            except Exception as e:
                logger.warning("스트리밍 실패, 일반 응답 폴백: %s", e)
        # 폴백: 전체 응답을 단어 단위로 분할
        result = self.ask(query, template, top_k)
        words  = result.get("answer", "").split(" ")
        for i, word in enumerate(words):
            time.sleep(AppConfig.STREAM_CHUNK_DELAY)
            yield word + (" " if i < len(words) - 1 else "")

    def get_index_stats(self) -> Optional[dict]:
        if self._retriever:
            return self._retriever.stats()
        return None

    def _remote_ask(self, query, template, top_k, partition) -> dict:
        import urllib.request
        payload = json.dumps({"query": query, "template": template,
                              "top_k": top_k, "partition": partition}).encode()
        req = urllib.request.Request(
            f"{self._remote_ep}/ask", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            return {"answer": f"[오류] {e}", "citations": [], "backend": "remote"}

    @staticmethod
    def _demo_response(query: str) -> dict:
        return {
            "query": query,
            "answer": (
                "**[데모 모드]** 에이전트가 연결되지 않았습니다.\n\n"
                "실행 방법:\n"
                "1. `ANTHROPIC_API_KEY` 환경변수 설정\n"
                "2. `kec_rag_agent_v2.py`와 `manual_chunks_v2.jsonl`이 "
                "같은 폴더에 있는지 확인\n"
                "3. `streamlit run kec_rag_mobile_app_v2.py`\n\n"
                f"질문: *{query}*"
            ),
            "citations": [], "has_evidence": False,
            "confidence": "없음", "elapsed_ms": 0, "backend": "demo",
        }


# =============================================================================
# CSS
# =============================================================================

MOBILE_CSS = """
<style>
.main .block-container { padding-top:1rem; padding-bottom:4rem; max-width:820px; }

/* 채팅 버블 */
.chat-bubble { padding:.75rem 1rem; border-radius:1.2rem; margin-bottom:.5rem;
               line-height:1.6; font-size:.95rem; word-break:keep-all; }
.chat-bubble.user      { background:#1E4D9B; color:#fff; margin-left:12%;
                          border-bottom-right-radius:.3rem; }
.chat-bubble.assistant { background:#f0f2f6; color:#222; margin-right:12%;
                          border-bottom-left-radius:.3rem; }

/* 뱃지 */
.citation-badge { display:inline-block; background:#e8f0fe; color:#1E4D9B;
                  border:1px solid #c5d5f8; border-radius:1rem;
                  padding:.1rem .5rem; font-size:.78rem; margin:.1rem; }
.chunk-text   { background:#e8f4f8; color:#0d6471; }
.chunk-table  { background:#fff3cd; color:#856404; }
.chunk-figure { background:#f3e8ff; color:#6f42c1; }

/* 신뢰도 */
.conf-high   { color:#1e7e34; font-weight:bold; }
.conf-medium { color:#856404; font-weight:bold; }
.conf-low    { color:#721c24; font-weight:bold; }
.conf-none   { color:#6c757d; }

/* 백엔드 상태 */
.backend-milvus { background:#d4edda; color:#155724; border:1px solid #c3e6cb;
                  border-radius:.4rem; padding:.15rem .5rem; font-size:.78rem; }
.backend-local  { background:#fff3cd; color:#856404; border:1px solid #ffc107;
                  border-radius:.4rem; padding:.15rem .5rem; font-size:.78rem; }
.backend-remote { background:#cce5ff; color:#004085; border:1px solid #b8daff;
                  border-radius:.4rem; padding:.15rem .5rem; font-size:.78rem; }

/* 헤더 */
.kec-header { background:linear-gradient(135deg,#1E4D9B 0%,#2c6fbe 100%);
              color:white; padding:1rem 1.5rem; border-radius:.75rem;
              margin-bottom:1.5rem; text-align:center; }
.kec-header h2 { margin:0; font-size:1.2rem; }
.kec-header p  { margin:.25rem 0 0; font-size:.82rem; opacity:.85; }

/* 모바일 */
@media (max-width:480px) {
  .main .block-container { padding-left:.5rem; padding-right:.5rem; }
  .chat-bubble { font-size:.9rem; padding:.6rem .8rem; }
  .chat-bubble.user { margin-left:3%; }
  .chat-bubble.assistant { margin-right:3%; }
  .stButton > button { min-height:44px; font-size:1rem; }
  .stChatInput { position:sticky; bottom:0; background:white;
                 padding-top:.5rem; z-index:100; }
}
@media (min-width:481px) and (max-width:768px) {
  .chat-bubble.user { margin-left:8%; }
  .chat-bubble.assistant { margin-right:8%; }
}
</style>
"""


# =============================================================================
# UI 유틸
# =============================================================================

CHUNK_TYPE_ICON = {"text": "📝", "table": "📊", "figure": "🖼️"}
CHUNK_TYPE_CSS  = {"text": "chunk-text", "table": "chunk-table", "figure": "chunk-figure"}
CHUNK_TYPE_KOR  = {"text": "본문", "table": "표", "figure": "그림"}

BACKEND_HTML = {
    "milvus": '<span class="backend-milvus">🔵 Milvus 하이브리드</span>',
    "local":  '<span class="backend-local">🟡 BM25 로컬</span>',
    "remote": '<span class="backend-remote">🌐 원격</span>',
    "demo":   '<span class="backend-local">⚪ 데모</span>',
}

def format_citations_v2(citations: list[dict], chunks_meta: list[dict]) -> str:
    """인용 출처 + 청크 타입 뱃지."""
    badges = []
    seen = set()
    for c in citations:
        mid = c.get("manual_id", "")
        pg  = c.get("page_start", 0)
        ctype = c.get("chunk_type", "text")
        key = (mid, pg)
        if key in seen:
            continue
        seen.add(key)
        icon = CHUNK_TYPE_ICON.get(ctype, "📄")
        css  = CHUNK_TYPE_CSS.get(ctype, "")
        fig  = f" ({c['figure_id']})" if c.get("figure_id") else ""
        label = f"{mid} p.{pg}{fig}"
        badges.append(f'<span class="citation-badge {css}">{icon} {label}</span>')
    return " ".join(badges)


def confidence_html(confidence: str) -> str:
    cls  = {"높음":"high","보통":"medium","낮음":"low","없음":"none"}.get(confidence,"none")
    icon = {"높음":"🟢","보통":"🟡","낮음":"🔴","없음":"⚫"}.get(confidence,"⚫")
    return f'<span class="conf-{cls}">{icon} {confidence}</span>'


def render_bubble(role: str, content: str) -> None:
    icon = "👤" if role == "user" else "🤖"
    st.markdown(f'<div class="chat-bubble {role}">{icon} {content}</div>',
                unsafe_allow_html=True)


def render_chunk_expander(chunks_meta: list[dict]) -> None:
    """검색된 청크 상세를 토글 UI로 표시."""
    if not chunks_meta:
        return
    with st.expander(f"🔍 검색된 청크 {len(chunks_meta)}개 보기"):
        for i, cm in enumerate(chunks_meta, 1):
            ctype = cm.get("chunk_type","text")
            icon  = CHUNK_TYPE_ICON.get(ctype,"📄")
            kor   = CHUNK_TYPE_KOR.get(ctype,"")
            method= cm.get("method","")
            page  = cm.get("page",0)
            score = cm.get("score",0)
            fig   = f"  ({cm['figure_id']})" if cm.get("figure_id") else ""
            st.markdown(
                f"**[{i}]** {icon} {kor} | `{method}` p.{page}{fig} "
                f"| 점수: {score:.2f}",
            )
            st.caption(cm.get("snippet",""))
            st.divider()


# =============================================================================
# 사이드바
# =============================================================================

def render_sidebar(bridge: AgentBridge, cache: FaqCache) -> dict:
    """사이드바 렌더링. 선택된 설정 dict 반환."""
    st.sidebar.title("⚙️ 설정")

    # 백엔드 상태
    bk = bridge.backend
    st.sidebar.markdown(f"**검색 백엔드:** {BACKEND_HTML.get(bk,'')}", unsafe_allow_html=True)
    st.sidebar.divider()

    # 템플릿 선택
    template_options = {
        "💬 일반 QA":      "general_qa",
        "✅ 체크리스트 검토": "checklist_review",
        "📋 사고 보고서":   "incident_report",
    }
    selected_label = st.sidebar.radio("응답 템플릿", list(template_options.keys()))
    template = template_options[selected_label]

    # 검색 설정
    top_k = st.sidebar.slider("검색 청크 수 (top_k)", 3, 10, 6)
    use_stream = st.sidebar.toggle("실시간 스트리밍", value=True)

    st.sidebar.divider()

    # 인덱스 통계
    stats = bridge.get_index_stats()
    if stats:
        st.sidebar.markdown("**📚 인덱스 통계**")
        st.sidebar.markdown(f"총 청크: **{stats['total']}**")
        bt = stats.get("by_type", {})
        st.sidebar.markdown(
            f"📝 본문 {bt.get('text',0)} / "
            f"📊 표 {bt.get('table',0)} / "
            f"🖼️ 그림 {bt.get('figure',0)}"
        )
        bm = stats.get("by_method", {})
        if bm:
            with st.sidebar.expander("공법별 청크 수"):
                for method, cnt in sorted(bm.items(), key=lambda x: -x[1]):
                    st.markdown(f"• {method}: {cnt}")
        st.sidebar.divider()

    # 평가 품질 요약
    with st.sidebar.expander("📊 검색 품질 (v1 평가)"):
        st.markdown("""
| 지표 | 값 |
|------|-----|
| 키워드 재현율@5 | 33.3% |
| Easy 난이도 | 37.9% |
| Hard 난이도 | 18.9% |
| 네거티브 정확도 | 2/2 |
""")
        st.caption("※ 골든셋 20건 기준 (BM25 로컬 검색)")

    # FAQ 캐시
    cs = cache.stats()
    st.sidebar.markdown(f"**🗂️ FAQ 캐시**: {cs['total_entries']}건 / {cs['total_hits']}회 조회")
    if cs["total_entries"] > 0:
        if st.sidebar.button("캐시 초기화"):
            n = cache.clear()
            st.sidebar.success(f"{n}건 삭제됨")

    st.sidebar.divider()
    st.sidebar.caption("KEC 비탈면 매뉴얼 AI | v2.0")
    return {"template": template, "top_k": top_k, "use_stream": use_stream}


# =============================================================================
# 빠른 질문 예시
# =============================================================================

QUICK_QUESTIONS = [
    ("📐 설계기준", "그라운드 앵커 자유장 최소 길이 기준은?"),
    ("🔨 시공기준", "숏크리트 타설 두께 기준과 철망 보강 조건은?"),
    ("🪵 보강공법", "앵커와 억지말뚝의 적용 조건 차이는?"),
    ("🔍 점검주기", "비탈면 정기점검 주기는?"),
    ("⚡ 긴급대응", "낙석 발생 시 긴급 안전 조치는?"),
    ("📋 보고서",   "비탈면 경사 1:1.0, 높이 12m, 앵커 미설치 상태"),
]


# =============================================================================
# Streamlit 메인 앱
# =============================================================================

def run_streamlit_app() -> None:
    if not STREAMLIT_AVAILABLE:
        print("streamlit 설치 필요: pip install streamlit")
        sys.exit(1)

    st.set_page_config(
        page_title=AppConfig.APP_TITLE,
        page_icon=AppConfig.APP_ICON,
        layout="centered",
        initial_sidebar_state="auto",
    )
    st.markdown(MOBILE_CSS, unsafe_allow_html=True)

    # ── 세션 상태 초기화 ──────────────────────────────────────────────────────
    if "messages"    not in st.session_state: st.session_state.messages = []
    if "bridge"      not in st.session_state: st.session_state.bridge   = AgentBridge()
    if "cache"       not in st.session_state: st.session_state.cache    = FaqCache()

    bridge: AgentBridge = st.session_state.bridge
    cache:  FaqCache    = st.session_state.cache

    # ── 사이드바 ──────────────────────────────────────────────────────────────
    cfg = render_sidebar(bridge, cache)
    template   = cfg["template"]
    top_k      = cfg["top_k"]
    use_stream = cfg["use_stream"]

    # ── 헤더 ──────────────────────────────────────────────────────────────────
    bk_html = BACKEND_HTML.get(bridge.backend, "")
    st.markdown(
        f"""<div class="kec-header">
        <h2>🏔️ {AppConfig.APP_TITLE}</h2>
        <p>비탈면 설계·시공·유지관리 매뉴얼 AI 검색 도우미 {bk_html}</p>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── 빠른 질문 버튼 ────────────────────────────────────────────────────────
    if not st.session_state.messages:
        st.markdown("**빠른 질문 예시:**")
        cols = st.columns(3)
        for idx, (label, question) in enumerate(QUICK_QUESTIONS):
            if cols[idx % 3].button(label, key=f"quick_{idx}", use_container_width=True):
                st.session_state._pending_query = question
                st.session_state._pending_template = (
                    "incident_report"
                    if any(kw in question for kw in ["보고서","붕괴","사고"])
                    else template
                )
                st.rerun()
        st.divider()

    # ── 대화 이력 렌더링 ──────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                st.markdown(msg["content"])
                # 인용 출처
                if msg.get("citations"):
                    cit_html = format_citations_v2(
                        msg["citations"],
                        msg.get("chunks_meta", []),
                    )
                    if cit_html:
                        st.markdown(cit_html, unsafe_allow_html=True)
                # 신뢰도
                if msg.get("confidence"):
                    conf_html = confidence_html(msg["confidence"])
                    elapsed   = msg.get("elapsed_ms", 0)
                    st.markdown(
                        f"{conf_html} &nbsp;|&nbsp; ⏱ {elapsed}ms",
                        unsafe_allow_html=True,
                    )
                # 검색 청크 상세
                render_chunk_expander(msg.get("chunks_meta", []))

    # ── 빠른 질문 pending 처리 ────────────────────────────────────────────────
    pending_query = getattr(st.session_state, "_pending_query", None)
    if pending_query:
        del st.session_state._pending_query
        use_tmpl = getattr(st.session_state, "_pending_template", template)
        if hasattr(st.session_state, "_pending_template"):
            del st.session_state._pending_template
        _process_query(pending_query, use_tmpl, top_k, use_stream, bridge, cache)
        st.rerun()

    # ── 채팅 입력 ─────────────────────────────────────────────────────────────
    user_input = st.chat_input("매뉴얼에서 궁금한 내용을 질문하세요...")
    if user_input:
        _process_query(user_input, template, top_k, use_stream, bridge, cache)
        st.rerun()


def _process_query(
    query: str,
    template: str,
    top_k: int,
    use_stream: bool,
    bridge: AgentBridge,
    cache: FaqCache,
) -> None:
    """질문 처리 → 메시지 이력에 추가."""
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": query})

    # 캐시 확인
    cached = cache.get(query, template)
    if cached:
        st.session_state.messages.append({
            "role": "assistant",
            "content":    cached.get("answer",""),
            "citations":  cached.get("citations",[]),
            "chunks_meta":cached.get("retrieved_chunks_meta",[]),
            "confidence": cached.get("confidence","보통"),
            "elapsed_ms": cached.get("elapsed_ms",0),
            "_cached":    True,
        })
        return

    # 에이전트 호출
    with st.spinner("🔍 매뉴얼 검색 중..."):
        if use_stream and bridge.backend in ("milvus","local"):
            # 실시간 스트리밍
            full_text = ""
            with st.chat_message("assistant"):
                placeholder = st.empty()
                for chunk in bridge.stream_ask(query, template, top_k):
                    full_text += chunk
                    placeholder.markdown(full_text + "▌")
                placeholder.markdown(full_text)
            # 전체 메타데이터는 non-stream ask로 별도 획득
            result = bridge.ask(query, template, top_k)
            result["answer"] = full_text
        else:
            result = bridge.ask(query, template, top_k)

    # 메시지 이력 추가
    msg = {
        "role":       "assistant",
        "content":    result.get("answer",""),
        "citations":  result.get("citations",[]),
        "chunks_meta":result.get("retrieved_chunks_meta",[]),
        "confidence": result.get("confidence","보통"),
        "elapsed_ms": result.get("elapsed_ms",0),
    }
    st.session_state.messages.append(msg)

    # 캐시 저장 (근거있는 답변만)
    if result.get("has_evidence", True):
        cache.put(query, result, template)


# =============================================================================
# FastAPI 백엔드 (선택)
# =============================================================================

if FASTAPI_AVAILABLE:
    api_app = FastAPI(title="KEC RAG API v2")
    _bridge: Optional[AgentBridge] = None
    _cache:  Optional[FaqCache]    = None

    def _get_bridge():
        global _bridge
        if _bridge is None: _bridge = AgentBridge()
        return _bridge
    def _get_cache():
        global _cache
        if _cache is None: _cache = FaqCache()
        return _cache

    class AskRequest(BaseModel):
        query:    str
        template: str = "general_qa"
        top_k:    int = 6
        partition: Optional[str] = None

    @api_app.post("/ask")
    def api_ask(req: AskRequest):
        cached = _get_cache().get(req.query, req.template)
        if cached:
            return cached
        result = _get_bridge().ask(req.query, req.template, req.top_k, req.partition)
        if result.get("has_evidence", True):
            _get_cache().put(req.query, result, req.template)
        return result

    @api_app.get("/ask/stream")
    def api_stream(query: str, template: str = "general_qa", top_k: int = 6):
        def gen():
            for chunk in _get_bridge().stream_ask(query, template, top_k):
                yield f"data: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    @api_app.get("/cache/stats")
    def api_cache_stats():
        return _get_cache().stats()

    @api_app.get("/index/stats")
    def api_index_stats():
        return _get_bridge().get_index_stats() or {}

    @api_app.get("/health")
    def api_health():
        bridge = _get_bridge()
        return {"status": "ok", "backend": bridge.backend,
                "timestamp": datetime.now().isoformat()}


# =============================================================================
# 엔트리포인트
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["streamlit","api","cache-test"], default="streamlit")
    args = parser.parse_args()

    if args.mode == "streamlit":
        run_streamlit_app()
    elif args.mode == "api":
        import uvicorn
        uvicorn.run("kec_rag_mobile_app_v2:api_app",
                    host=AppConfig.API_HOST, port=AppConfig.API_PORT, reload=True)
    elif args.mode == "cache-test":
        cache = FaqCache(":memory:")
        cache.put("테스트 질문", {"answer": "테스트 답변", "citations": [],
                                  "has_evidence": True, "confidence": "보통", "elapsed_ms": 0})
        hit = cache.get("테스트 질문")
        assert hit is not None and hit["answer"] == "테스트 답변"
        print("✅ 캐시 테스트 통과")
