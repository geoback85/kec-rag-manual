"""
kec_rag_mobile_app_v2.py — KEC RAG Streamlit App
"""

import os
import sys
import json
import gzip
import math
import re
import logging
import sqlite3
import hashlib
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Generator
from collections import defaultdict, Counter
from dataclasses import dataclass, field

import streamlit as st

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kec_rag_app")

# ── 앱 설정 ───────────────────────────────────────────────────────────────────
APP_TITLE = "KEC 비탈면 매뉴얼 AI 도우미"
CHUNKS_FILE_GZ   = _HERE / "manual_chunks_v2.jsonl.gz"
CHUNKS_FILE_JSON = _HERE / "manual_chunks_v2.jsonl"
CHUNKS_FILE = CHUNKS_FILE_GZ if CHUNKS_FILE_GZ.exists() else CHUNKS_FILE_JSON

# ── BM25 검색기 ───────────────────────────────────────────────────────────────

def tokenize(text: str) -> list:
    text = re.sub(r"[^\w\s가-힣]", " ", text.lower())
    words = [w for w in text.split() if len(w) >= 2]
    bigrams = []
    for w in words:
        if len(w) >= 4:
            bigrams.extend(w[i:i+2] for i in range(len(w)-1))
    return words + bigrams


@dataclass
class Chunk:
    chunk_id:   str = ""
    manual_id:  str = ""
    method:     str = ""
    page_start: int = 0
    chunk_type: str = "text"
    content:    str = ""
    figure_id:  str = ""
    bm25_score: float = 0.0


@st.cache_resource(show_spinner="🔍 매뉴얼 데이터 로딩 중...")
def load_retriever():
    """청크 파일을 로드하고 BM25 인덱스를 빌드합니다 (캐시)."""
    if not CHUNKS_FILE.exists():
        return None, []

    chunks = []
    opener = gzip.open if str(CHUNKS_FILE).endswith(".gz") else open
    with opener(str(CHUNKS_FILE), "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            chunks.append(Chunk(
                chunk_id   = rec.get("chunk_id", ""),
                manual_id  = rec.get("doc_name", ""),
                method     = rec.get("method", ""),
                page_start = rec.get("page", 0),
                chunk_type = rec.get("type", "text"),
                content    = rec.get("text", ""),
                figure_id  = rec.get("figure_caption", "") or rec.get("table_title", ""),
            ))

    # BM25 인덱스 빌드
    n = len(chunks)
    avgdl = 0.0
    tf_list = []
    df = defaultdict(int)

    for c in chunks:
        tokens = tokenize(c.content)
        tf = Counter(tokens)
        tf_list.append(tf)
        for t in tf:
            df[t] += 1
        avgdl += len(tokens)

    avgdl /= max(n, 1)
    idf = {t: math.log((n - f + 0.5) / (f + 0.5) + 1) for t, f in df.items()}

    return {"chunks": chunks, "tf": tf_list, "idf": idf, "avgdl": avgdl, "n": n}, chunks


def bm25_search(index, query: str, top_k: int = 6, method_filter=None):
    if not index:
        return []
    chunks = index["chunks"]
    tf_list = index["tf"]
    idf = index["idf"]
    avgdl = index["avgdl"]
    k1, b = 1.5, 0.75

    q_tokens = tokenize(query)
    scores = []

    for i, (chunk, tf) in enumerate(zip(chunks, tf_list)):
        if method_filter and method_filter not in chunk.method:
            continue
        dl = sum(tf.values())
        score = 0.0
        for t in q_tokens:
            if t not in tf:
                continue
            tf_val = tf[t]
            tf_norm = tf_val * (k1 + 1) / (tf_val + k1 * (1 - b + b * dl / max(avgdl, 1)))
            score += idf.get(t, 0) * tf_norm
        if score > 0:
            type_w = {"text": 1.0, "table": 1.2, "figure": 0.9}.get(chunk.chunk_type, 1.0)
            scores.append((i, score * type_w))

    scores.sort(key=lambda x: x[1], reverse=True)
    result = []
    for idx, sc in scores[:top_k]:
        c = chunks[idx]
        c.bm25_score = round(sc, 3)
        result.append(c)
    return result


# ── FAQ 캐시 ──────────────────────────────────────────────────────────────────

@st.cache_resource
def get_faq_cache():
    return {}   # 메모리 캐시 (세션 간 공유)


# ── LLM 호출 ─────────────────────────────────────────────────────────────────

def call_claude(query: str, context: str) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "⚠️ ANTHROPIC_API_KEY가 설정되지 않았습니다. Streamlit Cloud Secrets에서 설정해 주세요."

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        system = (
            "당신은 한국도로공사(KEC) 비탈면·사면 유지관리 전문가 AI입니다. "
            "반드시 [참고 문서] 내용만을 근거로 답변하세요. "
            "근거가 없으면 '자료상 근거 없음'이라고 명시하세요."
        )
        user = f"[참고 문서]\n{context}\n\n[질의]\n{query}"
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            temperature=0.1,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text
    except Exception as e:
        return f"[LLM 오류] {e}"


def build_context(chunks: list) -> str:
    lines = []
    type_icon = {"text": "📝", "table": "📊", "figure": "🖼️"}
    for i, c in enumerate(chunks, 1):
        icon = type_icon.get(c.chunk_type, "📄")
        lines.append(f"[청크{i}] {icon} {c.manual_id} | {c.method} | p.{c.page_start}")
        lines.append(c.content[:600])
        lines.append("")
    return "\n".join(lines)


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🏔️",
    layout="centered",
    initial_sidebar_state="auto",
)

st.markdown("""
<style>
.main .block-container { padding-top:1rem; max-width:820px; }
.chat-bubble { padding:.75rem 1rem; border-radius:1.2rem; margin-bottom:.5rem; line-height:1.6; }
.chat-bubble.user { background:#1E4D9B; color:#fff; margin-left:12%; }
.chat-bubble.assistant { background:#f0f2f6; color:#222; margin-right:12%; }
.source-badge { display:inline-block; background:#e8f0fe; color:#1E4D9B;
  border:1px solid #c5d5f8; border-radius:1rem; padding:.1rem .5rem; font-size:.78rem; margin:.1rem; }
@media (max-width:480px) {
  .chat-bubble.user { margin-left:3%; }
  .chat-bubble.assistant { margin-right:3%; }
}
</style>
""", unsafe_allow_html=True)

# ── 세션 초기화 ───────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── 사이드바 ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ 설정")
    top_k = st.slider("검색 청크 수", 3, 10, 6)
    st.divider()

    # 인덱스 통계
    index, all_chunks = load_retriever()
    if all_chunks:
        st.markdown(f"**📚 인덱스**: {len(all_chunks)}청크")
        by_type = Counter(c.chunk_type for c in all_chunks)
        st.markdown(f"📝{by_type.get('text',0)} / 📊{by_type.get('table',0)} / 🖼️{by_type.get('figure',0)}")
    else:
        st.warning("청크 파일 없음")

    st.divider()
    api_ok = bool(os.getenv("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY", ""))
    st.markdown(f"**LLM**: {'✅ 연결됨' if api_ok else '⚠️ API 키 없음'}")
    st.caption("KEC 비탈면 매뉴얼 RAG v2")

# ── 헤더 ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#1E4D9B,#2c6fbe);color:white;
  padding:1rem;border-radius:.75rem;margin-bottom:1.5rem;text-align:center">
  <h2 style="margin:0;font-size:1.2rem">🏔️ KEC 비탈면 매뉴얼 AI 도우미</h2>
  <p style="margin:.25rem 0 0;font-size:.82rem;opacity:.85">비탈면 설계·시공·유지관리 매뉴얼 AI 검색</p>
</div>""", unsafe_allow_html=True)

# ── 빠른 질문 버튼 ────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("**빠른 질문 예시:**")
    quick_qs = [
        "그라운드 앵커 자유장 최소 길이는?",
        "숏크리트 타설 두께 기준은?",
        "억지말뚝 시공 허용오차는?",
        "낙석 발생 시 긴급 조치는?",
        "록볼트 정착 방식의 종류는?",
        "콘크리트 옹벽 안전율 기준은?",
    ]
    cols = st.columns(3)
    for i, q in enumerate(quick_qs):
        if cols[i % 3].button(q[:20] + "...", key=f"q{i}", use_container_width=True):
            st.session_state._pending = q
            st.rerun()
    st.divider()

# ── 이전 메시지 표시 ──────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            src_html = " ".join(
                f'<span class="source-badge">{s}</span>' for s in msg["sources"]
            )
            st.markdown(src_html, unsafe_allow_html=True)

# ── 빠른 질문 처리 ────────────────────────────────────────────────────────────
pending = getattr(st.session_state, "_pending", None)
if pending:
    del st.session_state._pending
    user_query = pending
else:
    user_query = st.chat_input("매뉴얼에서 궁금한 내용을 질문하세요...")

if user_query:
    # 사용자 메시지
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 검색 + 답변
    with st.chat_message("assistant"):
        with st.spinner("🔍 검색 중..."):
            index, _ = load_retriever()
            chunks = bm25_search(index, user_query, top_k=top_k)

        if not chunks:
            answer = "검색 결과가 없습니다. 다른 키워드로 시도해 주세요."
            sources = []
        else:
            context = build_context(chunks)
            with st.spinner("💭 답변 생성 중..."):
                answer = call_claude(user_query, context)
            sources = [f"{c.manual_id} p.{c.page_start}" for c in chunks[:4]]

        st.markdown(answer)
        if sources:
            src_html = " ".join(f'<span class="source-badge">{s}</span>' for s in sources)
            st.markdown(src_html, unsafe_allow_html=True)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })
