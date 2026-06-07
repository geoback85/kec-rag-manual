"""
kec_rag_mobile_app_v2.py — KEC RAG Streamlit App (Standalone)
"""

import os
import sys
import json
import gzip
import math
import re
import logging
from pathlib import Path
from typing import Optional
from collections import defaultdict, Counter
from dataclasses import dataclass

import streamlit as st

_HERE = Path(__file__).parent
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kec_rag_app")

CHUNKS_FILE = _HERE / "manual_chunks_v2.jsonl.gz"
if not CHUNKS_FILE.exists():
    CHUNKS_FILE = _HERE / "manual_chunks_v2.jsonl"


# ── BM25 검색기 ──────────────────────────────────────────────────────────────

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
def load_index():
    if not CHUNKS_FILE.exists():
        return None

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

    n = len(chunks)
    if n == 0:
        return None

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

    avgdl /= n
    idf = {t: math.log((n - f + 0.5) / (f + 0.5) + 1) for t, f in df.items()}

    return {"chunks": chunks, "tf": tf_list, "idf": idf, "avgdl": avgdl}


def search(index, query: str, top_k: int = 6, method_filter: str = None) -> list:
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
        score = sum(
            idf.get(t, 0) * tf[t] * (k1 + 1) / (
                tf[t] + k1 * (1 - b + b * dl / max(avgdl, 1))
            )
            for t in q_tokens if t in tf
        )
        if score > 0:
            w = {"text": 1.0, "table": 1.2, "figure": 0.9}.get(chunk.chunk_type, 1.0)
            scores.append((i, score * w))

    scores.sort(key=lambda x: x[1], reverse=True)
    result = []
    seen_content = set()  # 중복 내용 제거

    for idx, sc in scores:
        if len(result) >= top_k:
            break
        c = chunks[idx]
        # 앞 100자 기준으로 중복 탐지
        content_sig = c.content[:100].strip()
        if content_sig in seen_content:
            continue
        seen_content.add(content_sig)
        c.bm25_score = round(sc, 3)
        result.append(c)
    return result


# ── LLM 호출 ──────────────────────────────────────────────────────────────────

def get_secret(key: str) -> str:
    val = os.getenv(key, "")
    if not val:
        try:
            val = st.secrets.get(key, "")
        except Exception:
            pass
    return val


def detect_provider() -> tuple:
    """사용 가능한 API 제공자와 키 자동 감지 (우선순위 순)."""
    checks = [
        ("anthropic",  "ANTHROPIC_API_KEY"),
        ("groq",       "GROQ_API_KEY"),
        ("gemini",     "GEMINI_API_KEY"),
        ("openai",     "OPENAI_API_KEY"),
        ("perplexity", "PERPLEXITY_API_KEY"),
    ]
    for provider, env_key in checks:
        val = get_secret(env_key)
        if val:
            return provider, val
    return None, None


# 제공자별 설정 (OpenAI 호환 제공자)
_PROVIDER_CONFIG = {
    "groq":       {"label": "✅ Groq (LLaMA3, 무료)",
                   "base_url": "https://api.groq.com/openai/v1",
                   "model": "llama-3.3-70b-versatile"},
    "perplexity": {"label": "✅ Perplexity AI",
                   "base_url": "https://api.perplexity.ai",
                   "model": "llama-3.1-sonar-large-128k-online"},
    "openai":     {"label": "✅ GPT (OpenAI)",
                   "model": "gpt-4o-mini"},
}


def call_llm(query: str, context: str) -> str:
    provider, api_key = detect_provider()

    if not provider:
        return (
            "⚠️ **API 키가 설정되지 않았습니다.**\n\n"
            "Streamlit Cloud → 앱 Settings → **Secrets** 탭에 아래 중 하나를 추가하세요:\n\n"
            "```toml\n"
            "# ① Groq — 완전 무료, Google 계정으로 가입 (추천)\n"
            "# https://console.groq.com → API Keys → Create\n"
            "GROQ_API_KEY = \"gsk_...\"\n\n"
            "# ② Anthropic Claude\n"
            "ANTHROPIC_API_KEY = \"sk-ant-...\"\n\n"
            "# ③ OpenAI\n"
            "OPENAI_API_KEY = \"sk-...\"\n"
            "```"
        )

    system = (
        "당신은 한국도로공사(KEC) 비탈면·사면 유지관리 전문가 AI입니다. "
        "반드시 [참고 문서] 내용만을 근거로 답변하세요. "
        "근거가 없으면 '자료상 근거 없음'이라고 명시하세요. "
        "답변 마지막에 [인용 출처]를 포함하세요."
    )
    user_msg = f"[참고 문서]\n{context}\n\n[질의]\n{query}"

    try:
        if provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            msg = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2048,
                temperature=0.1,
                system=system,
                messages=[{"role": "user", "content": user_msg}],
            )
            return msg.content[0].text

        else:  # groq / perplexity / openai — OpenAI 호환 API
            from openai import OpenAI
            cfg = _PROVIDER_CONFIG[provider]
            client = OpenAI(
                api_key=api_key,
                base_url=cfg.get("base_url"),
            )
            resp = client.chat.completions.create(
                model=cfg["model"],
                max_tokens=2048,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user_msg},
                ],
            )
            return resp.choices[0].message.content

    except Exception as e:
        return f"[LLM 오류] {e}"


def build_context(chunks: list) -> str:
    icon = {"text": "📝", "table": "📊", "figure": "🖼️"}
    lines = []
    total_chars = 0
    MAX_CONTEXT = 6000  # LLM 컨텍스트 최대 길이

    for i, c in enumerate(chunks, 1):
        content = c.content[:500]  # 청크당 최대 500자
        header = f"[청크{i}] {icon.get(c.chunk_type,'📄')} {c.manual_id} | {c.method} | p.{c.page_start}"
        block = f"{header}\n{content}\n"
        if total_chars + len(block) > MAX_CONTEXT:
            break
        lines.append(block)
        total_chars += len(block)
    return "\n".join(lines)


# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="KEC 비탈면 매뉴얼 AI 도우미",
    page_icon="🏔️",
    layout="centered",
)

st.markdown("""
<style>
.main .block-container{padding-top:1rem;max-width:820px}
.src-badge{display:inline-block;background:#e8f0fe;color:#1E4D9B;
  border:1px solid #c5d5f8;border-radius:1rem;padding:.1rem .5rem;
  font-size:.78rem;margin:.1rem}
@media(max-width:480px){.main .block-container{padding-left:.5rem;padding-right:.5rem}}
</style>
""", unsafe_allow_html=True)

# 사이드바
with st.sidebar:
    st.title("⚙️ 설정")
    top_k = st.slider("검색 청크 수", 3, 10, 6)

    idx = load_index()
    if idx:
        chunks_all = idx["chunks"]
        by_type = Counter(c.chunk_type for c in chunks_all)
        st.markdown(f"**📚 인덱스**: {len(chunks_all)}청크")
        st.caption(f"📝{by_type.get('text',0)} 📊{by_type.get('table',0)} 🖼️{by_type.get('figure',0)}")
    else:
        st.warning("청크 파일 없음")

    st.divider()
    provider, _ = detect_provider()
    cfg = _PROVIDER_CONFIG.get(provider, {})
    llm_label = cfg.get("label", "⚠️ API 키 필요")
    st.markdown(f"**LLM**: {llm_label}")
    st.caption("KEC 비탈면 매뉴얼 RAG v2")

# 헤더
st.markdown("""
<div style="background:linear-gradient(135deg,#1E4D9B,#2c6fbe);color:white;
  padding:1rem;border-radius:.75rem;margin-bottom:1.5rem;text-align:center">
  <h2 style="margin:0;font-size:1.2rem">🏔️ KEC 비탈면 매뉴얼 AI 도우미</h2>
  <p style="margin:.25rem 0 0;font-size:.82rem;opacity:.85">비탈면 설계·시공·유지관리 매뉴얼 AI 검색 (BM25 로컬)</p>
</div>""", unsafe_allow_html=True)

# 세션 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 빠른 질문
if not st.session_state.messages:
    st.markdown("**빠른 질문 예시:**")
    quick = [
        "그라운드 앵커 자유장 최소 길이는?",
        "숏크리트 타설 두께 기준은?",
        "억지말뚝 시공 허용오차는?",
        "낙석 발생 시 긴급 조치는?",
        "록볼트 정착 방식의 종류는?",
        "콘크리트 옹벽 안전율 기준은?",
    ]
    cols = st.columns(3)
    for i, q in enumerate(quick):
        if cols[i % 3].button(q[:18] + "…", key=f"q{i}", use_container_width=True):
            st.session_state._pending = q
            st.rerun()
    st.divider()

# 이전 대화
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            html = " ".join(f'<span class="src-badge">{s}</span>' for s in msg["sources"])
            st.markdown(html, unsafe_allow_html=True)

# 입력 처리
pending = getattr(st.session_state, "_pending", None)
if pending:
    del st.session_state._pending
    user_query = pending
else:
    user_query = st.chat_input("매뉴얼에서 궁금한 내용을 질문하세요...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        idx = load_index()
        with st.spinner("🔍 검색 중..."):
            found = search(idx, user_query, top_k=top_k) if idx else []

        if not found:
            answer = "검색 결과가 없습니다. 다른 키워드로 시도해 주세요."
            sources = []
        else:
            context = build_context(found)
            with st.spinner("💭 답변 생성 중..."):
                answer = call_llm(user_query, context)
            sources = list({f"{c.manual_id} p.{c.page_start}" for c in found[:4]})

        st.markdown(answer)
        if sources:
            html = " ".join(f'<span class="src-badge">📄 {s}</span>' for s in sources)
            st.markdown(html, unsafe_allow_html=True)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })
