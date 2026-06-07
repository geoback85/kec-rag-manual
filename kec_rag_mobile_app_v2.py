"""
kec_rag_mobile_app_v2.py — 한국도로공사 도로교통연구원 비탈면 AI Agent
"""

import os
import sys
import json
import gzip
import math
import re
import base64
import logging
from pathlib import Path
from typing import Optional
from collections import defaultdict, Counter
from dataclasses import dataclass

import streamlit as st

_HERE = Path(__file__).parent
logging.basicConfig(level=logging.INFO)

CHUNKS_FILE = _HERE / "manual_chunks_v2.jsonl.gz"
if not CHUNKS_FILE.exists():
    CHUNKS_FILE = _HERE / "manual_chunks_v2.jsonl"

# PDF 폴더 — 로컬 실행 시 자동 탐색, Cloud에서는 None
PDF_DIR = None
for candidate in [
    _HERE,                        # 저장소 폴더 안
    _HERE.parent / "2026_Manual", # 상위 폴더
]:
    if candidate.exists() and any(candidate.glob("*.pdf")):
        PDF_DIR = candidate
        break


# ── 데이터클래스 ──────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id:   str   = ""
    manual_id:  str   = ""
    source_path: str  = ""
    method:     str   = ""
    page_start: int   = 0
    page_end:   int   = 0
    chunk_type: str   = "text"
    content:    str   = ""
    figure_id:  str   = ""
    bm25_score: float = 0.0


# ── BM25 검색기 ──────────────────────────────────────────────────────────────

def tokenize(text: str) -> list:
    text = re.sub(r"[^\w\s가-힣]", " ", text.lower())
    words = [w for w in text.split() if len(w) >= 2]
    bigrams = [w[i:i+2] for w in words if len(w) >= 4 for i in range(len(w)-1)]
    return words + bigrams


def _normalize_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s.strip().lower())
    s = re.sub(r"[^\w\s가-힣.%:/-]", "", s)
    return s


def _dedup_key(c: Chunk) -> tuple:
    if c.chunk_type == "table":
        if c.figure_id:
            return ("table", c.manual_id, c.figure_id.strip().lower())
        return ("table", c.manual_id, _normalize_text(c.content)[:200])
    return (c.chunk_type, c.manual_id, _normalize_text(c.content)[:200])


@st.cache_resource(show_spinner="📚 매뉴얼 인덱스 로딩 중...")
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
                chunk_id    = rec.get("chunk_id", ""),
                manual_id   = rec.get("doc_name", ""),
                source_path = rec.get("source", ""),
                method      = rec.get("method", ""),
                page_start  = rec.get("page", 0),
                page_end    = rec.get("page", 0),
                chunk_type  = rec.get("type", "text"),
                content     = rec.get("text", ""),
                figure_id   = rec.get("figure_caption", "") or rec.get("table_title", ""),
            ))

    n = len(chunks)
    if n == 0:
        return None

    avgdl = 0.0
    tf_list, df = [], defaultdict(int)
    for c in chunks:
        tokens = tokenize(c.content)
        tf = Counter(tokens)
        tf_list.append(tf)
        for t in tf: df[t] += 1
        avgdl += len(tokens)
    avgdl /= n
    idf = {t: math.log((n - f + 0.5) / (f + 0.5) + 1) for t, f in df.items()}
    return {"chunks": chunks, "tf": tf_list, "idf": idf, "avgdl": avgdl}


def search(index, query: str, top_k: int = 6, method_filter: str = None) -> list:
    if not index:
        return []
    chunks, tf_list = index["chunks"], index["tf"]
    idf, avgdl = index["idf"], index["avgdl"]
    k1, b = 1.5, 0.75
    q_tokens = tokenize(query)
    scores = []

    for chunk, tf in zip(chunks, tf_list):
        if method_filter and method_filter not in chunk.method:
            continue
        dl = sum(tf.values())
        score = sum(
            idf.get(t, 0) * tf[t] * (k1 + 1) /
            (tf[t] + k1 * (1 - b + b * dl / max(avgdl, 1)))
            for t in q_tokens if t in tf
        )
        if score > 0:
            w = {"text": 1.0, "table": 1.2, "figure": 0.9}.get(chunk.chunk_type, 1.0)
            scores.append((chunk, score * w))

    scores.sort(key=lambda x: x[1], reverse=True)
    seen, result = set(), []
    for chunk, score in scores:
        key = _dedup_key(chunk)
        if key in seen:
            continue
        seen.add(key)
        chunk.bm25_score = round(score, 4)
        result.append(chunk)
        if len(result) >= top_k:
            break
    return result


# ── LLM ───────────────────────────────────────────────────────────────────────

def get_secret(key: str) -> str:
    val = os.getenv(key, "")
    if not val:
        try:
            val = st.secrets.get(key, "")
        except Exception:
            pass
    return val


def detect_provider() -> tuple:
    for provider, env_key in [
        ("anthropic",  "ANTHROPIC_API_KEY"),
        ("groq",       "GROQ_API_KEY"),
        ("openai",     "OPENAI_API_KEY"),
        ("perplexity", "PERPLEXITY_API_KEY"),
    ]:
        val = get_secret(env_key)
        if val:
            return provider, val
    return None, None


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

# ── 1차: 근거 기반 원답 생성 프롬프트 ───────────────────────────────────────

SYSTEM_RAW = """당신은 한국도로공사(KEC) 비탈면·사면 유지관리 및 설계 전문가 AI입니다.
반드시 [참고 문서] 내용만 근거로 답변하세요.
근거가 없으면 '자료상 근거 없음'이라고 명시하세요.

출력 규칙:
- 반드시 Markdown으로 작성할 것
- 첫 문단은 2~3문장 이내의 핵심 답변
- 필요한 경우만 bullet list 사용 (한 항목 1~2줄 이내)
- 한 문단은 3문장 이하
- 표가 필요한 경우에만 markdown table 사용
- '청크'라는 표현 금지 — '인용' 또는 '근거'로 쓸 것
- 마지막에 반드시 '## 인용 출처' 섹션을 넣고 bullet list로 정리"""

QUERY_RAW = """[참고 문서]
{context}

[질의]
{query}

아래 형식으로 답변하세요.

## 답변
사용자에게 설명하듯 자연스럽게, 반복 문장 없이 답변

## 인용 출처
- 문서명 | 공법 | 페이지"""

# ── 2차: 가독성 정리 프롬프트 ────────────────────────────────────────────────

SYSTEM_REFINE = """당신은 기술 문서를 사용자 친화적으로 정리하는 편집 AI입니다.
원답의 의미를 바꾸지 말고, 읽기 쉬운 Markdown 답변으로 다듬으세요.

규칙:
- 반복 문장 제거
- 같은 의미의 나열은 묶어서 요약
- 문단은 짧게 (3문장 이하)
- 필요 시 bullet list 사용
- '청크' 표현 금지 — '인용' 또는 '출처'로 쓸 것
- 마지막 '## 인용 출처' 섹션 반드시 유지
- 내용 추가 금지, 원답 의미만 정리"""

QUERY_REFINE = """[질문]
{query}

[원답]
{raw_answer}

위 원답을 규칙에 따라 다듬으세요."""


def build_context(chunks: list) -> str:
    icon = {"text": "📝", "table": "📊", "figure": "🖼️"}
    lines, total = [], 0
    for i, c in enumerate(chunks, 1):
        block = (
            f"[인용{i}] {icon.get(c.chunk_type,'📄')} "
            f"{c.manual_id} | {c.method} | p.{c.page_start}\n"
            f"{c.content[:500]}\n"
        )
        if total + len(block) > 6000:
            break
        lines.append(block)
        total += len(block)
    return "\n".join(lines)


def _call_single(system: str, user: str, temperature: float = 0.15) -> str:
    """공통 LLM 호출 헬퍼."""
    provider, api_key = detect_provider()
    if not provider:
        return ""
    try:
        if provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            msg = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=3000,  # 잘림 방지
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return msg.content[0].text
        else:
            from openai import OpenAI
            cfg = _PROVIDER_CONFIG[provider]
            client = OpenAI(api_key=api_key, base_url=cfg.get("base_url"))
            resp = client.chat.completions.create(
                model=cfg["model"],
                max_tokens=3000,  # 잘림 방지
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            )
            return resp.choices[0].message.content
    except Exception as e:
        return f"[LLM 오류] {e}"


def generate_raw_answer(query: str, context: str) -> str:
    """1단계: 근거 기반 원답 생성."""
    return _call_single(
        SYSTEM_RAW,
        QUERY_RAW.format(context=context, query=query),
        temperature=0.1,
    )


def refine_answer(query: str, raw_answer: str) -> str:
    """2단계: 가독성·중복 정리."""
    if not raw_answer or raw_answer.startswith("[LLM 오류]"):
        return raw_answer
    return _call_single(
        SYSTEM_REFINE,
        QUERY_REFINE.format(query=query, raw_answer=raw_answer),
        temperature=0.05,
    )


def call_llm(query: str, context: str) -> str:
    """2-step 생성: 원답 → 정리."""
    provider, _ = detect_provider()
    if not provider:
        return (
            "⚠️ **API 키가 설정되지 않았습니다.**\n\n"
            "Streamlit Cloud → 앱 Settings → **Secrets** 탭에 아래 중 하나를 추가하세요:\n"
            "```toml\n"
            "# Groq — 완전 무료 (추천)\nGROQ_API_KEY = \"gsk_...\"\n\n"
            "# Anthropic Claude\nANTHROPIC_API_KEY = \"sk-ant-...\"\n\n"
            "# OpenAI\nOPENAI_API_KEY = \"sk-...\"\n"
            "```"
        )
    raw = generate_raw_answer(query, context)
    return refine_answer(query, raw)


# ── PDF 뷰어 ──────────────────────────────────────────────────────────────────

def find_pdf(chunk: Chunk) -> Optional[Path]:
    """source_path에서 PDF 파일 탐색. Cloud에서는 None."""
    if not PDF_DIR:
        return None
    source_name = Path(chunk.source_path).stem if chunk.source_path else chunk.manual_id
    for ext in [".pdf", ".PDF"]:
        candidate = PDF_DIR / (source_name + ext)
        if candidate.exists():
            return candidate
        # 부분 매칭
        matches = list(PDF_DIR.glob(f"*{source_name[:10]}*{ext}"))
        if matches:
            return matches[0]
    return None


def render_pdf_panel(chunk: Chunk):
    """선택된 청크의 PDF를 expander에 표시."""
    pdf_path = find_pdf(chunk)
    label = f"📄 {chunk.manual_id} p.{chunk.page_start} — {chunk.figure_id or chunk.method}"

    with st.expander(f"🔍 원문 보기: {label}", expanded=True):
        if pdf_path:
            with open(str(pdf_path), "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            page = max(chunk.page_start, 1)
            st.markdown(
                f'<iframe src="data:application/pdf;base64,{b64}#page={page}" '
                f'width="100%" height="750" type="application/pdf"></iframe>',
                unsafe_allow_html=True,
            )
            st.caption(f"📌 {pdf_path.name}  |  p.{page}")
        else:
            # PDF 없으면 텍스트 청크 표시
            st.markdown(f"**{label}**")
            st.markdown(chunk.content)
            if not PDF_DIR:
                st.info("💡 PDF 뷰어는 로컬 실행 시 PDF 파일이 같은 폴더에 있을 때 활성화됩니다.")


# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="비탈면 AI Agent | KEC 도로교통연구원",
    page_icon="🏔️",
    layout="centered",
)

st.markdown("""
<style>
.main .block-container{padding-top:1rem;max-width:860px}
.src-badge{display:inline-block;background:#e8f0fe;color:#1E4D9B;
  border:1px solid #c5d5f8;border-radius:1rem;
  padding:.15rem .6rem;font-size:.78rem;margin:.1rem;cursor:pointer}
.src-badge:hover{background:#1E4D9B;color:white}
@media(max-width:480px){.main .block-container{padding-left:.5rem;padding-right:.5rem}}
</style>
""", unsafe_allow_html=True)

# ── 사이드바 ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ 설정")
    top_k = st.slider("최대 인용 수", 3, 10, 5)
    show_sources = st.toggle("인용 근거 보기", value=True)
    st.divider()

    idx = load_index()
    if idx:
        chunks_all = idx["chunks"]
        by_type = Counter(c.chunk_type for c in chunks_all)
        st.markdown(f"**📚 인덱스** {len(chunks_all)}건")
        st.caption(f"📝{by_type.get('text',0)}  📊{by_type.get('table',0)}  🖼️{by_type.get('figure',0)}")
    else:
        st.warning("청크 파일 없음")

    st.divider()
    provider, _ = detect_provider()
    cfg = _PROVIDER_CONFIG.get(provider, {})
    st.markdown(f"**LLM**: {cfg.get('label','⚠️ API 키 필요')}")

    if PDF_DIR:
        pdf_count = len(list(PDF_DIR.glob("*.pdf")))
        st.markdown(f"**PDF**: {pdf_count}개 연결됨 ✅")
    else:
        st.caption("PDF 뷰어: 로컬 실행 시 활성화")

    st.divider()
    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.session_state.last_results = []
        st.rerun()
    st.caption("비탈면 AI Agent v2 | KEC 도로교통연구원")

# ── 헤더 ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#1E4D9B,#2c6fbe);color:white;
  padding:1rem 1.5rem;border-radius:.75rem;margin-bottom:1.2rem;text-align:center">
  <p style="margin:0;font-size:.85rem;opacity:.85">한국도로공사 도로교통연구원</p>
  <h2 style="margin:.2rem 0;font-size:1.15rem;font-weight:bold">
    🏔️ 비탈면 매뉴얼 기반 AI 검색·검증 Agent
  </h2>
  <p style="margin:.2rem 0 0;font-size:.78rem;opacity:.8">
    비탈면 설계·시공·유지관리 매뉴얼 BM25 + LLM 검색
  </p>
</div>""", unsafe_allow_html=True)

# ── 세션 초기화 ───────────────────────────────────────────────────────────────
if "messages"     not in st.session_state: st.session_state.messages     = []
if "last_results" not in st.session_state: st.session_state.last_results = []
if "view_chunk"   not in st.session_state: st.session_state.view_chunk   = None

# ── 빠른 질문 ─────────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("**빠른 질문:**")
    quick = [
        ("⚓ 그라운드 앵커", "그라운드 앵커 자유장 최소 길이 기준은?"),
        ("🪨 낙석방지", "낙석방지망 설치 및 점검 기준은?"),
        ("🔩 록볼트", "록볼트 정착 방식의 종류와 적용 기준은?"),
        ("🏗️ 숏크리트", "숏크리트 타설 두께 및 철망 보강 기준은?"),
        ("🪵 억지말뚝", "억지말뚝 배치 및 시공 기준은?"),
        ("🧱 콘크리트 옹벽", "콘크리트 옹벽 설계 안전율 기준은?"),
    ]
    cols = st.columns(3)
    for i, (label, q) in enumerate(quick):
        if cols[i % 3].button(label, key=f"q{i}", use_container_width=True):
            st.session_state._pending = q
            st.rerun()
    st.divider()

# ── 대화 이력 렌더링 ──────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # 인용 배지 (assistant 메시지)
        if msg["role"] == "assistant" and msg.get("results") and show_sources:
            results = msg["results"]
            st.markdown("**인용 근거:**")
            cols_src = st.columns(min(len(results), 4))
            for j, c in enumerate(results):
                icon = {"text":"📝","table":"📊","figure":"🖼️"}.get(c.chunk_type,"📄")
                badge_label = f"{icon} {c.method or c.manual_id} p.{c.page_start}"
                with cols_src[j % 4]:
                    if st.button(badge_label, key=f"src_{msg.get('id','')}{j}",
                                 use_container_width=True, type="secondary"):
                        st.session_state.view_chunk = c

# ── PDF/원문 패널 ─────────────────────────────────────────────────────────────
if st.session_state.view_chunk:
    render_pdf_panel(st.session_state.view_chunk)
    if st.button("닫기 ✕", key="close_pdf"):
        st.session_state.view_chunk = None
        st.rerun()

# ── 입력 처리 ─────────────────────────────────────────────────────────────────
pending = getattr(st.session_state, "_pending", None)
if pending:
    del st.session_state._pending
    user_query = pending
else:
    user_query = st.chat_input("매뉴얼에서 궁금한 내용을 질문하세요...")

if user_query:
    msg_id = str(len(st.session_state.messages))

    # 사용자 메시지
    st.session_state.messages.append({"role": "user", "content": user_query, "id": msg_id})
    with st.chat_message("user"):
        st.markdown(user_query)

    # AI 답변
    with st.chat_message("assistant"):
        idx = load_index()
        with st.spinner("🔍 관련 문서 검색 중..."):
            results = search(idx, user_query, top_k=top_k) if idx else []

        if not results:
            answer = "관련 매뉴얼 내용을 찾지 못했습니다. 다른 키워드로 시도해 주세요."
        else:
            context = build_context(results)
            with st.spinner("✍️ 답변 작성 중 (1/2)..."):
                raw = generate_raw_answer(user_query, context)
            with st.spinner("✨ 답변 다듬는 중 (2/2)..."):
                answer = refine_answer(user_query, raw)

        st.markdown(answer)

        # 인용 배지
        if results and show_sources:
            st.markdown("**인용 근거:**")
            cols_src = st.columns(min(len(results), 4))
            for j, c in enumerate(results):
                icon = {"text":"📝","table":"📊","figure":"🖼️"}.get(c.chunk_type,"📄")
                badge_label = f"{icon} {c.method or c.manual_id} p.{c.page_start}"
                with cols_src[j % 4]:
                    if st.button(badge_label, key=f"new_{msg_id}_{j}",
                                 use_container_width=True, type="secondary"):
                        st.session_state.view_chunk = c

        # 검색 청크 상세 (접힌 형태)
        if results:
            with st.expander(f"검색된 인용 근거 {len(results)}건 상세"):
                icon_map = {"text":"📝","table":"📊","figure":"🖼️"}
                for j, c in enumerate(results, 1):
                    st.markdown(
                        f"**[인용{j}]** {icon_map.get(c.chunk_type,'📄')} "
                        f"`{c.manual_id}` | {c.method} | p.{c.page_start} "
                        f"| score={c.bm25_score}"
                    )
                    st.caption(c.content[:100].replace("\n"," ") + "…")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "results": results,
        "id": msg_id,
    })
    st.session_state.last_results = results
