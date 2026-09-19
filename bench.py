"""
Benchmark runner (Lab 07, Giai đoạn 3-4) — công cụ đo của cá nhân, không phải bài có test.

Chạy:
    python bench.py                          # mock embeddings (nhanh, KHÔNG có ngữ nghĩa)
    EMBEDDING_PROVIDER=gemini python bench.py
    python bench.py > ket_qua_benchmark.txt

Trên Windows, nếu console báo lỗi UnicodeEncodeError thì đặt PYTHONIOENCODING=utf-8.

Luồng: file .md -> tách frontmatter -> chunk phần thân -> 1 chunk = 1 Document
       -> EmbeddingStore -> 5 benchmark query -> in top-3 + chấm 2 mức.
"""
from __future__ import annotations

import os
import sys
import time
import unicodedata
from pathlib import Path

from dotenv import load_dotenv

from src.agent import NO_CONTEXT_MESSAGE, PROMPT_TEMPLATE
from src import (
    EMBEDDING_PROVIDER_ENV,
    ChunkingStrategyComparator,
    Document,
    EmbeddingStore,
    GeminiEmbedder,
    KnowledgeBaseAgent,
    LocalEmbedder,
    OpenAIEmbedder,
    RecursiveChunker,
    _mock_embed,
)

CORPUS_DIR = Path("data/thu-vien-uth")

# --- DÒNG DUY NHẤT MỖI THÀNH VIÊN ĐỔI SANG CHIẾN LƯỢC RIÊNG (Bài tập 3.1) ---
CHUNKER = RecursiveChunker(chunk_size=400)
# ---------------------------------------------------------------------------

# 5 benchmark query CHÍNH THỨC CỦA NHÓM (chốt ở REPORT_NHOM mục 3).
# Cả 4 thành viên chạy đúng bộ này, chỉ khác dòng CHUNKER ở trên.
#   expect_docs  = các doc_id được tính là tài liệu gold -> chấm ở mức TÀI LIỆU
#   expect_texts = chuỗi nguyên văn trong corpus, ngữ cảnh phải chứa MỘT trong số
#                  này -> chấm ở mức NỘI DUNG (chống "đúng tài liệu, sai đoạn")
QUERIES = [
    {
        "q": "Trả sách quá hạn thì bị phạt bao nhiêu tiền?",
        "gold": "1.000đ/cuốn/ngày (Nội quy Thư viện 01/11/2022; trang Phục vụ mượn - trả tài liệu).",
        "expect_docs": ["noi-quy-thu-vien", "phuc-vu-muon-tra-tai-lieu"],
        "expect_texts": ["1.000đ/cuốn/ngày", "1000 đồng/1 cuốn/1 ngày"],
        "filter": None,
    },
    {
        "q": "Sách mượn về nhà được gia hạn mấy lần, mỗi lần bao lâu?",
        "gold": "Được gia hạn 01 lần, thời gian 45 ngày (trang Phục vụ mượn - trả tài liệu).",
        "expect_docs": ["phuc-vu-muon-tra-tai-lieu"],
        "expect_texts": ["gia hạn: 01 lần"],
        "filter": None,
    },
    {
        # Câu CẦN metadata filter: không nêu người hỏi là ai, mà corpus có hai bản
        # quy định cùng nguồn, cùng từ vựng, khác audience và khác đáp án
        # (sinh viên: 05 tài liệu tiếng Việt — cán bộ/giảng viên: 10 tài liệu).
        "q": "Mỗi bạn đọc được mượn tối đa bao nhiêu tài liệu về nhà?",
        "gold": "Sinh viên: 05 tài liệu tiếng Việt + 03 tài liệu tiếng Anh (tham khảo).",
        "expect_docs": ["quy-dinh-muon-tra-sinh-vien", "noi-quy-thu-vien", "phuc-vu-muon-tra-tai-lieu"],
        "expect_texts": ["Tiếng Việt: 05 tài liệu", "5 cuốn Tiếng Việt"],
        "filter": {"audience": "student"},
    },
    {
        "q": "Khi vào phòng đọc được mang theo những gì?",
        "gold": "Chỉ được mang: máy tính cá nhân, sách, tập vở và dụng cụ học tập.",
        "expect_docs": ["noi-quy-thu-vien", "phuc-vu-phong-doc"],
        "expect_texts": ["Máy tính cá nhân, Sách, tập vở"],
        "filter": None,
    },
    {
        "q": "Muốn kiểm tra tỉ lệ trùng lặp cho khóa luận, đồ án thì dùng dịch vụ nào?",
        "gold": "Dịch vụ quét trùng lặp bằng phần mềm Turnitin.",
        "expect_docs": ["dich-vu-quet-trung-lap"],
        "expect_texts": ["Turnitin"],
        "filter": None,
    },
]


def parse_frontmatter(raw: str) -> tuple[dict, str]:
    """Tách YAML frontmatter đơn giản (key: value) khỏi phần thân."""
    if not raw.startswith("---"):
        return {}, raw

    _, _, rest = raw.partition("\n")
    block, sep, body = rest.partition("\n---")
    if not sep:
        return {}, raw

    metadata: dict = {}
    for line in block.splitlines():
        if line.lstrip().startswith("#"):
            continue
        key, delim, value = line.partition(":")
        if not delim:
            continue
        value = value.split(" #", 1)[0].strip().strip('"').strip("'")
        metadata[key.strip()] = value
    return metadata, body.lstrip("-").lstrip()


def load_chunked_documents(chunker) -> list[Document]:
    """Chunk XẢY RA Ở ĐÂY, bên ngoài store: 1 chunk = 1 Document."""
    documents: list[Document] = []
    for path in sorted(CORPUS_DIR.glob("*.md")):
        frontmatter, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        for position, chunk in enumerate(chunker.chunk(body)):
            documents.append(
                Document(
                    id=f"{path.stem}#{position}",
                    content=chunk,
                    # Metadata trải vào MỌI chunk, nếu không search_with_filter vô dụng.
                    metadata={**frontmatter, "doc_id": path.stem, "chunk_index": position},
                )
            )
    return documents


def pick_embedder():
    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    factories = {"local": LocalEmbedder, "openai": OpenAIEmbedder, "gemini": GeminiEmbedder}
    if provider in factories:
        try:
            return factories[provider]()
        except Exception as error:  # thiếu key/thư viện -> quay về mock, nói rõ lý do
            print(f"[!] {provider} không dùng được ({error}); dùng mock.", file=sys.stderr)
    return _mock_embed


def _fold(text: str) -> str:
    """Bỏ dấu tiếng Việt + hạ chữ thường, để so khớp chuỗi không phụ thuộc dấu."""
    decomposed = unicodedata.normalize("NFD", text.lower().replace("đ", "d"))
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def grade(results: list[dict], expect_docs: list[str], expect_texts: list[str]) -> tuple[int, str]:
    """Chấm 2 mức: doc_id trong top-3 CHƯA đủ, ngữ cảnh phải chứa câu trả lời."""
    doc_ids = [result["metadata"].get("doc_id") for result in results]
    context = _fold(" ".join(result["content"] for result in results))
    hit_ranks = [rank for rank, doc_id in enumerate(doc_ids, start=1) if doc_id in expect_docs]

    if not hit_ranks:
        return 0, "gold doc vắng mặt trong top-3"
    if not any(_fold(text) in context for text in expect_texts):
        return 0, "đúng tài liệu nhưng ngữ cảnh KHÔNG chứa câu trả lời"
    if hit_ranks[0] == 1:
        return 2, "gold ở top-1 + ngữ cảnh có đáp án"
    return 1, f"gold ở top-{hit_ranks[0]} + ngữ cảnh có đáp án"


def pick_llm():
    """LLM cho agent. Có GEMINI_API_KEY thì dùng model thật, không thì echo prompt."""
    load_dotenv(override=False)
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        try:
            from google import genai

            client = genai.Client(api_key=api_key)
            model = os.getenv("GEMINI_LLM_MODEL", "gemini-3.6-flash")
            # Free tier: 5 request/phút. Tự giãn nhịp + retry khi dính 429,
            # nếu không lượt chạy sẽ chết giữa chừng ở câu thứ 6.
            min_gap = float(os.getenv("GEMINI_MIN_GAP_SECONDS", "20"))
            last_call = [0.0]

            def call(prompt: str) -> str:
                for attempt in range(4):
                    wait = min_gap - (time.monotonic() - last_call[0])
                    if wait > 0:
                        time.sleep(wait)
                    last_call[0] = time.monotonic()
                    try:
                        response = client.models.generate_content(model=model, contents=prompt)
                        return (response.text or "").strip()
                    except Exception as error:
                        # 429 = hết quota phút, 503 = model quá tải. Cả hai đều tạm thời.
                        transient = ("RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE_EXCEEDED")
                        if attempt == 3 or not any(tag in str(error) for tag in transient):
                            raise
                        print(f"[!] lỗi tạm thời, chờ 30s rồi thử lại ({attempt + 1}/3)", file=sys.stderr)
                        time.sleep(30)
                return ""

            return call, model
        except Exception as error:
            print(f"[!] Gemini LLM không dùng được ({error}); agent chỉ echo prompt.", file=sys.stderr)
    return (lambda prompt: prompt), "echo (không chấm được câu trả lời của agent)"


def answer_over(agent, question: str, results: list[dict]) -> str:
    """Trả lời trên ĐÚNG ngữ cảnh vừa truy xuất.

    Không dùng agent.answer() ở đây: hàm đó gọi store.search() nên bỏ qua
    metadata_filter — câu trả lời in ra sẽ không khớp với top-3 hiển thị ngay trên.
    """
    if not results:
        return NO_CONTEXT_MESSAGE
    prompt = PROMPT_TEMPLATE.format(context=agent._build_context(results), question=question)
    try:
        return " ".join(agent.llm_fn(prompt).split())
    except Exception as error:
        # LLM là phần tuỳ chọn của bench: hết quota thì ghi nhận, đừng để
        # sập cả lượt đo retrieval (phần duy nhất được chấm điểm).
        return f"(không gọi được LLM: {str(error).splitlines()[0][:90]})"


def show(results: list[dict]) -> None:
    if not results:
        print("      (không có kết quả)")
        return
    for rank, result in enumerate(results, start=1):
        preview = " ".join(result["content"].split())[:88]
        print(f"      {rank}. score={result['score']:+.4f}  {result['id']:<24} {preview}...")


def main() -> int:
    embedder = pick_embedder()
    backend = getattr(embedder, "_backend_name", embedder.__class__.__name__)

    documents = load_chunked_documents(CHUNKER)
    if not documents:
        print(f"Không đọc được tài liệu nào trong {CORPUS_DIR}/")
        return 1

    store = EmbeddingStore(collection_name="bench", embedding_fn=embedder)
    store.add_documents(documents)
    llm_fn, llm_name = pick_llm()
    agent = KnowledgeBaseAgent(store=store, llm_fn=llm_fn)

    files = sorted({doc.metadata["doc_id"] for doc in documents})
    print("=" * 78)
    print(f"Chiến lược : {CHUNKER.__class__.__name__}")
    print(f"Embedding  : {backend}")
    print(f"LLM        : {llm_name}")
    print(f"Corpus     : {len(files)} tài liệu -> {store.get_collection_size()} chunk ({', '.join(files)})")
    if backend.startswith("mock"):
        print("CẢNH BÁO   : mock embedding băm MD5, KHÔNG mã hoá ngữ nghĩa — mọi score dưới đây là nhiễu.")
    print("=" * 78)

    total = 0
    for number, case in enumerate(QUERIES, start=1):
        results = store.search_with_filter(case["q"], top_k=3, metadata_filter=case["filter"])
        points, note = grade(results, case["expect_docs"], case["expect_texts"])
        total += points

        print(f"\n[Q{number}] {case['q']}")
        print(f"      filter={case['filter']}  gold={'/'.join(case['expect_docs'])}")
        print(f"      gold answer: {case['gold']}")
        show(results)
        print(f"      -> {points}/2 điểm — {note}")
        print(f"      AGENT: {answer_over(agent, case['q'], results)[:320]}")

        if case["filter"]:  # A/B bắt buộc cho câu cần filter
            print("      --- A/B: cùng câu hỏi, BỎ metadata_filter ---")
            no_filter = store.search_with_filter(case["q"], top_k=3, metadata_filter=None)
            show(no_filter)
            ab_points, ab_note = grade(no_filter, case["expect_docs"], case["expect_texts"])
            print(f"      -> nếu bỏ filter: {ab_points}/2 điểm — {ab_note}")
            print(f"      AGENT (bỏ filter): {answer_over(agent, case['q'], no_filter)[:320]}")

    print(f"\n{'=' * 78}\nTỔNG: {total}/10 điểm truy xuất ({CHUNKER.__class__.__name__} + {backend})")

    print(f"\n{'=' * 78}\nBaseline 3 chiến lược (Bài tập 3.1 bước 1) — trên phần thân đã bỏ frontmatter")
    for path in sorted(CORPUS_DIR.glob("*.md")):
        _, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        comparison = ChunkingStrategyComparator().compare(body, chunk_size=400)
        print(f"\n  {path.name}  ({len(body)} ký tự)")
        for name, stats in comparison.items():
            print(f"    {name:<14} count={stats['count']:<4} avg_length={stats['avg_length']}")

    print(f"\n{'=' * 78}\nVí dụ prompt RAG (KnowledgeBaseAgent, llm_fn = echo):")
    print(agent.answer(QUERIES[0]["q"], top_k=2)[:700])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
