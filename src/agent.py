from typing import Callable

from .store import EmbeddingStore

NO_CONTEXT_MESSAGE = "Không tìm thấy thông tin liên quan trong cơ sở tri thức."

PROMPT_TEMPLATE = """Bạn là trợ lý trả lời câu hỏi dựa trên tài liệu được cung cấp.

NGỮ CẢNH:
{context}

CÂU HỎI: {question}

YÊU CẦU:
- Chỉ dùng thông tin trong phần NGỮ CẢNH ở trên, không suy đoán ngoài tài liệu.
- Trích dẫn số hiệu nguồn ([1], [2], ...) ngay sau thông tin lấy từ nguồn đó.
- Nếu ngữ cảnh không đủ để trả lời, nói rõ là không tìm thấy trong tài liệu.

TRẢ LỜI:"""


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        results = self.store.search(question, top_k=top_k)
        if not results:
            # Empty store or no candidate: say so instead of paying for an LLM
            # call on an empty context, which is exactly how hallucinations start.
            return NO_CONTEXT_MESSAGE

        prompt = PROMPT_TEMPLATE.format(context=self._build_context(results), question=question)
        return self.llm_fn(prompt)

    @staticmethod
    def _build_context(results: list[dict]) -> str:
        """Number every chunk and name its source so the answer stays traceable."""
        blocks = []
        for position, result in enumerate(results, start=1):
            metadata = result.get("metadata") or {}
            source = (
                metadata.get("source_url")
                or metadata.get("source")
                or metadata.get("doc_id")
                or result.get("id")
                or "không rõ nguồn"
            )
            blocks.append(f"[{position}] (nguồn: {source}) {result['content']}")
        return "\n\n".join(blocks)
