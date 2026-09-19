# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Lâm Hoàng Phúc (MSSV 2A202602582)
**Nhóm:** K4-L3A — Nguyễn Văn Tài (2A202603004), Nguyễn Đăng Thực (2A202603014), Lâm Hoàng Phúc (2A202602582), Nguyễn Đức Minh (2A202602891). Vai của tôi: **R2 · Benchmark**.
**Ngày:** 2026-09-19

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

> **Môi trường chạy:** Python 3.12.7 (Windows 10), `pytest 9.1.1`. Embedding backend dùng cho phần đo đạc: `gemini-embedding-001` (3072 chiều), LLM cho agent: `gemini-3.1-flash-lite` — mọi số liệu ở mục 4 và 5 đều chạy bằng embedder thật, có đối chứng với `MockEmbedder` để thấy khác biệt.
> Trên Windows console cp1252 cần đặt `PYTHONIOENCODING=utf-8` trước khi chạy `main.py` / `bench.py`, nếu không Python sẽ ném `UnicodeEncodeError` khi in tiếng Việt (lỗi môi trường, không phải lỗi code).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Hai vector embedding **chỉ về cùng một hướng** trong không gian ngữ nghĩa, tức hai đoạn text nói về cùng một chủ đề/ý niệm — bất kể chúng dài ngắn khác nhau hay dùng từ vựng khác nhau. Cosine chỉ đo **góc**, không đo độ lớn, nên nó trả lời câu "hai đoạn này có cùng nói về một thứ không", chứ không phải "hai đoạn này có giống nhau từng chữ không".

**Ví dụ có độ tương tự CAO:**
- Câu A: `Sinh viên đăng ký học phần trên cổng học vụ.`
- Câu B: `Việc ghi danh môn học được thực hiện qua hệ thống trực tuyến của trường.`
- Tại sao tương đồng: hai câu gần như **không chia sẻ từ vựng** ("đăng ký" vs "ghi danh", "học phần" vs "môn học", "cổng học vụ" vs "hệ thống trực tuyến") nhưng mô tả đúng một hành động. Đo thật được **0.8214** — đây là bằng chứng embedding nắm *nghĩa* chứ không so khớp chuỗi.

**Ví dụ có độ tương tự THẤP:**
- Câu A: `Sinh viên đăng ký học phần trên cổng học vụ.`
- Câu B: `Con mèo ngủ trên mái nhà lúc trời mưa.`
- Tại sao khác: khác hoàn toàn chủ đề (học vụ vs đời sống), không có quan hệ ngữ nghĩa nào ngoài cùng là tiếng Việt và cùng có giới từ "trên". Đo thật được **0.5509** — thấp nhất trong 5 cặp, nhưng **vẫn không gần 0**; xem phần phản ngẫm ở mục 4.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Vì độ dài vector của text embedding phần lớn phản ánh **độ dài/tần suất từ** của đoạn text chứ không phải nội dung. Một đoạn 2 câu và một đoạn 20 câu cùng nói về học phí sẽ cùng *hướng* nhưng khác *độ lớn* — Euclid phạt nặng cặp này, cosine thì không. Thêm nữa, phần lớn mô hình đã chuẩn hoá vector về `||v|| = 1`; khi đó cosine bằng đúng tích vô hướng, nên ta được phép xếp hạng bằng dot product mà vẫn là cosine (đây là lý do docstring của `search` cho phép dùng dot).

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*
> `step = chunk_size − overlap = 500 − 50 = 450`
> `số chunk = ceil((10000 − 50) / 450) = ceil(9950 / 450) = ceil(22.11…) = 23`
>
> *Đáp án:* **23 chunks.** Kiểm chứng lại bằng chính code trong repo thay vì tin công thức suông:
> ```bash
> python -c "from src.chunking import FixedSizeChunker; print(len(FixedSizeChunker(chunk_size=500, overlap=50).chunk('a'*10000)))"
> # -> 23   (khớp công thức)
> ```
> Chunk cuối chỉ dài 100 ký tự, và tổng ký tự được lưu là **11,100** cho tài liệu gốc 10,000 — phần overlap làm phình kho lưu trữ thêm 11%.

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> `ceil((10000 − 100) / 400) = ceil(24.75) = 25` chunks — tăng 2 chunk (+8.7%), đã kiểm lại bằng `FixedSizeChunker` và ra đúng **25**. Overlap lớn hơn tốn thêm chunk và thêm chi phí embedding, nhưng đáng khi tài liệu có **thông tin nằm vắt qua ranh giới cắt**: một câu chứa con số ("hạn nộp là ngày 15") bị cắt đôi thì không chunk nào trả lời được câu hỏi đó. Overlap cho mỗi thông tin **hơn một cơ hội** xuất hiện nguyên vẹn trong ít nhất một chunk.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Tách câu bằng regex **lookbehind** `(?<=[.!?])\s+`: cắt ở khoảng trắng *đứng sau* dấu kết câu nên dấu câu vẫn dính vào câu mà nó đóng. Nếu split bằng `[.!?]\s+` thì dấu câu bị nuốt và mọi chunk thành câu cụt. Cùng một biểu thức xử lý được cả `". "`, `"! "`, `"? "` lẫn `".\n"` vì `\s+` bao cả xuống dòng. Edge case đã xử lý: text rỗng hoặc chỉ có khoảng trắng trả `[]`; mỗi câu được `strip()` và câu rỗng bị loại trước khi gom nhóm, nên không sinh chunk rỗng.
> **Edge case tôi biết là mình chưa xử lý:** chữ viết tắt và số thập phân. `"TS. Nam giảng môn này"` bị cắt sau `"TS."`, `"điểm 3.5 trở lên"` bị cắt sau `"3."`, và `"v.v. "` cũng vậy. Muốn sửa đúng thì cần danh sách viết tắt tiếng Việt + negative lookbehind cho chữ số, chi phí không tương xứng trong phạm vi lab nên tôi ghi nhận thay vì giấu.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Thuật toán chạy **hai chiều**, và chiều thứ hai mới là chỗ dễ thiếu:
> - *Đệ quy xuống:* mảnh nào vẫn dài hơn `chunk_size` thì gọi lại `_split(piece, rest)` với danh sách separator còn lại — cắt bằng ranh giới "to" (`\n\n`) trước để giữ ngữ nghĩa, chỉ hạ xuống ranh giới nhỏ hơn khi bắt buộc.
> - *Gom lên:* dùng một `buffer` nối các mảnh liền kề cho tới sát `chunk_size` rồi mới chốt chunk. Thiếu bước này, một file nhiều dòng ngắn sinh ra hàng trăm chunk vụn 5–10 ký tự và retrieval hỏng hoàn toàn (kiểm thử: 60 dòng ngắn → **6 chunk** chứ không phải 60). Khi phải đệ quy vào một mảnh dài, tôi đẩy các chunk con vào kết quả nhưng **giữ chunk con cuối lại trong `buffer`** để nó còn cơ hội gom tiếp với mảnh kế bên.
> - *Giữ separator:* trước khi gom, tôi **gắn separator trở lại đuôi mảnh mà nó theo sau** (`split()` ăn mất nó). Bỏ qua chi tiết này thì ranh giới chunk rơi vào `". "` sẽ **xoá luôn dấu chấm** — tôi phát hiện đúng lỗi đó khi tự kiểm bất biến, chi tiết ở mục 5.
>
> **Ba base case:** (1) chuỗi rỗng → `[]`; (2) mảnh đã `<= chunk_size` → giữ nguyên `[current_text]`; (3) hết separator **hoặc** gặp separator `""` → `_hard_cut()` cắt cứng theo `chunk_size`. Case (3) chính là nhánh mà test `test_empty_separators_falls_back_gracefully` (`separators=[]`) đi vào. Ngoài ra nếu separator hiện tại không có trong text thì bỏ qua, hạ thẳng xuống separator kế tiếp thay vì tạo một mảnh vô nghĩa.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Tôi **bỏ hẳn nhánh ChromaDB** và chỉ dùng in-memory. Lý do: `chromadb` không nằm trong `requirements.txt`, và code khởi tạo gốc gán `self._use_chroma = True` *trước khi* client được tạo — nếu máy chấm bài tình cờ có `chromadb`, mọi method sẽ rẽ vào nhánh chưa cài đặt và cả 14 test sập. Một store đổi backend tuỳ theo thứ đang cài trên máy thì không tái lập được.
> `add_documents` chỉ `append` — **1 `Document` = 1 record**, không tự chunk; việc chunk thuộc về tầng ngoài (`bench.py`), để người gọi quyết định cách cắt file. `_make_record` **copy** metadata (người gọi giữ quyền sở hữu dict của họ) và bảo đảm record luôn có `metadata['doc_id']` bằng `setdefault` — dùng `setdefault` chứ không gán đè để khi một file sinh nhiều chunk với id `"file#0"`, `"file#1"` thì `doc_id` do caller đặt (trỏ về **file gốc**) vẫn thắng.
> Xếp hạng dùng `compute_similarity` (cosine đầy đủ) chứ không dùng thẳng `_dot`. Docstring cho phép dot vì vector đã chuẩn hoá — và tôi đã **đo lại để xác nhận**: cả `MockEmbedder` lẫn `gemini-embedding-001` (3072 chiều) đều trả `||v|| = 1.000000`, nên ở đây dot và cosine cho **cùng một thứ hạng**. Tôi vẫn chọn cosine vì `embedding_fn` là tham số **được tiêm từ ngoài**: store không có cách nào biết hàm mà người dùng truyền vào có chuẩn hoá hay không (một bạn trong nhóm thử TF-IDF hay bag-of-words là dot sai ngay). Đổi lại chỉ là hai phép tính chuẩn (norm) mỗi lần so sánh — giá rẻ để không phải phụ thuộc vào một giả định về backend.
> Kết quả trả về **loại bỏ khoá `embedding`** (vector 3072 chiều làm bẩn output khi in ra terminal) và tie-break theo thứ tự nạp (`index`) để thứ hạng ổn định giữa các lần chạy khi có điểm bằng nhau.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> **Lọc TRƯỚC, rồi mới search.** Nếu lấy top-k rồi mới bỏ cái không khớp, k slot đã bị tài liệu sai đối tượng chiếm hết và ta có thể còn lại 0 kết quả dù store vẫn còn tài liệu hợp lệ. Quan trọng hơn: `search()` và `search_with_filter()` **đi chung một đường code** `_search_records()`, chỉ khác *tập ứng viên đầu vào*, nên hai hàm không thể lệch kết quả (đó cũng là lý do `test_no_filter_returns_all_candidates` pass hiển nhiên).
> **Mở rộng: giá trị filter nhận cả một tập giá trị.** `_matches()` coi `list`/`tuple`/`set` là "khớp nếu thuộc tập", còn giá trị đơn vẫn so khớp đẳng thức như cũ — tương thích ngược, 42/42 test vẫn pass. Lý do cần nó: đẳng thức chính xác **không diễn đạt được quan hệ bao hàm**. Tài liệu gắn `audience: all` áp dụng cho cả sinh viên, nhưng `{"audience": "student"}` vẫn loại nó và loại luôn câu trả lời nằm trong đó (đo được ở mục 5, câu 1). `{"audience": ["student", "all"]}` giữ lại cả hai. Một chi tiết dễ sai: `str` cũng là `Sequence`, nên phải loại trừ `str` tường minh, nếu không `"student"` sẽ khớp với từng ký tự của chính nó.
> `delete_document` dựng lại danh sách với mọi record có `metadata['doc_id'] != doc_id`, so sánh độ dài trước/sau để biết có xoá được gì không mà trả `True`/`False` — xoá theo `doc_id` nên một lệnh dọn sạch **mọi chunk** của cùng một file.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Ba nhịp: `store.search(question, top_k)` → dựng ngữ cảnh → `llm_fn(prompt)`. Phần đầu tư là cách dựng ngữ cảnh: mỗi chunk được **đánh số `[1] [2] [3]` kèm nguồn** (`source_url` → `source` → `doc_id` → `id`, lấy cái đầu tiên có), rồi prompt yêu cầu model **trích dẫn đúng số hiệu đó** khi trả lời. Nhờ vậy câu trả lời truy vết được về đúng chunk và đúng file — tiêu chí *Source Traceability* trong `docs/EVALUATION.md`, và với corpus quy định thì đây không phải tính năng phụ mà là điều kiện để ai đó dám dùng câu trả lời.
> Hai ràng buộc chống bịa: prompt nói rõ *chỉ dùng thông tin trong ngữ cảnh, không suy đoán*, và *nếu ngữ cảnh không đủ thì phải nói là không tìm thấy*. Trường hợp store rỗng / không có ứng viên thì trả thẳng `NO_CONTEXT_MESSAGE` — **không gọi LLM** trên ngữ cảnh rỗng, vì đó đúng là cách hallucination bắt đầu (và cũng đỡ tốn tiền API).
> Prompt viết bằng tiếng Việt vì corpus là quy định tiếng Việt — instruction cùng ngôn ngữ với ngữ cảnh cho grounding tốt hơn; docstring/comment trong `src/` giữ tiếng Anh cho đồng bộ với code có sẵn.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
$ pytest tests/ -v
============================= test session starts =============================
platform win32 -- Python 3.12.7, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\CODE\AITHUCCHIEN\LABS\K4-DAY07-LamHoangPhuc-2A202602582
plugins: anyio-4.15.1
collecting ... collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================= 42 passed in 0.09s ==============================
```

**Số lượng bài test vượt qua (pass):** **42 / 42**

`main.py` cũng chạy trọn vẹn từ đầu đến cuối (`python main.py "Chunking là gì?"`): nạp 5 tài liệu trong `data/`, in top-3 kèm score và source, rồi in câu trả lời của `KnowledgeBaseAgent`. Dòng `Skipping missing file: data/customer_support_playbook.txt` là bình thường — repo không có file đó.

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

Dự đoán được ghi **trước khi chạy**, sau đó mới gọi `compute_similarity()`. Cột Gemini là điểm thật (`gemini-embedding-001`); cột Mock để đối chứng.

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế (Gemini) | Đúng? | (Mock để đối chứng) |
|------|-----------|-----------|---------|--------------|-------|-------|
| 1 — cùng nghĩa, khác từ vựng | Sinh viên đăng ký học phần trên cổng học vụ. | Việc ghi danh môn học được thực hiện qua hệ thống trực tuyến của trường. | cao | **+0.8214** | ✅ | +0.2863 |
| 2 — cùng chủ đề, **nghĩa ngược nhau** | Sinh viên được phép rút học phần sau tuần thứ ba. | Sinh viên **không** được phép rút học phần sau tuần thứ ba. | cao | **+0.9783** | ✅ (nhưng cao hơn tôi tưởng rất nhiều) | −0.1013 |
| 3 — trùng từ khoá, khác chủ đề | Thư viện cho sinh viên mượn tài liệu học tập. | Thư viện phần mềm này cho lập trình viên mượn cấu trúc dữ liệu có sẵn. | trung bình | **+0.7550** | ✅ | +0.1866 |
| 4 — song ngữ, cùng nghĩa | Hạn nộp học phí là ngày 15 của tháng. | The tuition payment deadline is the 15th of the month. | cao | **+0.8576** | ✅ | −0.0000 |
| 5 — không liên quan | Sinh viên đăng ký học phần trên cổng học vụ. | Con mèo ngủ trên mái nhà lúc trời mưa. | thấp | **+0.5509** | ✅ (về thứ hạng) | −0.1367 |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Bất ngờ nhất là **cặp 2: hai câu mâu thuẫn nhau lại có điểm CAO NHẤT (0.9783)** — cao hơn cả cặp diễn đạt lại cùng một ý (0.8214). Chữ "không" đảo ngược hoàn toàn quy định nhưng gần như không dịch chuyển vector. Bài học trực tiếp cho RAG: cosine đo **độ giống chủ đề, không đo tính đúng/sai hay chiều khẳng định–phủ định**, nên retrieval hoàn toàn có thể đưa lên top-1 một chunk nói **ngược** với đáp án đúng, và điểm số sẽ không hề cảnh báo. Đây là lý do phải kiểm ở mức nội dung chứ không chỉ tin vào score.
> Bất ngờ thứ hai: **sàn similarity không phải 0**. Cặp hoàn toàn vô quan vẫn đạt 0.5509, tức mọi câu tiếng Việt đều "hơi giống nhau" dưới mắt mô hình này. Hệ quả thực tế: **không được đặt ngưỡng tuyệt đối kiểu `score > 0.5 là liên quan`** — chỉ có *khoảng cách tương đối* giữa top-1 và phần còn lại mới mang thông tin.
> Cặp 4 cho thấy mô hình đa ngữ ánh xạ hai ngôn ngữ vào cùng một không gian (0.8576) — câu hỏi tiếng Việt vẫn tìm được tài liệu tiếng Anh. Và cột Mock xác nhận cảnh báo của lab: `MockEmbedder` băm MD5 nên cặp song ngữ cùng nghĩa ra **−0.0000**, cặp cùng nghĩa khác từ chỉ 0.2863 — dùng mock để đo chất lượng retrieval là đo nhiễu.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá chính thức của nhóm** (chốt trong `REPORT_NHOM.md` mục 3) trên mã nguồn cá nhân của tôi trong gói `src`.

> Công cụ đo: `bench.py` (kèm trong repo) · output đầy đủ: `ket_qua_benchmark.txt`
> Lệnh: `PYTHONIOENCODING=utf-8 EMBEDDING_PROVIDER=gemini python bench.py > ket_qua_benchmark.txt`
> Cấu hình: `RecursiveChunker(chunk_size=400)` · embedding `gemini-embedding-001` · LLM `gemini-3.1-flash-lite` · corpus `data/thu-vien-uth/` (8 tài liệu → **33 chunk**) · `top_k=3`
>
> **Lưu ý về khả năng so sánh với bảng trong `REPORT_NHOM.md`:** bảng đó chạy bằng `text-embedding-3-small` + `gpt-4.1-nano`. Máy tôi không có OpenAI API key nên tôi chạy bằng Gemini — **cùng corpus, cùng 5 câu hỏi, cùng `top_k`, chỉ khác backend**. Tổng điểm trùng nhau (9/10) nhưng điểm rơi vào câu khác nhau, nên tôi ghi rõ backend ở mọi con số thay vì gộp chung.

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Trả sách quá hạn thì bị phạt bao nhiêu tiền? | `quy-dinh-muon-tra-can-bo-giang-vien#2` — "4.1 … Phạt **500đ/ngày/1 cuốn sách**" | +0.8473 | ⚠️ đúng chủ đề nhưng **sai con số**; gold (`noi-quy-thu-vien#10`, "1.000đ/cuốn/ngày") tụt xuống hạng 3 (**1/2**) | Nêu **cả hai** mức phạt kèm trích dẫn: "500đ/ngày/1 cuốn [1], [2]" và "1.000đ/cuốn/ngày [3]" |
| 2 | Sách mượn về nhà được gia hạn mấy lần, mỗi lần bao lâu? | `phuc-vu-muon-tra-tai-lieu#1` — "… gia hạn: 01 lần, 45 ngày" | +0.8445 | ✅ top-1, ngữ cảnh có đáp án (**2/2**) | "Gia hạn **01 lần** với thời gian là **45 ngày** [1]" — đúng |
| 3 | Mỗi bạn đọc được mượn tối đa bao nhiêu tài liệu về nhà? *(có `metadata_filter={"audience":"student"}`)* | `quy-dinh-muon-tra-sinh-vien#1` — "Đối với bạn đọc là Sinh viên: + Tiếng Việt: **05 tài liệu** + Tiếng Anh: 03 tài liệu" | +0.8239 | ✅ top-1, cả 3 slot đều là tài liệu sinh viên (**2/2**) | "Sinh viên: **05 tài liệu Tiếng Việt** và **03 tài liệu Tiếng Anh** [1]" — đúng đối tượng |
| 4 | Khi vào phòng đọc được mang theo những gì? | `noi-quy-thu-vien#3` — "Chỉ được mang vào phòng đọc: Máy tính cá nhân, Sách, tập vở…" | +0.8431 | ✅ top-1, ngữ cảnh có đáp án (**2/2**) | "Máy tính cá nhân [1], [2]; Sách, tập vở và dụng cụ học tập [1], [2]" — đúng |
| 5 | Muốn kiểm tra tỉ lệ trùng lặp cho khóa luận, đồ án thì dùng dịch vụ nào? | `dich-vu-quet-trung-lap#0` — "Dịch vụ quét trùng lặp … phần mềm chống đạo văn Turnitin" | +0.8473 | ✅ top-1, ngữ cảnh có đáp án (**2/2**) | "Dịch vụ quét trùng lặp bằng phần mềm **Turnitin** [1]" — đúng |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** **5/5** — nhưng chỉ **4/5 câu có gold ở top-1**, nên tổng là **9/10** điểm theo thang `docs/SCORING.md`.

**Chấm hai mức:** `bench.py` của tôi khai báo cho mỗi câu một danh sách `expect_docs` (chấm mức **tài liệu**) *và* `expect_texts` — chuỗi nguyên văn phải xuất hiện trong ngữ cảnh truy xuất được (chấm mức **nội dung**). Lần này hai cách chấm **cho cùng 9/10**, và khác với lần chạy trên corpus khởi động, sự trùng khớp này có ý nghĩa: kho có 33 chunk mà `top_k=3` chỉ lấy **9%** kho, nên việc gold lọt top-3 không còn là chuyện đương nhiên. (Trên corpus khởi động 4 chunk trước đây, top-3 lấy tới 75% kho — mọi chiến lược đều "thắng" và phép đo vô nghĩa.)

**A/B bắt buộc — metadata filter có giúp không? (câu 3)**

| Lần chạy | Top-3 | Điểm |
|---|---|---|
| **CÓ** `metadata_filter={"audience":"student"}` | 1. `quy-dinh-muon-tra-sinh-vien#1` (+0.8239) · 2. `#0` (+0.8221) · 3. `#2` (+0.8185) — **cả 3 slot đều là tài liệu sinh viên** | **2/2** |
| **KHÔNG** filter | 1. `phuc-vu-muon-tra-tai-lieu#1` (+0.8625) · 2. **`quy-dinh-muon-tra-can-bo-giang-vien#0`** (+0.8341) · 3. `quy-dinh-muon-tra-sinh-vien#1` (+0.8239) | 2/2 |

> **Điểm số không đổi, nhưng câu trả lời thì đổi — và đó mới là chỗ filter chứng minh giá trị.** Chấm theo thang retrieval thì cả hai đều 2/2 vì tài liệu gold vẫn nằm trong top-3. Nhưng khi đưa đúng ngữ cảnh đó cho agent:
> - **Có filter:** agent trả lời gọn cho đúng đối tượng — "Sinh viên: 05 tài liệu Tiếng Việt và 03 tài liệu Tiếng Anh".
> - **Không filter:** slot hạng 2 bị tài liệu **cán bộ/giảng viên** chiếm, và agent mở đầu câu trả lời bằng "**Cán bộ, Giảng viên: 10 tài liệu** [2]" rồi mới tới phần sinh viên. Một sinh viên hỏi câu này sẽ đọc được con số **10** trước tiên — sai hạn mức của chính mình.
>
> Đây đúng là kịch bản `K4_VARIANT.md` yêu cầu dựng: câu hỏi **không nêu người hỏi là ai**, corpus có hai tài liệu cùng nguồn (`split_from: quy-dinh-muon-tra-tai-lieu`), cùng từ vựng, khác `audience` và **khác đáp án**. Bài học: **thang điểm retrieval không bắt được lỗi này** — top-3 vẫn "đúng" trong cả hai lần chạy. Phải nhìn tới câu trả lời cuối mới thấy filter cứu cái gì.
>
> Đối chiếu với lần chạy thử trên 2 tài liệu khởi động `data/university/` trước khi nhóm có corpus (lần chạy đó không được commit, nêu ở đây để ghi lại quá trình): ở đó filter **làm hỏng** câu hỏi vì tài liệu gold gắn `audience: all` nên bị loại sạch. Cùng một cơ chế lọc, hai kết quả trái ngược — khác biệt nằm hoàn toàn ở **khâu chuẩn bị dữ liệu**: corpus mới tách sẵn hai bản theo `audience` nên không còn giá trị `all` mơ hồ để lọc nhầm.

**Thí nghiệm mở rộng: filter một giá trị vs. filter tập giá trị.** `REPORT_NHOM.md` mục 3 đề xuất dùng `{"audience": ["student", "all"]}` để tránh việc filter cứng loại mất tài liệu `audience: all`. Tôi cài đặt phần mở rộng đó vào `search_with_filter` (xem mục 2) rồi đo cả hai câu bị ảnh hưởng, cùng cấu hình `RecursiveChunker(400)` + `gemini-embedding-001`:

| Điều kiện lọc | Câu 3 — "Tiếng Việt: 05 tài liệu" | Câu 1 — "1.000đ/cuốn/ngày" |
|---|---|---|
| không filter | hạng 3 | hạng 3 |
| `audience="student"` | **hạng 1** ✅ | **không còn trong top-3** ❌ |
| `audience=["student","all"]` | hạng 2 | **hạng 2** ✅ |

> **Kết quả này không khớp với dự đoán trong báo cáo nhóm** — mục 3 viết rằng với filter dạng tập, "chunk chứa đáp án chiếm hạng 1 và 2" ở câu 3. Trên `src/` của tôi với Gemini, câu 3 **tụt từ hạng 1 xuống hạng 2** khi nới filter, vì chunk `phuc-vu-muon-tra-tai-lieu#1` (gắn `audience: all`, nói về hạn mức đọc tại chỗ) được nhận lại và giành top-1 với +0.8625.
> Chỗ filter dạng tập thật sự cứu là **câu 1**: đáp án `1.000đ` nằm trong tài liệu `audience: all`, nên filter cứng `student` loại sạch nó (0/2), còn `["student","all"]` kéo nó về hạng 2.
> Nói gọn: đây là **đánh đổi precision–recall thuần tuý, không phải một bản sửa thắng mọi mặt**. Filter hẹp cho câu hỏi đã biết rõ đối tượng (câu 3), filter rộng cho câu hỏi mà đáp án có thể nằm ở văn bản dùng chung (câu 1). Không có một cấu hình nào tối ưu cho cả hai, và đó chính là lý do phải đo từng câu thay vì chốt một chính sách lọc rồi áp cho toàn bộ benchmark.

**Failure case của tôi: corpus tự mâu thuẫn (câu 1).** Đây là câu duy nhất mất điểm, và nguyên nhân không nằm ở chiến lược chunking.

| Tài liệu | Mức phạt quá hạn |
|---|---|
| `noi-quy-thu-vien.md:68` *(có `document_version: "2022-11-01"`)* | **1.000đ**/cuốn/ngày |
| `phuc-vu-muon-tra-tai-lieu.md:42` | **1000 đồng**/1 cuốn/1 ngày |
| `quy-dinh-muon-tra-sinh-vien.md:49` *(`document_version: not-stated`)* | **500đ**/ngày/1 cuốn |
| `quy-dinh-muon-tra-can-bo-giang-vien.md:43` *(`not-stated`)* | **500đ**/ngày/1 cuốn |

> Bốn tài liệu công khai của cùng một thư viện, **hai con số khác nhau**. Retrieval không hề "sai": nó lấy đúng ba chunk nói về phạt quá hạn, chỉ là hai chunk đứng đầu ghi 500đ còn chunk nhóm chọn làm gold ghi 1.000đ. Score không đưa ra bất kỳ tín hiệu nào về việc bản nào còn hiệu lực.
> Agent xử lý tình huống này tốt hơn tôi dự đoán: thay vì chọn bừa một con số, nó **nêu cả hai kèm trích dẫn nguồn** ("500đ/ngày/1 cuốn [1], [2]" và "1.000đ/cuốn/ngày [3]"). Đó là hệ quả trực tiếp của hai ràng buộc trong prompt — *chỉ dùng thông tin trong ngữ cảnh* và *trích dẫn số hiệu nguồn* — và là hành vi đúng cho corpus quy định: người đọc thấy ngay có mâu thuẫn và biết đi kiểm ở đâu.
> **Đề xuất sửa (ưu tiên hướng 1):** (1) dùng `document_version` làm tiêu chí ưu tiên khi hai chunk mâu thuẫn — chỉ `noi-quy-thu-vien` có ngày ban hành (01/11/2022), ba file kia `not-stated`; (2) ở khâu thu thập, ghi chú chéo giữa các tài liệu cùng chủ đề để phát hiện mâu thuẫn **trước khi** nạp; (3) nếu chấp nhận chi phí, thêm một bước hậu kiểm phát hiện các chunk cùng chủ đề nhưng khác con số rồi đánh dấu cho người biên tập.

**Điểm yếu riêng của chiến lược Recursive mà tôi đo được: chunk trùng lặp ăn mất slot top-k.** Hai file `quy-dinh-muon-tra-sinh-vien` và `quy-dinh-muon-tra-can-bo-giang-vien` dùng **nguyên văn giống nhau** ở mục 2.2 và mục 4, nên Recursive sinh ra các cặp chunk giống hệt và chúng nhận **điểm bằng nhau tuyệt đối**:

- Câu 1: `can-bo-giang-vien#2` và `sinh-vien#3` cùng **+0.8473**, chiếm hạng 1 và 2 → chỉ còn **1 slot** cho phần còn lại của kho, và đó là lý do gold tụt xuống hạng 3.
- Câu 5: `can-bo-giang-vien#1` và `sinh-vien#2` cùng **+0.6308**, chiếm hạng 2 và 3.

> Nói cách khác, ở 2/5 câu thì `top_k=3` thực chất chỉ còn **2 vị trí có ích**. Cách sửa rẻ nhất là khử trùng lặp sau khi xếp hạng (giữ chunk có điểm cao nhất trong nhóm nội dung giống nhau, hoặc dùng `document_version`/`audience` để chọn bản đại diện) trước khi cắt top-k. Đây là điểm yếu tôi sẽ mang ra demo, vì nó chỉ lộ ra khi corpus có tài liệu tách theo `audience` — đúng thiết kế dữ liệu mà nhóm chọn.

**Đối chứng số chunk với bảng của nhóm.** Bảng trong `REPORT_NHOM.md` ban đầu ghi dòng của tôi là `35 chunk / avg 262` — số đó chạy trên `src/` của bạn Tài. Chạy trên `src/` của tôi ra **33 chunk / avg 277.7**. Tôi đã kiểm chéo hai chiến lược còn lại trên chính `src/` của mình: `FixedSizeChunker(400, 80)` → 30 / 365.8 và `SentenceChunker(3)` → 41 / 222.7, **khớp y hệt** bảng của nhóm. Vậy khác biệt nằm đúng ở `RecursiveChunker`, không phải ở corpus hay cách nạp. Ghép chunk lại thì hai bản giữ **cùng 3021/3044 ký tự** (không bản nào mất nội dung), nên nguyên nhân là **bước gom**: bản của tôi nối các mảnh liền kề tới sát `chunk_size` nên ra ít chunk hơn và dài hơn. Đã báo lại để nhóm sửa dòng của tôi.

Một chênh lệch nữa với **Phụ lục** của `REPORT_NHOM.md`: bảng đó ghi kết quả của tôi là `doc_id=10/10 · nội dung=9/10`, lấy từ lần chạy trên `src/` của Tài với `text-embedding-3-small`. Trên `src/` của tôi với `gemini-embedding-001`, log `ket_qua_benchmark.txt` cho **9/10** ở cả hai cách chấm: câu 1 mất 1 điểm vì top-1 rơi vào chunk "500đ" thay vì chunk "1.000đ". Tổng điểm trùng nhau nhưng **điểm mất ở câu khác nhau**, nên tôi không gộp hai con số này làm một.

**Đối chứng embedding backend — bằng chứng định lượng cho cảnh báo của lab.** Đo trên corpus khởi động `data/university/` (2 tài liệu, trước khi nhóm chốt corpus; log của lần chạy này đã bị ghi đè bởi lần chạy chính thức nên chỉ còn số liệu chép lại đây): một thay đổi chỉ khác ở **dấu chấm và khoảng trắng cuối chunk**, không đổi một từ nào, khiến `MockEmbedder` nhảy từ **1/10 lên 6/10** trong khi `gemini-embedding-001` giữ nguyên **8/10**. Mock băm MD5 nên đổi một ký tự là sinh vector hoàn toàn khác, thứ hạng bị xáo ngẫu nhiên. Điểm đáng sợ không phải là mock cho điểm thấp — mà là nó có thể tình cờ cho điểm **đẹp** (6/10) khiến ta tưởng chiến lược của mình tốt. Bài học áp dụng cho phần nhóm: mọi thành viên **phải dùng chung một backend** thì bảng so sánh mới đo chiến lược thay vì đo nhiễu.

**Lỗi tôi tự tìm ra khi kiểm bất biến (không có test nào bắt được):** tôi kiểm tra "chunk ghép lại có bằng văn bản gốc không" trên 4 file × 3 mức `chunk_size`, và phát hiện `RecursiveChunker` **làm mất 5 dấu chấm** trong `data/rag_system_design.md`. Nguyên nhân: `str.split(". ")` ăn mất separator, và khi ranh giới chunk rơi đúng chỗ đó thì dấu chấm biến mất hẳn — chunk kết thúc cụt giữa câu. Cách sửa: gắn separator trở lại **đuôi mảnh mà nó theo sau** trước khi gom, rồi `rstrip()` mỗi chunk lúc chốt — bỏ khoảng trắng thừa ở ranh giới nhưng **giữ lại dấu chấm**. Sau khi sửa: 42/42 test vẫn pass, 12/12 tổ hợp file × `chunk_size` không mất ký tự nào, và mọi chunk vẫn `<= chunk_size`. Bài học: **42 test pass không có nghĩa là code đúng** — test chỉ kiểm cấu trúc (kiểu trả về, số lượng, thứ tự), còn bất biến "không mất nội dung" thì phải tự nghĩ ra mà kiểm. Chính bản sửa này là lý do số chunk của tôi lệch so với bảng nhóm ở đoạn trên.

**Một lỗi nữa tôi sửa khi dựng `bench.py` cho corpus này:** ban đầu tôi in câu trả lời của agent bằng `agent.answer(question, top_k=3)`. Hàm đó gọi `store.search()` **nên bỏ qua `metadata_filter`** — câu trả lời in ra không khớp với top-3 hiển thị ngay phía trên, và ở câu 3 nó vẫn nhắc "Cán bộ, Giảng viên: 10 tài liệu" dù bảng top-3 chỉ toàn tài liệu sinh viên. Nếu không phát hiện, toàn bộ phần A/B của tôi sẽ là kết luận rút ra từ dữ liệu sai. Cách sửa: dựng prompt từ **đúng tập kết quả vừa truy xuất** (`answer_over()` trong `bench.py`, dùng lại `PROMPT_TEMPLATE` và `_build_context` của agent) thay vì gọi lại `search()`.

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> *(Điền sau buổi demo — CP7. Phần này cần nghe các nhóm khác trình bày mới viết trung thực được.)*

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 — 42/42 test pass |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 9 / 10 — corpus chính thức `data/thu-vien-uth/` (8 tài liệu → 33 chunk) + 5 query chốt của nhóm; mất 1 điểm ở câu 1 do corpus tự mâu thuẫn về mức phạt |
| **Tổng phần cá nhân** | **59 / 60** |
