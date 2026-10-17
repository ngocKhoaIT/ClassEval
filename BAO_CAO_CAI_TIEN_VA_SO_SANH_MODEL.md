# Báo cáo cải tiến ClassEval và so sánh model

## 1. Những nội dung đã cải tiến

### Hướng nghiên cứu

Phần đánh giá tập trung vào hai hạn chế đã nêu trong phần tổng quan:

- **Prompt chưa được khảo sát sâu:** bổ sung và thử ba kiểu prompt ở chế độ sinh cả lớp: Chain of Thought (CoT), Few-shot và Plan-then-code. Kết quả có các lần chạy GPT-3.5 và Gemini theo ba kiểu prompt này.
- **Model trong bài báo đã cũ:** đưa kết quả Gemini mới hơn vào cùng nguồn benchmark để đối chiếu với GPT-3.5 và model local. Chưa có lượt chạy GPT đời mới theo các kiểu prompt mới trong dữ liệu hiện tại; cần bổ sung sau khi xác định model và hoàn thành lần chạy.

### Công cụ chạy model local

- Tạo giao diện web local để chọn bài ClassEval, model/checkpoint, prompt và độ dài sinh; theo dõi log, xem mã sinh và tải JSON.
- Lưu kết quả chạy vào `output/model_output`. Hỗ trợ mở JSON đã có để xem lại mà không phải chạy lại model.
- Cho phép chỉ định `task_ids` khi chạy từ CLI để chạy chọn lọc bài cần thử.
- Chuyển cache Hugging Face vào `model_cache/huggingface` trong project để tránh lỗi quyền ghi vào cache mặc định của Windows.
- Inference tự chọn CUDA khi PyTorch hỗ trợ; nếu không thì chạy CPU. Đã cài PyTorch 2.5.1 + CUDA 12.4 trong `venv` và xác nhận RTX 3050 Ti chạy tensor trên `cuda:0`.
- Đọc API key từ `.env`, tránh ghi key trực tiếp vào mã nguồn hoặc README.

### Tập trung nguồn đánh giá

- Gộp các chỉ số Gemini CoT, few-shot và plan của `ClassEval2` vào `output/result/pass_at_k_result.json`, cùng nguồn chỉ số đang dùng cho các model khác.
- Chuyển trang `/compare` và API chỉ số sang đọc nguồn JSON tổng hợp này; bỏ bộ chọn nhiều nguồn.
- Chuyển thêm JSON dự đoán, kết quả chi tiết và báo cáo theo lớp của ba lần chạy Gemini vào `output/model_output` và `output/result`.

## 2. Bảng so sánh

Bảng tóm tắt dưới đây dùng **Pass@1 · greedy**. “Class” và “Method” là tỉ lệ đạt toàn bộ kiểm thử ở mức lớp và phương thức; “C.partial” và “M.partial” là tỉ lệ đạt một phần. H, I, C là **cách sinh mã** (holistic, incremental, compositional); Baseline, CoT, Few-shot và Plan-then-code là **kiểu prompt**. Vì vậy DeepSeek CoT và GPT-3.5 CoT đều có thể thuộc cách sinh H. Hàng GPT-3.5 baseline 27.0% / 52.5% lấy từ bảng bài báo được cung cấp cho báo cáo; các hàng còn lại lấy từ nguồn kết quả tổng hợp trong project.

Trên web, **Cũ** là 33 kết quả benchmark gốc của bài báo theo H/I/C. **Mới** là 13 kết quả bổ sung ở Pass@1 greedy: SantaCoder và DeepSeek chạy lại trong source, cùng GPT-3.5 và Gemini theo prompt mới được đưa vào từ ClassEval2. Các kết quả mới hiện chưa có Pass@1/3/5 theo sampling trong JSON tổng hợp.

Trang `/compare` cho chọn Pass@1 greedy hoặc Pass@1/3/5 sampling; xem mức đạt toàn bộ hay đạt một phần; lọc Cũ/Mới, model local, cách sinh và kiểu prompt; tìm kiếm, sắp xếp và lấy lần chạy tốt nhất theo từng model trong mỗi nhóm. Cột “Chênh lệch” trên web là **Method − Class** với mức đạt toàn bộ, hoặc **M.partial − C.partial** khi chọn mức đạt một phần, tính bằng điểm phần trăm.

| Model và phương pháp | Nguồn | Class | Method | C.partial | M.partial | Tasks |
|---|---|---:|---:|---:|---:|---:|
| SantaCoder · benchmark cũ (H, greedy) | Kết quả đã lưu trước lượt chạy GPU mới | 2.0% | 3.0% | 3.0% | 8.0% | 100 |
| SantaCoder · baseline GPU mới (H, greedy) | Chạy mới trên RTX 3050 Ti | 0.0% | 0.2% | 1.0% | 4.2% | 100 |
| SantaCoder + CoT | Chạy mới trên RTX 3050 Ti | 0.0% | 0.0% | 0.0% | 1.4% | 100 |
| SantaCoder + Few-shot | Chạy mới trên RTX 3050 Ti | 0.0% | 0.0% | 0.0% | 1.0% | 98 |
| SantaCoder + Plan-then-code | Chạy mới trên RTX 3050 Ti | 0.0% | 0.0% | 0.0% | 0.0% | 100 |
| DeepSeek-Coder-Instruct 1.3B + CoT | Chạy mới trên RTX 3050 Ti | 17.0% | 45.6% | 27.0% | 67.7% | 100 |
| DeepSeek-Coder-Instruct 1.3B + Few-shot | Chạy mới trên RTX 3050 Ti | 14.3% | 41.5% | 25.5% | 63.4% | 98 |
| DeepSeek-Coder-Instruct 1.3B + Plan-then-code | Chạy mới trên RTX 3050 Ti | 16.0% | 40.6% | 28.0% | 59.2% | 100 |
| GPT-3.5 Baseline | Bài báo gốc | 27.0% | 52.5% | — | — | 100 |
| GPT-3.5 + CoT | ClassEval2 | 26.0% | 50.4% | 43.0% | 69.7% | 100 |
| GPT-3.5 + Few-shot | ClassEval2 | 25.5% | 56.1% | 50.0% | 75.8% | 98 |
| GPT-3.5 + Plan-then-code | ClassEval2 | 25.0% | 57.8% | 51.0% | 75.7% | 100 |
| Gemini 3.8 Flash + CoT | ClassEval2 | 39.0% | 58.8% | 58.0% | 65.5% | 100 |
| Gemini 3.8 Flash + Few-shot | ClassEval2 | 40.8% | 59.6% | 60.2% | 66.1% | 98 |
| Gemini 3.8 Flash + Plan-then-code | ClassEval2 | **47.0%** | **73.3%** | **76.0%** | **84.5%** | 100 |
| GPT model đời mới | Chưa có kết quả trong nguồn | — | — | — | — | — |

### Đối chiếu bài báo gốc và lượt chạy mới

| Model / prompt | Bài báo gốc: Class / Method | Mới: Class / Method | Mới: C.partial / M.partial | Tasks mới |
|---|---:|---:|---:|---:|
| SantaCoder · kết quả cũ tốt nhất (I, greedy) so với baseline GPU mới (H) | 10.0% / 27.3% | 0.0% / 0.2% | 1.0% / 4.2% | 100 |
| SantaCoder + CoT | — | 0.0% / 0.0% | 0.0% / 1.4% | 100 |
| SantaCoder + Few-shot | — | 0.0% / 0.0% | 0.0% / 1.0% | 98 |
| SantaCoder + Plan-then-code | — | 0.0% / 0.0% | 0.0% / 0.0% | 100 |
| DeepSeek-Coder-Instruct 1.3B + CoT | — | 17.0% / 45.6% | 27.0% / 67.7% | 100 |
| DeepSeek-Coder-Instruct 1.3B + Few-shot | — | 14.3% / 41.5% | 25.5% / 63.4% | 98 |
| DeepSeek-Coder-Instruct 1.3B + Plan-then-code | — | 16.0% / 40.6% | 28.0% / 59.2% | 100 |

Số SantaCoder 10.0% / 27.3% là Pass@1 greedy tốt nhất trong ba cách sinh và tương ứng với **I** trong JSON; hàng SantaCoder **H** cũ riêng là 2.0% / 3.0%. Baseline GPU mới là H, nên so sánh với kết quả tốt nhất I chỉ có ý nghĩa tham khảo. Hàng GPT-3.5 baseline 27.0% / 52.5% từ bảng bài báo cũng không trùng một hàng H/I/C đơn lẻ trong JSON. DeepSeek-Coder-Instruct 1.3B không có trong bài báo gốc nên không có số cũ để điền.

### Toàn bộ dữ liệu trên trang so sánh

Các bảng tiếp theo liệt kê đầy đủ những hàng trong `output/result/pass_at_k_result.json` mà trang `/compare` dùng. Giá trị được làm tròn một chữ số thập phân như trên web. Bảng tóm tắt phía trên có thêm số tác vụ và nguồn từng lượt chạy; bốn bảng chi tiết dưới đây bao gồm cả các model bài báo chưa xuất hiện trong bảng tóm tắt.

<!-- BEGIN METRIC TABLES -->

#### Pass@1 · greedy — 46 hàng (33 cũ, 13 mới)

| Nhóm | Model | Cách sinh | Prompt | Class | Method | C.partial | M.partial | Tasks | Mã kết quả |
|---|---|:---:|---|---:|---:|---:|---:|---:|---|
| Cũ | ChatGLM | H | Baseline | 2.0% | 7.4% | 3.0% | 15.8% | 100 | `ChatGLM_H(greedy)` |
| Cũ | ChatGLM | I | Baseline | 2.0% | 9.2% | 3.0% | 21.4% | 100 | `ChatGLM_I(greedy)` |
| Cũ | ChatGLM | C | Baseline | 1.0% | 6.6% | 3.0% | 15.0% | 100 | `ChatGLM_C(greedy)` |
| Cũ | CodeGeeX | H | Baseline | 1.0% | 1.4% | 1.0% | 2.8% | 100 | `CodeGeeX_H(greedy)` |
| Cũ | CodeGeeX | I | Baseline | 9.0% | 21.6% | 12.0% | 32.7% | 100 | `CodeGeeX_I(greedy)` |
| Cũ | CodeGeeX | C | Baseline | 3.0% | 20.6% | 6.0% | 33.1% | 100 | `CodeGeeX_C(greedy)` |
| Cũ | GPT-3.5 | H | Baseline | 26.0% | 46.3% | 34.0% | 55.1% | 100 | `GPT-3.5_H(greedy)` |
| Cũ | GPT-3.5 | I | Baseline | 27.0% | 53.3% | 36.0% | 64.3% | 100 | `GPT-3.5_I(greedy)` |
| Cũ | GPT-3.5 | C | Baseline | 19.0% | 52.5% | 33.0% | 64.7% | 100 | `GPT-3.5_C(greedy)` |
| Cũ | GPT-4 | H | Baseline | 37.0% | 62.5% | 50.0% | 71.7% | 100 | `GPT-4_H(greedy)` |
| Cũ | GPT-4 | I | Baseline | 29.0% | 51.7% | 38.0% | 59.5% | 100 | `GPT-4_I(greedy)` |
| Cũ | GPT-4 | C | Baseline | 29.0% | 58.9% | 46.0% | 69.1% | 100 | `GPT-4_C(greedy)` |
| Cũ | Incoder | H | Baseline | 3.0% | 12.6% | 5.0% | 24.4% | 100 | `Incoder_H(greedy)` |
| Cũ | Incoder | I | Baseline | 6.0% | 21.4% | 7.0% | 35.5% | 100 | `Incoder_I(greedy)` |
| Cũ | Incoder | C | Baseline | 4.0% | 20.2% | 8.0% | 34.5% | 100 | `Incoder_C(greedy)` |
| Cũ | Instruct-CodeGen | H | Baseline | 7.0% | 17.0% | 11.0% | 24.6% | 100 | `Instruct-CodeGen_H(greedy)` |
| Cũ | Instruct-CodeGen | I | Baseline | 11.1% | 25.1% | 14.1% | 35.0% | 100 | `Instruct-CodeGen_I(greedy)` |
| Cũ | Instruct-CodeGen | C | Baseline | 6.0% | 21.0% | 14.0% | 32.7% | 100 | `Instruct-CodeGen_C(greedy)` |
| Cũ | Instruct-StarCoder | H | Baseline | 10.0% | 23.4% | 13.0% | 31.9% | 100 | `Instruct-StarCoder_H(greedy)` |
| Cũ | Instruct-StarCoder | I | Baseline | 7.0% | 17.4% | 13.0% | 22.4% | 100 | `Instruct-StarCoder_I(greedy)` |
| Cũ | Instruct-StarCoder | C | Baseline | 9.0% | 25.5% | 16.0% | 35.1% | 100 | `Instruct-StarCoder_C(greedy)` |
| Cũ | PolyCoder | H | Baseline | 1.0% | 1.8% | 1.0% | 1.8% | 100 | `PolyCoder_H(greedy)` |
| Cũ | PolyCoder | I | Baseline | 3.0% | 14.6% | 3.0% | 27.7% | 100 | `PolyCoder_I(greedy)` |
| Cũ | PolyCoder | C | Baseline | 2.0% | 11.0% | 5.0% | 26.1% | 100 | `PolyCoder_C(greedy)` |
| Cũ | SantaCoder | H | Baseline | 2.0% | 3.0% | 3.0% | 8.0% | 100 | `SantaCoder_H(greedy)` |
| Cũ | SantaCoder | I | Baseline | 10.0% | 27.3% | 12.0% | 40.5% | 100 | `SantaCoder_I(greedy)` |
| Cũ | SantaCoder | C | Baseline | 2.0% | 20.2% | 7.0% | 35.3% | 100 | `SantaCoder_C(greedy)` |
| Cũ | Vicuna | H | Baseline | 2.0% | 8.0% | 3.0% | 15.6% | 100 | `Vicuna_H(greedy)` |
| Cũ | Vicuna | I | Baseline | 3.0% | 10.8% | 4.0% | 22.6% | 100 | `Vicuna_I(greedy)` |
| Cũ | Vicuna | C | Baseline | 2.0% | 12.2% | 4.0% | 23.4% | 100 | `Vicuna_C(greedy)` |
| Cũ | WizardCoder | H | Baseline | 10.0% | 19.8% | 13.0% | 26.7% | 100 | `WizardCoder_H(greedy)` |
| Cũ | WizardCoder | I | Baseline | 6.0% | 25.1% | 9.0% | 31.3% | 100 | `WizardCoder_I(greedy)` |
| Cũ | WizardCoder | C | Baseline | 11.0% | 35.5% | 17.0% | 47.9% | 100 | `WizardCoder_C(greedy)` |
| Mới | DeepSeekCoder_1.3B | H | CoT | 17.0% | 45.6% | 27.0% | 67.7% | 100 | `DeepSeekCoder_1.3B_CoT_H(greedy)` |
| Mới | DeepSeekCoder_1.3B | H | Few-shot | 14.3% | 41.5% | 25.5% | 63.4% | 98 | `DeepSeekCoder_1.3B_Few-shot_H(greedy)` |
| Mới | DeepSeekCoder_1.3B | H | Plan-then-code | 16.0% | 40.6% | 28.0% | 59.2% | 100 | `DeepSeekCoder_1.3B_Plan_H(greedy)` |
| Mới | Gemini | H | CoT | 39.0% | 58.8% | 58.0% | 65.5% | 100 | `gemini_cot_greedy` |
| Mới | Gemini | H | Few-shot | 40.8% | 59.6% | 60.2% | 66.1% | 98 | `gemini_fewshot_greedy` |
| Mới | Gemini | H | Plan-then-code | 47.0% | 73.3% | 76.0% | 84.5% | 100 | `gemini_plan_greedy` |
| Mới | GPT-3.5 | H | CoT | 26.0% | 50.4% | 43.0% | 69.7% | 100 | `gpt35_cot_greedy` |
| Mới | GPT-3.5 | H | Few-shot | 25.5% | 56.1% | 50.0% | 75.8% | 98 | `gpt35_fewshot_greedy` |
| Mới | GPT-3.5 | H | Plan-then-code | 25.0% | 57.8% | 51.0% | 75.7% | 100 | `gpt35_plan_greedy` |
| Mới | SantaCoder | H | Baseline | 0.0% | 0.2% | 1.0% | 4.2% | 100 | `SantaCoder_GPU_H(greedy)` |
| Mới | SantaCoder | H | CoT | 0.0% | 0.0% | 0.0% | 1.4% | 100 | `SantaCoder_CoT_H(greedy)` |
| Mới | SantaCoder | H | Few-shot | 0.0% | 0.0% | 0.0% | 1.0% | 98 | `SantaCoder_Few-shot_H(greedy)` |
| Mới | SantaCoder | H | Plan-then-code | 0.0% | 0.0% | 0.0% | 0.0% | 100 | `SantaCoder_Plan_H(greedy)` |

#### Pass@1 · sampling — 33 hàng cũ

| Model | Cách sinh | Class | Method | C.partial | M.partial | Mã kết quả |
|---|:---:|---:|---:|---:|---:|---|
| ChatGLM | H | 1.0% | 5.8% | 1.2% | 13.4% | `ChatGLM_H` |
| ChatGLM | I | 1.2% | 7.7% | 2.0% | 17.5% | `ChatGLM_I` |
| ChatGLM | C | 1.4% | 8.2% | 2.4% | 16.5% | `ChatGLM_C` |
| CodeGeeX | H | 1.0% | 1.1% | 1.0% | 2.2% | `CodeGeeX_H` |
| CodeGeeX | I | 7.2% | 21.2% | 9.6% | 32.4% | `CodeGeeX_I` |
| CodeGeeX | C | 3.8% | 20.7% | 7.0% | 33.3% | `CodeGeeX_C` |
| GPT-3.5 | H | 29.6% | 50.4% | 40.2% | 58.5% | `GPT-3.5_H` |
| GPT-3.5 | I | 25.6% | 52.9% | 35.2% | 64.7% | `GPT-3.5_I` |
| GPT-3.5 | C | 18.2% | 51.2% | 30.4% | 63.4% | `GPT-3.5_C` |
| GPT-4 | H | 37.6% | 62.8% | 50.8% | 72.9% | `GPT-4_H` |
| GPT-4 | I | 26.2% | 52.3% | 36.2% | 61.8% | `GPT-4_I` |
| GPT-4 | C | 29.6% | 57.7% | 44.6% | 68.3% | `GPT-4_C` |
| Incoder | H | 2.6% | 12.1% | 4.8% | 23.0% | `Incoder_H` |
| Incoder | I | 6.2% | 21.1% | 8.8% | 34.2% | `Incoder_I` |
| Incoder | C | 3.4% | 20.6% | 7.2% | 34.1% | `Incoder_C` |
| Instruct-CodeGen | H | 7.4% | 17.1% | 11.4% | 25.7% | `Instruct-CodeGen_H` |
| Instruct-CodeGen | I | 8.2% | 24.9% | 12.2% | 34.9% | `Instruct-CodeGen_I` |
| Instruct-CodeGen | C | 5.8% | 20.2% | 10.4% | 31.3% | `Instruct-CodeGen_C` |
| Instruct-StarCoder | H | 10.2% | 23.1% | 14.0% | 31.7% | `Instruct-StarCoder_H` |
| Instruct-StarCoder | I | 8.4% | 20.4% | 13.6% | 26.5% | `Instruct-StarCoder_I` |
| Instruct-StarCoder | C | 9.0% | 22.8% | 14.6% | 31.5% | `Instruct-StarCoder_C` |
| PolyCoder | H | 0.0% | 0.1% | 0.2% | 0.7% | `PolyCoder_H` |
| PolyCoder | I | 1.4% | 13.2% | 2.4% | 25.4% | `PolyCoder_I` |
| PolyCoder | C | 2.6% | 10.2% | 5.4% | 24.0% | `PolyCoder_C` |
| SantaCoder | H | 1.0% | 3.0% | 2.0% | 8.8% | `SantaCoder_H` |
| SantaCoder | I | 8.6% | 27.7% | 11.8% | 41.8% | `SantaCoder_I` |
| SantaCoder | C | 3.2% | 21.0% | 8.4% | 37.1% | `SantaCoder_C` |
| Vicuna | H | 1.4% | 6.9% | 3.4% | 14.4% | `Vicuna_H` |
| Vicuna | I | 3.0% | 11.0% | 4.0% | 20.9% | `Vicuna_I` |
| Vicuna | C | 2.6% | 14.5% | 4.6% | 28.7% | `Vicuna_C` |
| WizardCoder | H | 9.2% | 20.4% | 13.8% | 26.6% | `WizardCoder_H` |
| WizardCoder | I | 5.4% | 22.0% | 7.4% | 30.2% | `WizardCoder_I` |
| WizardCoder | C | 12.2% | 35.2% | 18.4% | 47.8% | `WizardCoder_C` |

#### Pass@3 · sampling — 33 hàng cũ

| Model | Cách sinh | Class | Method | C.partial | M.partial | Mã kết quả |
|---|:---:|---:|---:|---:|---:|---|
| ChatGLM | H | 1.9% | 6.8% | 2.5% | 16.9% | `ChatGLM_H` |
| ChatGLM | I | 1.9% | 11.0% | 4.0% | 25.2% | `ChatGLM_I` |
| ChatGLM | C | 2.6% | 11.2% | 4.5% | 22.2% | `ChatGLM_C` |
| CodeGeeX | H | 1.0% | 1.4% | 1.0% | 2.9% | `CodeGeeX_H` |
| CodeGeeX | I | 9.4% | 27.1% | 12.9% | 38.6% | `CodeGeeX_I` |
| CodeGeeX | C | 4.6% | 25.2% | 8.8% | 39.5% | `CodeGeeX_C` |
| GPT-3.5 | H | 34.9% | 59.0% | 47.7% | 67.6% | `GPT-3.5_H` |
| GPT-3.5 | I | 30.2% | 57.3% | 40.6% | 69.3% | `GPT-3.5_I` |
| GPT-3.5 | C | 21.3% | 55.0% | 33.7% | 66.8% | `GPT-3.5_C` |
| GPT-4 | H | 41.3% | 67.4% | 57.5% | 77.6% | `GPT-4_H` |
| GPT-4 | I | 30.7% | 57.6% | 42.4% | 67.3% | `GPT-4_I` |
| GPT-4 | C | 34.1% | 62.3% | 50.6% | 72.7% | `GPT-4_C` |
| Incoder | H | 3.5% | 16.6% | 6.6% | 30.2% | `Incoder_H` |
| Incoder | I | 7.6% | 26.5% | 11.8% | 42.0% | `Incoder_I` |
| Incoder | C | 3.9% | 25.1% | 9.4% | 41.9% | `Incoder_C` |
| Instruct-CodeGen | H | 9.8% | 23.8% | 16.6% | 34.6% | `Instruct-CodeGen_H` |
| Instruct-CodeGen | I | 12.3% | 34.3% | 17.2% | 46.6% | `Instruct-CodeGen_I` |
| Instruct-CodeGen | C | 7.2% | 25.7% | 14.4% | 38.7% | `Instruct-CodeGen_C` |
| Instruct-StarCoder | H | 12.7% | 26.5% | 18.7% | 35.1% | `Instruct-StarCoder_H` |
| Instruct-StarCoder | I | 11.3% | 26.7% | 17.8% | 34.4% | `Instruct-StarCoder_I` |
| Instruct-StarCoder | C | 12.4% | 29.3% | 19.0% | 39.7% | `Instruct-StarCoder_C` |
| PolyCoder | H | 0.0% | 0.2% | 0.6% | 2.0% | `PolyCoder_H` |
| PolyCoder | I | 2.2% | 17.5% | 3.8% | 33.9% | `PolyCoder_I` |
| PolyCoder | C | 3.0% | 13.5% | 6.6% | 32.1% | `PolyCoder_C` |
| SantaCoder | H | 2.2% | 4.3% | 3.2% | 12.5% | `SantaCoder_H` |
| SantaCoder | I | 9.9% | 33.0% | 16.6% | 49.2% | `SantaCoder_I` |
| SantaCoder | C | 3.6% | 26.0% | 11.0% | 45.0% | `SantaCoder_C` |
| Vicuna | H | 1.9% | 11.5% | 5.7% | 24.2% | `Vicuna_H` |
| Vicuna | I | 3.6% | 15.8% | 5.8% | 30.8% | `Vicuna_I` |
| Vicuna | C | 3.0% | 18.8% | 6.4% | 37.4% | `Vicuna_C` |
| WizardCoder | H | 16.3% | 36.4% | 24.1% | 46.1% | `WizardCoder_H` |
| WizardCoder | I | 7.4% | 33.3% | 11.2% | 45.2% | `WizardCoder_I` |
| WizardCoder | C | 19.9% | 47.1% | 28.7% | 61.8% | `WizardCoder_C` |

#### Pass@5 · sampling — 33 hàng cũ

| Model | Cách sinh | Class | Method | C.partial | M.partial | Mã kết quả |
|---|:---:|---:|---:|---:|---:|---|
| ChatGLM | H | 2.0% | 7.2% | 3.0% | 18.4% | `ChatGLM_H` |
| ChatGLM | I | 2.0% | 12.0% | 5.0% | 28.1% | `ChatGLM_I` |
| ChatGLM | C | 3.0% | 12.4% | 5.0% | 24.2% | `ChatGLM_C` |
| CodeGeeX | H | 1.0% | 1.6% | 1.0% | 3.6% | `CodeGeeX_H` |
| CodeGeeX | I | 10.0% | 29.5% | 14.0% | 40.9% | `CodeGeeX_I` |
| CodeGeeX | C | 5.0% | 26.9% | 10.0% | 41.3% | `CodeGeeX_C` |
| GPT-3.5 | H | 36.0% | 61.1% | 49.0% | 70.1% | `GPT-3.5_H` |
| GPT-3.5 | I | 32.0% | 59.7% | 44.0% | 71.5% | `GPT-3.5_I` |
| GPT-3.5 | C | 22.0% | 56.5% | 35.0% | 68.1% | `GPT-3.5_C` |
| GPT-4 | H | 42.0% | 68.5% | 59.0% | 78.6% | `GPT-4_H` |
| GPT-4 | I | 33.0% | 59.5% | 45.0% | 69.5% | `GPT-4_I` |
| GPT-4 | C | 36.0% | 64.1% | 53.0% | 74.3% | `GPT-4_C` |
| Incoder | H | 4.0% | 18.6% | 7.0% | 33.1% | `Incoder_H` |
| Incoder | I | 8.0% | 29.1% | 13.0% | 44.9% | `Incoder_I` |
| Incoder | C | 4.0% | 26.5% | 10.0% | 44.5% | `Incoder_C` |
| Instruct-CodeGen | H | 11.0% | 27.1% | 19.0% | 39.1% | `Instruct-CodeGen_H` |
| Instruct-CodeGen | I | 13.0% | 37.1% | 18.0% | 51.1% | `Instruct-CodeGen_I` |
| Instruct-CodeGen | C | 8.0% | 27.9% | 16.0% | 40.9% | `Instruct-CodeGen_C` |
| Instruct-StarCoder | H | 14.0% | 27.7% | 21.0% | 36.3% | `Instruct-StarCoder_H` |
| Instruct-StarCoder | I | 13.0% | 30.1% | 20.0% | 38.3% | `Instruct-StarCoder_I` |
| Instruct-StarCoder | C | 14.0% | 31.9% | 21.0% | 42.9% | `Instruct-StarCoder_C` |
| PolyCoder | H | 0.0% | 0.4% | 1.0% | 3.0% | `PolyCoder_H` |
| PolyCoder | I | 3.0% | 19.6% | 5.0% | 37.9% | `PolyCoder_I` |
| PolyCoder | C | 3.0% | 14.8% | 7.0% | 34.9% | `PolyCoder_C` |
| SantaCoder | H | 3.0% | 5.0% | 4.0% | 13.6% | `SantaCoder_H` |
| SantaCoder | I | 10.0% | 34.9% | 19.0% | 51.7% | `SantaCoder_I` |
| SantaCoder | C | 4.0% | 28.3% | 12.0% | 47.9% | `SantaCoder_C` |
| Vicuna | H | 2.0% | 14.4% | 7.0% | 29.3% | `Vicuna_H` |
| Vicuna | I | 4.0% | 18.4% | 7.0% | 35.3% | `Vicuna_I` |
| Vicuna | C | 3.0% | 21.0% | 7.0% | 40.7% | `Vicuna_C` |
| WizardCoder | H | 20.0% | 43.7% | 30.0% | 54.1% | `WizardCoder_H` |
| WizardCoder | I | 8.0% | 38.1% | 13.0% | 51.7% | `WizardCoder_I` |
| WizardCoder | C | 23.0% | 51.1% | 32.0% | 65.7% | `WizardCoder_C` |

<!-- END METRIC TABLES -->

## 3. Cách đọc và giới hạn hiện tại

- Trong các kết quả được liệt kê, Gemini Plan-then-code có điểm cao nhất ở cả bốn chỉ số. So với GPT-3.5 Plan-then-code, kết quả này cao hơn 22.0 điểm phần trăm ở Class và 15.5 điểm ở Method.
- Trong ba biến thể local, DeepSeek CoT đạt tỉ lệ hoàn thành toàn bộ cao nhất ở Class (17.0%) và Method (45.6%). DeepSeek Plan-then-code có C.partial cao nhất (28.0%), còn DeepSeek CoT có M.partial cao nhất (67.7%). Few-shot thấp hơn CoT trên cả bốn chỉ số.
- Với SantaCoder, prompt CoT/Few-shot/Plan hiện có điểm hoàn thành toàn bộ bằng 0; Plan cũng đạt 0% partial ở cả lớp lẫn method. So với baseline GPU mới, các prompt dài này không cải thiện kết quả. Đây là kết quả quan sát của checkpoint SantaCoder và cấu hình prompt hiện tại.
- Gemini few-shot có 98 tác vụ vì hai bài được dùng làm ví dụ few-shot không được tính lại như bài kiểm tra.
- Các file Gemini trong nguồn hiện có chỉ cung cấp Pass@1 · greedy; không có Pass@3/Pass@5 cho Gemini.
- DeepSeek-Coder-Instruct 1.3B đã được chạy mới trên GPU với đủ ba prompt style; Few-shot dùng hai ví dụ ClassEval_6 và ClassEval_30 nên chấm 98 tác vụ. Các lượt hoàn tất lần lượt 100/100, 98/98 và 100/100, không có task lỗi.
- Hàng SantaCoder benchmark cũ và hàng baseline GPU mới là hai lượt chạy khác nhau. Các biến thể SantaCoder GPU mới đã hoàn thành đủ 100/98/100 tác vụ tương ứng (Few-shot loại hai ví dụ khỏi chấm điểm); không dùng các số cũ để thay thế kết quả mới.
- Local mới được chạy bằng CUDA trên RTX 3050 Ti. GPT-3.5 và Gemini là kết quả có sẵn từ ClassEval2, không phải lượt chạy local trong project này. Vì một số hàng Few-shot chỉ có 98 tác vụ, nên so sánh trực tiếp có chênh lệch mẫu nhỏ.
- Benchmark vẫn gồm 100 lớp Python và chưa bổ sung đa ngôn ngữ, phụ thuộc nhiều lớp, thư viện ngoài, vòng sửa lỗi, hay khoảng tin cậy thống kê.

## 4. Nguồn dữ liệu trong project

- Chỉ số tổng hợp: `output/result/pass_at_k_result.json`
- JSON DeepSeek-Coder: `output/model_output/deepseek_cot_gpu_20261015.json`, `output/model_output/deepseek_fewshot_gpu_20261015.json`, `output/model_output/deepseek_plan_gpu_20261015.json`
- JSON SantaCoder GPU mới: `output/model_output/santacoder_current_gpu_20261015.json`, `output/model_output/santacoder_cot_gpu_20261015.json`, `output/model_output/santacoder_fewshot_gpu_20261015.json`, `output/model_output/santacoder_plan_gpu_20261015.json`
- JSON Gemini: `output/model_output/gemini_cot_greedy.json`, `output/model_output/gemini_fewshot_greedy.json`, `output/model_output/gemini_plan_greedy.json`
- JSON GPT-3.5 theo prompt mới: `output/model_output/gpt35_cot_greedy.json`, `output/model_output/gpt35_fewshot_greedy.json`, `output/model_output/gpt35_plan_greedy.json`
- Báo cáo đánh giá theo lớp: `output/result/gemini_*_greedy_class_result.json`
- Giao diện so sánh: `local_compare.html` (mở tại `/compare` khi chạy `local_demo.py`)
- Môi trường GPU: `requirements-gpu.txt` (PyTorch 2.5.1 + CUDA 12.4)
