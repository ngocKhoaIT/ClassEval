# Báo cáo cải tiến ClassEval và so sánh model

## 1. Những nội dung đã cải tiến

### Hướng nghiên cứu

Phần đánh giá tập trung vào hai hạn chế đã nêu trong phần tổng quan:

- **Prompt chưa được khảo sát sâu:** bổ sung và thử ba cách tạo mã ở chế độ sinh cả lớp: Chain of Thought (CoT), few-shot và Plan-then-code. Kết quả có các lần chạy GPT-3.5 và Gemini theo ba chiến lược này.
- **Model trong bài báo đã cũ:** đưa kết quả Gemini mới hơn vào cùng nguồn benchmark để đối chiếu với GPT-3.5 và model local. Kết quả GPT đời mới ngoài GPT-3.5 chưa có trong dữ liệu hiện tại; cần bổ sung sau khi xác định model và hoàn thành lần chạy.

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

Bảng dưới đây dùng **Pass@1 · greedy**. “Class” và “Method” là tỉ lệ đạt toàn bộ kiểm thử ở mức lớp và phương thức; “C.partial” và “M.partial” là tỉ lệ đạt một phần. Các hàng GPT-3.5 baseline là số từ bảng bạn cung cấp về bài báo gốc; các hàng còn lại lấy từ nguồn kết quả tổng hợp trong project.

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
| SantaCoder · baseline H, greedy | 10.0% / 27.3% | 0.0% / 0.2% | 1.0% / 4.2% | 100 |
| SantaCoder + CoT | — | 0.0% / 0.0% | 0.0% / 1.4% | 100 |
| SantaCoder + Few-shot | — | 0.0% / 0.0% | 0.0% / 1.0% | 98 |
| SantaCoder + Plan-then-code | — | 0.0% / 0.0% | 0.0% / 0.0% | 100 |
| DeepSeek-Coder-Instruct 1.3B + CoT | — | 17.0% / 45.6% | 27.0% / 67.7% | 100 |
| DeepSeek-Coder-Instruct 1.3B + Few-shot | — | 14.3% / 41.5% | 25.5% / 63.4% | 98 |
| DeepSeek-Coder-Instruct 1.3B + Plan-then-code | — | 16.0% / 40.6% | 28.0% / 59.2% | 100 |

Số SantaCoder trong bài báo là Pass@1 greedy tốt nhất trong ba chiến lược sinh; bài báo chỉ công bố full success Class/Method cho biểu đồ này, không có C.partial/M.partial tương ứng. Baseline mới của SantaCoder là Holistic trên CUDA. Vì chiến lược sinh giữa số “tốt nhất trong bài” và baseline mới không được giữ cố định, chênh lệch chỉ có ý nghĩa tham khảo, không phải đối chứng cùng cấu hình. DeepSeek-Coder-Instruct 1.3B không có trong bài báo gốc nên không có số cũ để điền.

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
- Báo cáo đánh giá theo lớp: `output/result/gemini_*_greedy_class_result.json`
- Giao diện so sánh: `local_compare.html` (mở tại `/compare` khi chạy `local_demo.py`)
- Môi trường GPU: `requirements-gpu.txt` (PyTorch 2.5.1 + CUDA 12.4)
