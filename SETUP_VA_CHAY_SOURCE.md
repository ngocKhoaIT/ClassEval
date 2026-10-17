# Hướng dẫn setup và chạy ClassEval

Các lệnh dưới đây dùng **PowerShell trên Windows** và được chạy tại thư mục gốc của repository (nơi có `local_demo.py`, `requirements.txt`). Dự án gồm giao diện demo chạy một bài, script sinh code cho bộ benchmark và script chấm điểm kết quả.

## 1. Chuẩn bị

- Cài Git và **Python 3.10** (phiên bản đang được dùng trong môi trường của dự án). Kiểm tra bằng `py -3.10 --version`.
- Cần kết nối mạng cho lần cài thư viện và lần đầu tải checkpoint từ Hugging Face. Model local cần đủ RAM/VRAM; model lớn có thể không chạy được trên máy thông thường.
- Nếu chạy model local bằng NVIDIA GPU, cần driver hỗ trợ CUDA 12.4. Có thể dùng CPU nhưng sẽ chậm hơn nhiều.

Nếu chưa có source:

```powershell
git clone https://github.com/ngocKhoaIT/ClassEval.git
cd ClassEval
```

Nếu đã có thư mục source này, chỉ cần mở PowerShell tại thư mục gốc của nó.

## 2. Tạo môi trường và cài thư viện

```powershell
py -3.10 -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Với **Windows + NVIDIA GPU**, dùng lệnh sau thay cho lệnh cài `requirements.txt` ở trên (file này đã bao gồm các dependency chung):

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
.\venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Giá trị cuối của lệnh kiểm tra phải là `True` nếu muốn inference dùng GPU. Nếu là `False`, kiểm tra lại driver NVIDIA và môi trường PyTorch. Không cần kích hoạt virtual environment vì mọi lệnh trong tài liệu đều gọi trực tiếp Python của `venv`.

## 3. Cấu hình API key (chỉ khi dùng model qua API)

Model local không cần API key. Với GPT hoặc Gemini, tạo `.env` từ file mẫu rồi điền key tương ứng:

```powershell
Copy-Item .env.example .env
```

Trong `.env`, điền `OPENAI_API_KEY` cho GPT hoặc `GOOGLE_API_KEY` cho Gemini. `OPENAI_BASE_URL` là tùy chọn. Script `generation/inference.py` tự đọc `.env`. Không commit file này lên Git.

## 4. Chạy giao diện demo local

```powershell
.\venv\Scripts\python.exe local_demo.py
```

Mở `http://127.0.0.1:8765/` nếu trình duyệt không tự mở. Chọn bài ClassEval, model, prompt style và nhấn **Chạy model → JSON**. Demo xử lý **một bài mỗi lần**; kết quả lưu ở `output/model_output/`. Lần đầu dùng Hugging Face model, checkpoint được tải về `model_cache/huggingface/`. Dừng server bằng `Ctrl+C`.

Nếu cổng 8765 đang được dùng:

```powershell
.\venv\Scripts\python.exe local_demo.py --port 8766
```

Trang **So sánh chỉ số** đọc dữ liệu đã chấm trong `output/result/pass_at_k_result.json`; chạy demo một bài không tự tạo điểm benchmark.

## 5. Sinh kết quả benchmark bằng dòng lệnh

Ví dụ chạy SantaCoder cho **một bài để thử** (từ thư mục gốc):

```powershell
.\venv\Scripts\python.exe generation/inference.py --model 5 --checkpoint bigcode/santacoder --data_path data/ClassEval_data.json --task_ids ClassEval_0 --output_path output/model_output/santacoder_smoke.json --greedy 1 --generation_strategy 0
```

Để chạy toàn bộ bộ dữ liệu, bỏ `--task_ids` và dùng tên output riêng:

```powershell
.\venv\Scripts\python.exe generation/inference.py --model 5 --checkpoint bigcode/santacoder --data_path data/ClassEval_data.json --output_path output/model_output/santacoder_full.json --greedy 1 --generation_strategy 0
```

Một số tham số chính:

| Tham số | Ý nghĩa |
| --- | --- |
| `--model 5` | SantaCoder; xem `--help` để biết các ID model khác. |
| `--checkpoint` | Hugging Face model ID hoặc đường dẫn checkpoint local. |
| `--generation_strategy` | `0` holistic, `1` incremental, `2` compositional. |
| `--prompt_style` | Với strategy `0`: `0` baseline, `1` CoT, `2` few-shot, `3` plan-then-code. |
| `--greedy` | `1` tạo một mẫu/bài; `0` lấy mẫu (mặc định 5 mẫu/bài). |
| `--task_ids` | Chỉ chạy các bài được chỉ định; bỏ tham số để chạy toàn bộ. |

Ví dụ dùng Gemini qua API, sau khi đã đặt `GOOGLE_API_KEY` trong `.env`:

```powershell
.\venv\Scripts\python.exe generation/inference.py --model 14 --data_path data/ClassEval_data.json --task_ids ClassEval_0 --output_path output/model_output/gemini_smoke.json --greedy 1 --generation_strategy 0 --prompt_style 1 --max_length 8192
```

## 6. Chấm điểm kết quả

Chỉ chấm sau khi đã sinh kết quả cho **toàn bộ các bài**. File cần nằm trong `output/model_output/`. Truyền tên file **không gồm `.json`**:

```powershell
.\venv\Scripts\python.exe classeval_evaluation/evaluation.py --source_file_name santacoder_full --eval_data ClassEval_data --greedy 1
```

Nếu sinh bằng `--greedy 0`, dùng `--greedy 0` khi chấm (script tính Pass@1/3/5 trên 5 mẫu). Kết quả được ghi vào `output/result/`, gồm `pass_at_k_result.json` và file kết quả chi tiết; log nằm trong `log/`. Việc chấm sẽ thực thi code model tạo ra, nên chỉ chấm output từ nguồn bạn tin cậy.

## 7. Các lỗi thường gặp

- **`No module named ...`**: kiểm tra đang dùng `.\venv\Scripts\python.exe` và đã cài đúng requirements.
- **CUDA không khả dụng**: chạy lại lệnh kiểm tra PyTorch ở mục 2; script có thể dùng CPU khi không truyền `--cuda`, nhưng chậm với model local.
- **Thiếu API key**: kiểm tra `.env` ở thư mục gốc và tên biến `OPENAI_API_KEY` / `GOOGLE_API_KEY`.
- **Không tải được model**: kiểm tra mạng, model ID/quyền truy cập trên Hugging Face, dung lượng ổ đĩa và RAM/VRAM.

Chi tiết về tham số và chiến lược sinh code xem [generation/README.md](generation/README.md); cách chấm điểm xem [classeval_evaluation/README.md](classeval_evaluation/README.md).
