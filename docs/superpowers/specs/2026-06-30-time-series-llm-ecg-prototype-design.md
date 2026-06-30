# Time-Series LLM cho ECG — Thiết kế Prototype Nghiên cứu (4 ngày)

**Ngày:** 2026-06-30
**Đề tài:** Time-Series LLMs — Leveraging LLMs for Multimodal Medical Time Series Analysis and Smart Health Applications
**Phạm vi bản này:** Prototype chứng minh khả thi trong 4 ngày, modality ECG (MIT-BIH).

---

## 1. Mục tiêu & ràng buộc

**Mục tiêu:** Chứng minh một cách trung thực rằng *temporal tokenization* giúp biểu diễn tín hiệu ECG dưới dạng embedding có cấu trúc, và một LLM có thể lý giải (reasoning) trên biểu diễn đó để tạo ra dự đoán rủi ro **có giải thích**.

**Mục tiêu kép (đã chốt):** cân bằng giữa (a) bằng chứng nghiên cứu bằng số liệu và (b) demo trực quan present được.

**Ràng buộc môi trường:**
- Máy: macOS Apple Silicon, **không có GPU NVIDIA**, PyTorch chạy CPU/MPS → chỉ dùng model nhỏ, train ngắn.
- LLM reasoning: **dùng API (Anthropic Claude)**, không tự host LLM lớn.
- Thời gian: 4 ngày, mỗi ngày phải có deliverable tự đứng được.

**Không làm (out of scope, ghi rõ là future work):**
- Không train đa modality thật (wearable/clinical notes) — chỉ trình bày ở mức thiết kế.
- Không nhồi embedding trực tiếp vào không gian token của LLM (bất khả thi với API closed-weights).
- Không nhằm đạt SOTA; nhằm chứng minh khả thi pipeline.

---

## 2. Quyết định thiết kế cốt lõi

| Quyết định | Lựa chọn | Lý do |
|---|---|---|
| LLM backend | Anthropic Claude API | Mac không GPU; nhanh, chất lượng cao, kịp 4 ngày |
| Cầu nối embedding → LLM | **Text-bridge**: tokenizer + classifier → descriptor text → LLM | Trung thực, khả thi, tách bạch baseline vs proposed |
| Modality | Chỉ ECG (MIT-BIH) | Khả thi cao nhất; làm 1 modality cho tốt |
| Dataset | MIT-BIH Arrhythmia qua `wfdb` | Tải tự động, không cần xin quyền (khác MIMIC-IV) |
| Nhãn | Gộp AAMI 5 lớp: N, S, V, F, Q | Chuẩn cộng đồng ECG, giảm mất cân bằng cực đoan |
| Chia dữ liệu | Theo bệnh nhân (inter-patient) | Tránh leakage giữa train/test |

**Cách trả lời Research Questions:**
- **RQ1 (tokenization giúp LLM hiểu tín hiệu?):** classifier trên embedding đạt F1 tốt + t-SNE tách cụm theo nhãn → embedding mã hóa được cấu trúc sinh lý.
- **RQ2 (multimodal alignment?):** trình bày kiến trúc + cài sẵn projection layer (768→dim LLM); train đa modality là future work.
- **RQ3 (LLM reasoning cải thiện tính diễn giải?):** demo sinh giải thích có cấu trúc + kiểm tra faithfulness so với feature thật.

---

## 3. Kiến trúc & luồng dữ liệu

```
ECG beat (MIT-BIH)
   │
   ├─► [Baseline]   1D-CNN backbone ──► Linear head ──► nhãn      (Acc/F1/AUROC)
   │
   └─► [Proposed]   TemporalTokenizer (cùng backbone)
                          │
                          ├─► embedding (B×N×768) ──► t-SNE/PCA viz      (RQ1)
                          ├─► Classifier head      ──► nhãn + confidence
                          └─► Feature extractor (HR, mô tả nhịp, class)
                                       │  + patient context (tuổi, tiền sử)
                                       ▼
                            Prompt template → Claude API ──► risk + lý giải từng bước  (RQ3)
                                       │
                                       ▼
                            Faithfulness check (rule: lý do khớp feature thật)
```

**Nguyên tắc so sánh công bằng:** Baseline và Proposed dùng **chung backbone tokenizer**. Baseline = backbone + linear head train end-to-end. Proposed = cùng embedding + lớp LLM reasoning phía trên. Khác biệt nằm ở phần reasoning/diễn giải, không phải ở năng lực biểu diễn thô.

---

## 4. Các module (đơn vị tách bạch, test độc lập được)

### 4.1 `src/data/loader.py`
- **Làm gì:** tải MIT-BIH (wfdb), phát hiện/đọc R-peak từ annotation, cắt beat (cửa sổ cố định quanh R-peak), map annotation → 5 lớp AAMI, chuẩn hóa biên độ, chia train/test theo bệnh nhân.
- **Interface:** `load_dataset(window, split) -> (X, y, meta)` với `X: N×L`, `y: N`.
- **Phụ thuộc:** `wfdb`, `numpy`.

### 4.2 `src/models/temporal_encoder.py`
- **Làm gì:** `TemporalTokenizer` — 1D-CNN nhận `B×L×1` → `B×N×768`. Có hàm `embed()` trả token và `pooled()` trả vector tổng hợp.
- **Interface:** `forward(x) -> tokens (B×N×768)`.
- **Phụ thuộc:** `torch`.

### 4.3 `src/models/projector.py`
- **Làm gì:** projection `768 → llm_dim` (cài sẵn cho RQ2 design; dùng trong viz/alignment minh họa).
- **Interface:** `forward(emb) -> projected`.

### 4.4 Classifier heads
- **Baseline head + Proposed head** (linear/MLP) trên backbone → logits 5 lớp. Dùng để báo cáo metrics.

### 4.5 `src/models/llm.py`
- **Làm gì:** client Claude API. Nhận descriptor + patient context → trả JSON `{risk, reasons[], confidence}`.
- **Interface:** `explain(descriptor, patient_info) -> dict`.
- **Phụ thuộc:** `anthropic` SDK, env `ANTHROPIC_API_KEY`.
- **Fallback:** nếu không có key → trả reasoning bằng template/rule (pipeline vẫn chạy khi present).
- **Model:** mặc định `claude-haiku-4-5` (rẻ/nhanh); cấu hình đổi sang `claude-sonnet-4-6` nếu cần reasoning sâu.

### 4.6 `src/features.py` (feature extractor)
- **Làm gì:** từ output classifier + tín hiệu → descriptor text có cấu trúc (class dự đoán, confidence, HR ước lượng, mô tả nhịp bất thường).

### 4.7 `src/pipeline.py`
- **Làm gì:** ghép đầu-cuối: `analyze(ecg, patient_info) -> {label, confidence, descriptor, llm_report, faithfulness}`.

### 4.8 `demo/app.py`
- **Làm gì:** Streamlit UI — chọn/upload ECG + nhập tuổi/tiền sử → biểu đồ ECG, nhãn + confidence, báo cáo AI có giải thích.

---

## 5. Kế hoạch 4 ngày (mỗi ngày có deliverable kiểm chứng)

### Ngày 1 — Dữ liệu & baseline
- Scaffold repo (`src/`, `demo/`, `notebooks/`, `requirements.txt`).
- `data/loader.py`: tải MIT-BIH, cắt beat, gộp AAMI, split theo bệnh nhân.
- EDA ngắn: phân bố lớp, vẽ beat mẫu.
- Baseline 1D-CNN + classifier, train ngắn.
- **✅ Kiểm chứng:** in Accuracy/F1/AUROC + confusion matrix lưu file.

### Ngày 2 — Temporal Tokenizer & bằng chứng RQ1
- `temporal_encoder.py` + `projector.py`.
- Trích embedding, vẽ t-SNE/PCA tô màu theo nhãn → `notebooks/ECG_embedding.ipynb`.
- Proposed classifier head, so với baseline.
- **✅ Kiểm chứng:** ảnh t-SNE tách cụm + bảng so sánh baseline vs proposed.

### Ngày 3 — Cầu nối LLM & reasoning (RQ3)
- `features.py` (descriptor) + `llm.py` (Claude client + fallback).
- `pipeline.py` end-to-end.
- Faithfulness check đơn giản (rule).
- **✅ Kiểm chứng:** chạy pipeline 3–5 ca mẫu, in báo cáo + cờ faithfulness.

### Ngày 4 — Demo & đóng gói
- `demo/app.py` Streamlit.
- `README.md`: kiến trúc, cách chạy, kết quả, trả lời RQ1/RQ2/RQ3, hạn chế & future work.
- (Stretch) cache LLM, screenshot cho slide.
- **✅ Kiểm chứng:** demo chạy local đầu-cuối với 1 ca thật.

---

## 6. Đánh giá

**Định lượng (phân loại ECG):** Accuracy, macro-F1, AUROC (one-vs-rest), confusion matrix; baseline vs proposed.

**Định tính (reasoning):** kiểm tra faithfulness — lý do LLM đưa ra có khớp descriptor/feature thật không; tính nhất quán risk level với nhãn dự đoán.

---

## 7. Rủi ro & giảm thiểu

| Rủi ro | Giảm thiểu |
|---|---|
| Train lâu trên CPU/MPS | Giới hạn epoch & số bản ghi bệnh nhân; model nhỏ |
| Thiếu `ANTHROPIC_API_KEY` khi demo | Fallback reasoning bằng template/rule |
| Mất cân bằng lớp (N áp đảo) | Gộp AAMI + macro-F1 + (tùy chọn) class weight |
| Kẹt ở Ngày 3 (LLM) | Sản phẩm Ngày 1–2 vẫn đủ để báo cáo |
| Over-claim multimodal | Ghi rõ RQ2 ở mức thiết kế + future work |

---

## 8. Cấu trúc repo (mục tiêu)

```
Medical-Time-Series-LLM/
  README.md
  requirements.txt
  src/
    data/loader.py
    models/temporal_encoder.py
    models/projector.py
    models/llm.py
    features.py
    pipeline.py
  notebooks/ECG_embedding.ipynb
  demo/app.py
  docs/superpowers/specs/...
```

---

## 9. Stack kỹ thuật
PyTorch (CPU/MPS), `wfdb`, scikit-learn, numpy, matplotlib, Streamlit, `anthropic` SDK. Python 3.x (môi trường hiện tại 3.14 — kiểm tra tương thích wheel của torch khi cài; hạ xuống 3.11/3.12 nếu cần).
