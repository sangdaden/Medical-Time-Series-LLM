# Mở rộng: Multimodal Alignment + Forecasting — Thiết kế

**Ngày:** 2026-06-30
**Bối cảnh:** Mở rộng prototype ECG để chạm các từ khóa "multimodal / cross-modal /
forecasting" trong mô tả đề tài, một cách trung thực (dữ liệu thật, không bịa).

## Mục tiêu
1. **Multimodal alignment (RQ2 thật):** thêm modality thứ hai là chuỗi RR/HR-trend
   trích thật từ ECG; gióng 2 modality về không gian chung và fuse; chứng minh
   cross-modal cải thiện so với đơn modality.
2. **Forecasting:** từ cửa sổ N beat liên tiếp, dự báo có beat bất thường (S/V) trong
   M beat kế tiếp (binary, đánh giá bằng AUROC).
3. Cập nhật README + bảng đối chiếu mô tả.

## Phase 1 — Multimodal

### Modality #2: RR/HR-trend
- **RR interval** = (sample_R[i] − sample_R[i−1]) / 360 giây. HR = 60/RR bpm.
- Với mỗi beat *được giữ* (đã map AAMI, không sát mép), lấy vector **K=8 khoảng RR
  gần nhất** (pad ở đầu bản ghi). Đây là tín hiệu dạng thiết bị đeo (nhịp theo thời gian).

### Kiến trúc
```
ECG beat (432)  ──► TemporalTokenizer ──► pooled 768 ──► proj_ecg ─┐
                                                                   ├─► fuse ─► classifier ─► N/S/V
RR-trend (K=8)  ──► RREncoder (MLP)   ──► 64        ──► proj_rr  ──┘
```
- `proj_ecg`, `proj_rr`: chiếu mỗi modality về **shared dim = 128** (đây là phần
  "alignment").
- `fuse`: ghép (concat) 2 vector chung → MLP → logits 3 lớp.
- **Bằng chứng cross-modal:** so macro-F1/AUROC của ECG-only vs ECG+RR.

### Module
- `src/data/loader.py`: thêm `load_split_multimodal(records)` → (ecg N×432, rr N×K, y N).
- `src/models/rr_encoder.py`: `RREncoder` (MLP K→64).
- `src/models/multimodal.py`: `MultimodalClassifier` (ecg backbone + rr encoder +
  projections + fusion + head); hỗ trợ chế độ `use_rr=False` để chạy baseline đơn modality.
- `src/train_multimodal.py`: train cả 2 cấu hình, ghi `artifacts/multimodal_metrics.json`.

## Phase 2 — Forecasting

### Bài toán
- Sắp beat theo thứ tự thời gian trong từng bản ghi.
- Mẫu: cửa sổ **N=10 beat** liên tiếp → nhãn = 1 nếu có ≥1 beat S/V trong **M=5 beat
  kế tiếp**, ngược lại 0.
- Đặc trưng mỗi beat trong cửa sổ: RR interval + xác suất bất thường từ classifier
  (1 − P(N)) — gọn, nhanh.

### Kiến trúc
```
[N beat features]  ──► GRU ──► hidden ──► Linear ──► P(abnormal sắp tới)
```
- `src/forecast.py`: dựng cửa sổ (theo bản ghi, không leak), `RiskForecaster` (GRU),
  train + eval AUROC, ghi `artifacts/forecast_metrics.json`.
- Inter-patient: dùng DS1 train / DS2 test như phần chính.

## Phase 3 — Docs
- Cập nhật README: thêm mục Multimodal + Forecasting, cập nhật bảng RQ và alignment.
- Cập nhật bảng đối chiếu mô tả trong tài liệu.

## Phạm vi & trung thực
- RR-trend là **dữ liệu thật** trích từ ECG (không synthetic).
- Clinical-notes/MIMIC vẫn ngoài phạm vi (ghi rõ future work).
- Forecasting ở mức beat-stream nội bộ một bản ghi (không phải dự báo bệnh dài hạn).
