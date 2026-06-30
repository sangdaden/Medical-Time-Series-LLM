# Kịch bản trình bày 15 phút — Time-Series LLM for Medical Data (v2)

**Mục tiêu:** Thuyết phục giảng viên rằng bạn *thực sự* muốn tham gia nghiên cứu — bằng
cách cho thấy bạn (1) hiểu bài toán, (2) đã tự xây một ***framework** mở rộng được* chạm
đúng các từ khóa của đề tài (**a new framework**, multimodal, cross-modal, clinical
reasoning, forecasting), và (3) tư duy như người làm nghiên cứu (trung thực với hạn
chế, có hướng đi tiếp).

> **Câu một dòng để mở đầu/khẳng định:** "Em xây một *framework* adapt LLM cho dữ liệu
> y tế đa modality — đăng ký modality là dùng được, hiện có 3 nhóm (sinh lý / thiết bị
> đeo / hồ sơ lâm sàng) — kèm reasoning có giải thích và dự báo rủi ro."

**Chuẩn bị trước buổi gặp (~10 phút trước):**
```bash
cd /Users/sangphan/Research/Medical-Time-Series-LLM
source .venv/bin/activate
python -m src.train             # metrics.json, confusion_matrix.png, checkpoint.pt
python -m src.embed_viz         # tsne.png
python -m src.train_multimodal  # multimodal_metrics.json  (ECG-only vs ECG+RR)
python -m src.forecast          # forecast_metrics.json
open artifacts/confusion_matrix.png artifacts/tsne.png
streamlit run demo/app.py       # http://localhost:8501  (mở tab riêng)
```
> Mở sẵn: README.md (mục *Coverage vs project description* + *Results*), 2 ảnh, demo,
> và terminal. Tránh để giảng viên chờ.

---

## Tổng quan thời lượng

| Phần | Thời gian | Nội dung |
|---|---|---|
| 1 | 0:00–2:00 | Động lực |
| 2 | 2:00–3:30 | Bài toán + 3 câu hỏi nghiên cứu |
| 3 | 3:30–5:30 | Kiến trúc (2 modality + reasoning + forecasting) |
| 4 | 5:30–8:00 | **Demo trực tiếp** |
| 5 | 8:00–9:30 | Kết quả RQ1 (tokenization + t-SNE) |
| 6 | 9:30–11:00 | **Multimodal: cross-modal có lợi** ⭐ |
| 7 | 11:00–12:00 | **Forecasting** |
| 8 | 12:00–13:00 | Pivot story (tính trung thực) |
| 9 | 13:00–14:00 | Đối chiếu mô tả + hạn chế + tương lai |
| 10 | 14:00–15:00 | Vì sao tôi muốn tham gia + Q&A |

---

## Phần 1 — Động lực (0:00–2:00)

> "Dữ liệu y tế phần lớn là *chuỗi thời gian*: ECG, tín hiệu thiết bị đeo, dấu hiệu
> sinh tồn. Mô hình truyền thống phân loại tốt nhưng chỉ cho *một con số* — bác sĩ khó
> tin, khó dùng. LLM thì mạnh ở *suy luận và giải thích bằng ngôn ngữ* nhưng không
> 'đọc' được tín hiệu sinh lý thô. Câu hỏi của em: **làm sao bắc cầu hai thế giới đó**
> để có hệ thống vừa dự đoán, vừa *giải thích được*, và biết kết hợp nhiều nguồn dữ
> liệu. Đó là tinh thần đề tài Time-Series LLM cho y tế."

---

## Phần 2 — Bài toán & câu hỏi nghiên cứu (2:00–3:30)

**Mở README → mục Research questions.**
> "Em hiện thực hóa trên ECG (bộ chuẩn MIT-BIH) với 3 câu hỏi:
> - **RQ1:** Temporal tokenization có cho embedding *có cấu trúc* để máy hiểu tín hiệu?
> - **RQ2:** Làm sao *gióng nhiều modality* về một không gian biểu diễn chung?
> - **RQ3:** Suy luận bằng LLM có tăng *tính diễn giải* không?
> Và em làm thêm một bài toán *forecasting* để hướng tới giám sát rủi ro theo thời gian."

---

## Phần 3 — Kiến trúc: đây là một *framework* (3:30–5:30)

**Mở đầu — nhấn mạnh đây là framework, không phải script một lần:**
> "Điểm cốt lõi em muốn nhấn: em không xây một pipeline cứng, mà một **framework mở
> rộng được**. Mỗi *modality* = một bộ trích đặc trưng + một encoder, được **đăng ký**
> vào một registry; loader và mô hình tự động dùng nó. **Thêm modality mới chỉ ~10
> dòng, không sửa code lõi.** Đây đúng là tinh thần 'a new framework' trong đề tài."

**Chỉ sơ đồ Architecture + mục *Extending the framework* trong README.**
> "Hiện framework đăng ký sẵn **3 modality thật — đúng 3 nhóm trong mô tả đề tài**:
> 1. **physiological — `ecg`:** Temporal Tokenizer (1D-CNN) → embedding 768 chiều.
> 2. **wearable — `rr`:** chuỗi khoảng RR (nhịp tim theo thời gian) trích *thật* từ ECG.
> 3. **clinical records — `clinical`:** tuổi/giới/số thuốc đọc *thật* từ header MIT-BIH.
>
> **Multimodal alignment:** mỗi modality được *chiếu về không gian chung 128 chiều* rồi
> *fuse* — phần alignment của RQ2; mô hình tổng quát cho **N modality** bất kỳ.
> Sau đó: **Classifier** (N/S/V) → **LLM (OpenAI/Claude)** sinh *risk + lý giải* kèm
> **faithfulness check** → và một **Forecaster (GRU)** dự báo beat bất thường sắp tới."

**Điểm học thuật cần nói rõ (text-bridge):**
> "LLM qua API là closed-weights nên không thể nhồi embedding trực tiếp vào nó. Em
> chọn *text-bridge* — chuyển embedding thành mô tả văn bản — một cách trung thực với
> giới hạn công cụ; nhồi embedding thật là hướng tương lai."

---

## Phần 4 — Demo trực tiếp (5:30–8:00)

**Streamlit (đã mở sẵn):** nhập tuổi 70, tiền sử "Hypertension", chọn 1 beat → bấm
**Analyze** → đọc to: *"Mô hình đoán beat V, confidence X, LLM đánh giá risk High với
các lý do ..., faithfulness = True."*
> "Đầu ra không chỉ là nhãn, mà là *báo cáo có lý giải* — đúng cái bác sĩ cần để tin."

**Terminal — beat V thật (chứng minh không dàn dựng):**
```bash
python -c "
from src.data import loader
from src import pipeline, config
import numpy as np
X, y = loader.load_split(['200'], max_beats_per_record=None)
i = int(np.where(y==2)[0][0])
r = pipeline.analyze(X[i], {'age':70,'history':'Hypertension'})
print('Nhãn thật:', config.CLASSES[y[i]], '| Dự đoán:', r['label'], '| Risk:', r['llm_report']['risk'])
print('Lý giải:', r['llm_report']['reasons'])
"
```
**Dự phòng:** nếu mạng/API lỗi → *"Hệ thống có fallback nội bộ, vẫn chạy offline — em
thiết kế để buổi demo không bao giờ chết."*

---

## Phần 5 — Kết quả RQ1 (8:00–9:30)

**Mở artifacts/tsne.png + confusion_matrix.png.**
> "RQ1: mô hình đạt **macro-AUROC 0.91**. Quan trọng hơn con số — nhìn t-SNE: beat
> Ventricular (xanh lá) và Supraventricular (cam) gom thành *cụm riêng*, tách khỏi
> khối Normal. Tức embedding đã *mã hóa được cấu trúc sinh lý* → trả lời 'có' cho RQ1."

---

## Phần 6 — Framework đo đóng góp từng modality (9:30–11:00) ⭐

**Đây là điểm mạnh nhất — chạm đúng 'multimodal / cross-modal' VÀ chứng minh đây là framework.**
**Mở README → bảng Multimodal (hoặc artifacts/multimodal_metrics.json).**
> "Vì là framework, em *đo được* đóng góp từng modality: huấn luyện cùng dữ liệu cho
> từng tổ hợp tích lũy:
>
> | Tổ hợp | Macro-F1 | AUROC |
> |---|---|---|
> | ecg | 0.50 | 0.87 |
> | **ecg + rr** | **0.67** | **0.95** |
> | ecg + rr + clinical | 0.60 | 0.94 |
>
> Hai điều đáng nói:
> 1. **RR cho lợi ích cross-modal lớn (+0.17 macro-F1)** — RR mang thông tin *thời
>    điểm* (beat sớm = RR ngắn) mà một beat đơn lẻ không có. Đây là bằng chứng
>    cross-modal thật sự.
> 2. **Clinical (tuổi/giới/thuốc) *không* cải thiện phân loại beat** — em báo cáo
>    trung thực kết quả âm này: feature tĩnh theo bệnh nhân nên không phân biệt beat;
>    nó hữu ích hơn ở tầng *reasoning của LLM*. Việc *đo được* điều này chính là giá
>    trị của framework — và cho thấy em đánh giá khách quan, không tô hồng."

---

## Phần 7 — Forecasting (11:00–12:00)

**Mở artifacts/forecast_metrics.json.**
> "Em làm thêm bài toán *dự báo rủi ro*: từ 10 nhịp RR gần nhất, một mô hình GRU dự báo
> *có beat bất thường trong 5 beat kế tiếp không*. Trên split inter-patient, đạt
> **AUROC 0.85**. Tức nhịp tim gần đây *báo trước* được biến cố loạn nhịp — bước đầu
> hướng tới giám sát rủi ro bệnh nhân theo thời gian."

---

## Phần 8 — Pivot story: tính trung thực (12:00–13:00)

> "Em muốn kể một quyết định. Ban đầu em phân loại 5 lớp, accuracy 0.90 — nghe rất đẹp.
> Nhưng macro-F1 chỉ **0.20, đúng mức ngẫu nhiên**: dữ liệu mất cân bằng, lớp Normal
> chiếm 99%, mô hình chỉ việc luôn đoán 'Normal'. Em *không* giữ con số đẹp giả tạo đó.
> Em chuyển sang bài toán chuẩn 3 lớp, cân bằng lớp, đánh giá bằng macro-F1/AUROC —
> những chỉ số không bị đánh lừa. Accuracy giảm còn 0.69 nhưng *thật*. Trong y tế, một
> con số đẹp nhưng sai lệch nguy hiểm hơn không có. Em muốn làm nghiên cứu theo tinh
> thần đó."

---

## Phần 9 — Đối chiếu mô tả + hạn chế + tương lai (13:00–14:00)

**Mở README → bảng *Coverage vs project description*.**
> "Em tự đối chiếu prototype với mô tả đề tài: phần *lõi* đã có — temporal
> tokenization, multimodal alignment, cross-modal, clinical reasoning, forecasting.
> Còn lại 2 mục em xác định là *hướng nghiên cứu tiếp theo*:
> - **Thêm modality văn bản lâm sàng độc lập** (vd MIMIC-IV) — cần xin quyền dữ liệu.
> - **Nhồi embedding trực tiếp vào LLM mã nguồn mở** (soft-prompt adapter).
>
> Nếu được tham gia nhóm, hướng đầu tiên em muốn theo đuổi là mở rộng multimodal sang
> dữ liệu lâm sàng thật — phần có giá trị khoa học cao nhất."

---

## Phần 10 — Vì sao tôi muốn tham gia (14:00–15:00)

> "Em làm prototype này không phải để khoe code, mà để chứng minh em *đã bắt tay vào*
> đề tài, hiểu cả phần hay lẫn phần khó, và có hướng đi rõ ràng. Em thực sự hứng thú
> với giao điểm giữa mô hình chuỗi thời gian và suy luận LLM trong y tế, đặc biệt là
> phần đa modality. Em mong được thầy/cô hướng dẫn để đi sâu hơn, và sẵn sàng cam kết
> thời gian cũng như học những gì còn thiếu."

---

## Phụ lục — Câu hỏi giảng viên & cách trả lời

| Câu hỏi | Hướng trả lời |
|---|---|
| "Sao gọi là *framework*, khác gì một script?" | Có lớp trừu tượng `ModalitySpec` + registry: mỗi modality = bộ trích đặc trưng + encoder, *đăng ký* là loader và mô hình tự dùng. Thêm modality mới ~10 dòng, không sửa code lõi. Mô hình tổng quát cho N modality. README có ví dụ thêm modality `spo2`. |
| "Multimodal của em có thật không hay ghép cho có?" | Thật: cùng dữ liệu, thêm RR-trend → macro-F1 +0.17. RR trích thật từ ECG; clinical (tuổi/giới/thuốc) đọc thật từ header MIT-BIH. |
| "Sao thêm clinical lại không cải thiện?" | Trung thực: feature tĩnh theo bệnh nhân nên không phân biệt beat → không giúp phân loại beat (thậm chí giảm nhẹ). Nó hữu ích ở tầng reasoning LLM. Framework cho phép *đo* được điều này. |
| "Vì sao ECG+RR tốt hơn?" | RR mang thông tin *thời điểm* (beat sớm = RR ngắn) mà 1 beat waveform không có; đặc biệt giúp lớp S (vốn về timing). |
| "Forecasting có bị rò rỉ nhãn không?" | Không — đầu vào chỉ là RR của các beat *quá khứ*; nhãn là sự kiện *tương lai*. Đánh giá trên split inter-patient. |
| "Macro-F1 0.53 (đơn modality) thấp?" | Lớp S khó (ít mẫu, hình thái tinh tế); nhưng AUROC 0.91 cho thấy phân tách tốt. Và multimodal nâng F1 lên 0.74. |
| "Sao chưa dùng MIMIC / clinical notes?" | Cần credentialing, ngoài phạm vi prototype ngắn; là hướng nghiên cứu chính tiếp theo. |
| "Embedding có thật sự vào LLM không?" | Không, vì API closed-weights; em dùng text-bridge trung thực, nhồi embedding cần LLM mã nguồn mở — future work. |
| "Đóng góp mới?" | Pipeline tích hợp đầu-cuối + alignment đa modality có đo cross-modal gain + forecasting + faithfulness + khung đánh giá trung thực cho dữ liệu mất cân bằng. |

---

## Checklist phút chót
- [ ] 4 script đã chạy: `train`, `embed_viz`, `train_multimodal`, `forecast`
- [ ] 4 file JSON trong `artifacts/` + 2 ảnh mở sẵn
- [ ] Streamlit chạy ở `http://localhost:8501`
- [ ] Key trong `.env` (hoặc chấp nhận fallback)
- [ ] README mở ở *Coverage vs project description* + *Results*
- [ ] Tập nói Phần 6 (multimodal) và Phần 8 (pivot) cho trôi chảy
