# Kịch bản trình bày 15 phút — Time-Series LLM for Medical Data

**Mục tiêu:** Thuyết phục giảng viên rằng bạn *thực sự* muốn tham gia nghiên cứu đề
tài này — bằng cách cho thấy bạn đã (1) hiểu bài toán, (2) tự xây prototype chạy
được, (3) tư duy như một người làm nghiên cứu (trung thực với hạn chế, có hướng đi
tiếp).

**Chuẩn bị trước khi trình bày (làm 1 lần, ~5 phút trước buổi gặp):**
```bash
cd /Users/sangphan/Research/Medical-Time-Series-LLM
source .venv/bin/activate
python -m src.train          # tạo metrics.json, confusion_matrix.png, checkpoint.pt
python -m src.embed_viz      # tạo tsne.png
# mở sẵn 2 ảnh để khỏi chờ khi present:
open artifacts/confusion_matrix.png artifacts/tsne.png
# mở sẵn demo ở tab khác:
streamlit run demo/app.py    # http://localhost:8501
```
> Mở sẵn: README.md, ảnh t-SNE, demo Streamlit, và terminal. Tránh để giảng viên chờ.

---

## Tổng quan thời lượng

| Phần | Thời gian | Nội dung |
|---|---|---|
| 1 | 0:00–2:00 | Động lực: vì sao đề tài quan trọng |
| 2 | 2:00–4:00 | Bài toán + 3 câu hỏi nghiên cứu |
| 3 | 4:00–6:30 | Kiến trúc giải pháp |
| 4 | 6:30–9:30 | **Demo trực tiếp** |
| 5 | 9:30–11:30 | Kết quả & cách trả lời RQ |
| 6 | 11:30–13:00 | Quyết định nghiên cứu & tính trung thực |
| 7 | 13:00–14:30 | Hạn chế & hướng nghiên cứu tiếp |
| 8 | 14:30–15:00 | Vì sao tôi muốn tham gia |

---

## Phần 1 — Động lực (0:00–2:00)

**Mục tiêu:** Cho thấy bạn hiểu *bức tranh lớn*, không chỉ code.

**Lời nói gợi ý:**
> "Dữ liệu y tế ngày nay phần lớn là *chuỗi thời gian*: ECG, tín hiệu từ thiết bị
> đeo, dấu hiệu sinh tồn theo thời gian. Các mô hình truyền thống (CNN, RNN) phân
> loại tốt nhưng cho ra một con số — bác sĩ khó tin và khó dùng. Trong khi đó, LLM
> rất mạnh ở *suy luận và giải thích bằng ngôn ngữ*, nhưng lại không 'đọc' được tín
> hiệu sinh lý thô. Câu hỏi của em là: **làm sao bắc cầu giữa hai thế giới đó** — để
> có một hệ thống vừa dự đoán, vừa *giải thích được* cho con người. Đó chính là tinh
> thần của đề tài Time-Series LLM cho y tế."

**Mẹo:** Nhấn vào từ "giải thích được" (interpretability) — đây là giá trị cốt lõi
khác biệt so với mô hình cũ.

---

## Phần 2 — Bài toán & câu hỏi nghiên cứu (2:00–4:00)

**Mục tiêu:** Khung hóa vấn đề thành câu hỏi nghiên cứu cụ thể.

**Mở README.md, chỉ vào mục Research Questions.** Nói:
> "Em thu hẹp đề tài thành một prototype chứng minh khả thi trên modality ECG, với 3
> câu hỏi:
> - **RQ1:** Temporal tokenization có giúp biểu diễn tín hiệu thành embedding *có cấu
>   trúc* để mô hình hiểu được không?
> - **RQ2:** Làm sao gióng (align) nhiều modality về cùng một không gian biểu diễn?
> - **RQ3:** Suy luận bằng LLM có cải thiện *tính diễn giải* của dự đoán không?
>
> Em chọn ECG + bộ dữ liệu chuẩn MIT-BIH vì nó công khai, có nhãn rối loạn nhịp, và
> là 'ngôn ngữ chung' của giới nghiên cứu ECG."

---

## Phần 3 — Kiến trúc giải pháp (4:00–6:30)

**Mục tiêu:** Trình bày pipeline mạch lạc.

**Chỉ vào sơ đồ Architecture trong README.** Đi theo luồng:
> "Luồng gồm 5 bước:
> 1. **Temporal Tokenizer** (1D-CNN) cắt beat ECG thành embedding 768 chiều.
> 2. **Classifier head** phân loại nhịp N/S/V — đây là phần cho ra *số liệu*.
> 3. **Feature extractor** chuyển kết quả thành mô tả văn bản (loại beat, độ tin cậy,
>    nhịp).
> 4. **LLM** (OpenAI hoặc Claude) nhận mô tả đó + bối cảnh bệnh nhân → sinh đánh giá
>    rủi ro *có lý giải từng bước*.
> 5. **Faithfulness check** kiểm tra lý giải có bám vào dữ kiện thật không."

**Điểm cần làm rõ (quan trọng về mặt học thuật):**
> "Em chọn cách *bắc cầu bằng văn bản* (text-bridge) một cách có chủ đích. Vì LLM qua
> API là closed-weights, ta không thể nhồi embedding trực tiếp vào không gian token
> của nó. Nên thay vì 'giả vờ' làm điều bất khả thi, em chuyển embedding → mô tả văn
> bản để LLM suy luận. Đây là lựa chọn trung thực và đúng với giới hạn công cụ."

---

## Phần 4 — Demo trực tiếp (6:30–9:30)

**Mục tiêu:** Cho thấy nó *chạy thật*, không phải slide.

**Bước 1 — Demo Streamlit (đã mở sẵn ở http://localhost:8501):**
- Nhập tuổi 70, tiền sử "Hypertension".
- Chọn 1 beat từ record mẫu → chỉ biểu đồ ECG.
- Bấm **Analyze** → đọc to kết quả: *"Mô hình đoán beat V, độ tin cậy X, LLM đánh giá
  risk High với các lý do: ... và cờ faithfulness = True."*

**Lời nói:**
> "Điểm em muốn nhấn: đầu ra không chỉ là nhãn, mà là một *báo cáo có lý giải* —
> đúng cái mà bác sĩ cần để tin tưởng."

**Bước 2 — Chạy pipeline trên 1 beat V thật qua terminal (cho thấy không 'dàn dựng'):**
```bash
python -c "
from src.data import loader
from src import pipeline, config
import numpy as np
X, y = loader.load_split(['200'], max_beats_per_record=None)
i = int(np.where(y==2)[0][0])
r = pipeline.analyze(X[i], {'age':70,'history':'Hypertension'})
print('Nhãn thật:', config.CLASSES[y[i]], '| Dự đoán:', r['label'], '| Conf: %.2f'%r['confidence'])
print('Risk:', r['llm_report']['risk'])
print('Lý giải:', r['llm_report']['reasons'])
print('Faithful:', r['faithfulness']['faithful'])
"
```
> "Đây là beat Ventricular *thật* từ dữ liệu, mô hình đoán đúng, và lý giải bám sát
> dữ kiện."

**Dự phòng:** Nếu mạng/API trục trặc, nhấn mạnh: *"Hệ thống có fallback nội bộ nên
vẫn chạy offline — em thiết kế để buổi demo không bao giờ 'chết'."*

---

## Phần 5 — Kết quả & trả lời RQ (9:30–11:30)

**Mục tiêu:** Số liệu + bằng chứng trực quan.

**Mở artifacts/tsne.png và confusion_matrix.png.** Nói:
> "Về **RQ1**: trên tập test, mô hình đạt **macro-AUROC 0.91** và **macro-F1 0.53**.
> Quan trọng hơn con số, hãy nhìn t-SNE: các beat Ventricular (xanh lá) và
> Supraventricular (cam) gom thành *cụm riêng*, tách khỏi khối Normal. Điều này chứng
> minh embedding đã *mã hóa được cấu trúc sinh lý* — trả lời 'có' cho RQ1.
>
> Về **RQ3**: phần lý giải + faithfulness check cho thấy LLM tạo ra giải thích bám
> dữ kiện, cải thiện tính diễn giải so với một nhãn trần."

**Nếu được hỏi về RQ2:** xem Phần 7 (hướng tương lai).

---

## Phần 6 — Quyết định nghiên cứu & tính trung thực (11:30–13:00)

**Mục tiêu:** Đây là phần "ăn điểm" nhất — cho thấy bạn tư duy như nhà nghiên cứu.

**Lời nói (kể câu chuyện cái 'pivot'):**
> "Em muốn kể một quyết định quan trọng. Ban đầu em làm phân loại 5 lớp AAMI. Kết quả
> accuracy tới 0.90 — nghe rất đẹp. Nhưng khi xem kỹ, **macro-F1 chỉ 0.20, đúng bằng
> mức ngẫu nhiên**. Lý do: dữ liệu cực mất cân bằng — lớp Normal chiếm 99%, hai lớp
> hiếm gần như không có mẫu test. Mô hình chỉ việc luôn đoán 'Normal' là đã có
> accuracy cao giả tạo.
>
> Em đã *không* giấu con số đẹp đó. Em chuyển sang bài toán chuẩn 3 lớp N/S/V, dùng
> cân bằng lớp khi train và đánh giá bằng macro-F1/AUROC — những chỉ số *không bị
> đánh lừa* bởi mất cân bằng. Accuracy giảm xuống 0.69, nhưng giờ nó *thật*.
>
> Em nghĩ điều này quan trọng: trong y tế, một con số đẹp nhưng sai lệch còn nguy
> hiểm hơn là không có. Em muốn làm nghiên cứu theo tinh thần đó."

**Mẹo:** Đây là điểm khác biệt giữa "sinh viên làm bài tập" và "người làm nghiên
cứu". Hãy kể tự tin.

---

## Phần 7 — Hạn chế & hướng nghiên cứu tiếp (13:00–14:30)

**Mục tiêu:** Cho thấy bạn biết prototype này *chưa* phải đích đến.

> "Em rất rõ các hạn chế, và mỗi hạn chế là một hướng nghiên cứu:
> - **Mới 1 modality (ECG).** → RQ2 thật sự: huấn luyện *alignment đa modality* (ECG
>   + thiết bị đeo + bệnh án văn bản). Em đã cài sẵn lớp projection làm nền.
> - **Text-bridge thay vì nhồi embedding.** → Hướng tiếp: dùng LLM mã nguồn mở nhỏ +
>   huấn luyện *soft-prompt adapter* để đưa embedding trực tiếp vào LLM.
> - **Heart rate hiện là ước lượng thô** (1 beat không suy ra được nhịp). → Cần dùng
>   chuỗi nhiều beat / RR interval.
> - **Lớp Supraventricular khó nhất** → cần dữ liệu/augmentation tốt hơn.
> - **Chưa có giá trị lâm sàng được kiểm chứng** → cần hợp tác với chuyên gia y tế."

**Lời nói chuyển tiếp:**
> "Nếu được tham gia nhóm, hướng đầu tiên em muốn theo đuổi là **RQ2 — alignment đa
> modality thật sự**, vì đó là phần mới và có giá trị khoa học cao nhất."

---

## Phần 8 — Vì sao tôi muốn tham gia (14:30–15:00)

**Lời kết (chân thành, ngắn gọn):**
> "Em làm prototype này không phải để 'khoe code', mà để chứng minh em *đã bắt tay
> vào* đề tài, hiểu được cả phần hay lẫn phần khó, và có hướng đi rõ ràng. Em thực sự
> hứng thú với giao điểm giữa mô hình chuỗi thời gian và suy luận của LLM trong y tế,
> và em mong được thầy/cô hướng dẫn để đi sâu hơn, đặc biệt là phần đa modality. Em
> sẵn sàng cam kết thời gian và học những gì còn thiếu."

---

## Phụ lục — Câu hỏi giảng viên có thể hỏi & cách trả lời

| Câu hỏi | Hướng trả lời |
|---|---|
| "Vì sao macro-F1 chỉ 0.53?" | Lớp S rất khó (hình thái tinh tế, ít mẫu) kéo trung bình xuống; nhưng AUROC 0.91 cho thấy khả năng phân tách tốt. Đây là kết quả *thật* trên split inter-patient (khó hơn intra-patient). |
| "Sao không dùng MIMIC-IV / đa modality?" | MIMIC-IV cần xin quyền truy cập (credentialing), không kịp trong prototype ngắn. Em chọn làm 1 modality cho chắc, và để đa modality làm hướng nghiên cứu chính (RQ2). |
| "LLM có thể 'bịa' lý giải không?" | Đúng, đó là rủi ro. Em thêm *faithfulness check* dạng rule để phát hiện lý giải không bám dữ kiện — và đây cũng là một hướng nghiên cứu (đánh giá faithfulness nghiêm túc hơn). |
| "Inter-patient split là gì, vì sao quan trọng?" | Không để cùng một bệnh nhân ở cả train và test (de Chazal DS1/DS2). Tránh mô hình 'học thuộc' đặc trưng cá nhân → đánh giá sát thực tế hơn. |
| "Embedding có thực sự nhồi vào LLM không?" | Không, vì API là closed-weights. Em dùng text-bridge một cách trung thực; nhồi embedding cần LLM mã nguồn mở + adapter — là future work em muốn làm. |
| "Đóng góp mới của em là gì?" | Prototype tích hợp đầu-cuối + cơ chế faithfulness + khung đánh giá trung thực cho dữ liệu mất cân bằng. Đóng góp lớn hơn sẽ đến từ RQ2 (alignment đa modality). |

---

## Checklist phút chót
- [ ] `python -m src.train` đã chạy, có `artifacts/metrics.json`
- [ ] `artifacts/tsne.png` + `confusion_matrix.png` mở sẵn
- [ ] Streamlit chạy ở `http://localhost:8501`
- [ ] Có key trong `.env` (hoặc chấp nhận fallback)
- [ ] README mở ở mục Research Questions + Architecture
- [ ] Tập nói phần 6 (câu chuyện pivot) cho trôi chảy
