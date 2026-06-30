# Giải thích chi tiết cách làm + Bảng thuật ngữ

Tài liệu này mô tả chi tiết toàn bộ cách làm của dự án **Time-Series LLM for
Multimodal Medical Signals** (modality hiện có: ECG + RR/HR-trend), giải thích từng
bước "làm như thế nào" và giải nghĩa mọi từ viết tắt / thuật ngữ.

---

## Phần A — Ý tưởng tổng thể (1 câu)

Biến tín hiệu **ECG** thô (một chuỗi số theo thời gian) thành một dạng biểu diễn mà
máy "hiểu" được, dùng nó để **phân loại nhịp tim**, rồi để một **LLM** đọc kết quả
đó và **giải thích bằng ngôn ngữ** mức độ rủi ro cho bệnh nhân.

- **ECG** (*Electrocardiogram* — điện tâm đồ): đồ thị ghi hoạt động điện của tim theo
  thời gian. Mỗi nhịp tim tạo ra một "sóng" đặc trưng.
- **LLM** (*Large Language Model* — mô hình ngôn ngữ lớn): mô hình AI như GPT/Claude,
  giỏi đọc-hiểu và suy luận bằng văn bản.

---

## Phần B — Pipeline 5 bước (làm như thế nào)

> **Pipeline**: chuỗi các bước xử lý nối tiếp, đầu ra bước này là đầu vào bước sau.

### Bước 0 — Chuẩn bị dữ liệu (tiền xử lý)

**Làm gì:** Tải dữ liệu ECG, cắt nó thành từng nhịp riêng lẻ, gán nhãn.

**Chi tiết cách làm:**
1. Tải bộ **MIT-BIH** từ **PhysioNet** bằng thư viện `wfdb`.
   - **MIT-BIH**: bộ dữ liệu ECG chuẩn do MIT (*Massachusetts Institute of
     Technology*) và Beth Israel Hospital tạo.
   - **PhysioNet**: kho dữ liệu y sinh công khai trên mạng.
   - **wfdb** (*WaveForm DataBase*): thư viện Python để đọc dữ liệu tín hiệu y tế.
2. Mỗi bản ghi có **annotation** (chú thích) đánh dấu vị trí **R-peak** và loại nhịp.
   - **R-peak**: đỉnh nhọn cao nhất của một nhịp tim trên ECG (đỉnh của phức bộ
     **QRS**). Nó đánh dấu "tâm" của mỗi nhịp.
   - **QRS complex** (phức bộ QRS): cụm 3 sóng Q-R-S thể hiện tâm thất co bóp — phần
     dễ nhận nhất của mỗi nhịp.
3. **Cắt beat (segmentation):** quanh mỗi R-peak, lấy cửa sổ cố định **432 mẫu** (180
   mẫu trước + 252 mẫu sau). Với tần số **360 Hz** (*Hertz* — số mẫu/giây), 432 mẫu ≈
   1.2 giây — đủ chứa trọn một nhịp.
4. **Z-normalization** (chuẩn hóa z): với mỗi bản ghi, trừ trung bình và chia độ lệch
   chuẩn → đưa biên độ về cùng thang, để mô hình không bị lệch vì máy đo khác nhau.
   - Công thức: `(giá trị − trung bình) / độ lệch chuẩn`.
5. **Gán nhãn theo chuẩn AAMI**, gộp về 3 lớp **N / S / V**:
   - **AAMI** (*Association for the Advancement of Medical Instrumentation*): tổ chức
     đặt chuẩn nhóm các loại nhịp ECG.
   - **N** = Normal (nhịp bình thường)
   - **S** = Supraventricular ectopic (nhịp ngoại lai *trên thất* — phát sinh phía
     trên tâm thất)
   - **V** = Ventricular ectopic (nhịp ngoại lai *tại thất* — nguy hiểm hơn)

### Bước 1 — Temporal Tokenizer (mã hóa tín hiệu thành embedding)

**Làm gì:** Biến một beat (432 con số) thành một vector đặc trưng giàu thông tin.

- **Temporal** = thuộc về thời gian (tín hiệu theo thời gian).
- **Tokenizer**: bộ "token hóa" — chuyển dữ liệu thô thành các đơn vị biểu diễn
  (token) mà mô hình xử lý được. (Với chữ, token là từ; ở đây token là đoạn tín hiệu
  được mã hóa.)
- **Embedding** (vector nhúng): một dãy số (ở đây **768 chiều**) biểu diễn cô đọng nội
  dung của dữ liệu. Hai beat giống nhau → embedding gần nhau trong không gian số.

**Cách làm:** Dùng **1D-CNN**.
- **CNN** (*Convolutional Neural Network* — mạng nơ-ron tích chập): loại mạng học cách
  phát hiện các "mẫu hình cục bộ" (như hình dạng sóng).
- **1D** (*one-dimensional*): vì ECG là tín hiệu 1 chiều theo thời gian (khác ảnh là 2D).
- **Convolution (tích chập)**: trượt một "bộ lọc" nhỏ dọc tín hiệu để dò các đặc trưng
  (sóng nhọn, sóng rộng…). Mạng tự học bộ lọc nào hữu ích.

Kết quả: mỗi beat → một chuỗi token, rồi **mean-pooling** (lấy trung bình) thành 1
vector 768 chiều đại diện cả beat.
- **Pooling**: gộp nhiều vector thành một (mean-pooling = lấy trung bình).

### Bước 2 — Classifier head (phân loại nhịp)

**Làm gì:** Từ embedding 768 chiều, đoán beat thuộc lớp N, S hay V.

- **Classifier** (bộ phân loại): mô hình gán nhãn cho đầu vào.
- **Head** (đầu ra): lớp nhỏ gắn lên trên phần "thân" (backbone) để làm nhiệm vụ cụ thể.
- **Backbone** (xương sống): phần chính tạo embedding — ở đây là Temporal Tokenizer.
- Lớp này là **Linear layer** (lớp tuyến tính) + **softmax**:
  - **Linear layer**: phép biến đổi `output = W·x + b` (nhân ma trận trọng số W rồi
    cộng b).
  - **softmax**: hàm biến các điểm số thô (**logits**) thành xác suất cộng lại bằng 1.
    Lớp có xác suất cao nhất là dự đoán; xác suất đó là **confidence** (độ tin cậy).
  - **logits**: điểm số thô trước khi chuyển thành xác suất.

### Bước 3 — Feature extractor (embedding → mô tả văn bản)

**Làm gì:** Chuyển kết quả số thành một câu mô tả để LLM đọc được.

Ví dụ tạo ra: *"ECG analysis: predicted beat type = Ventricular ectopic beat (V),
confidence = 0.76. Estimated heart rate = ... bpm, rhythm appears irregular."*
- **bpm** (*beats per minute* — nhịp/phút): đơn vị nhịp tim.
- Đây là **text-bridge** (cầu nối bằng văn bản) — xem giải thích quan trọng ở Phần D.

### Bước 4 — LLM reasoning (suy luận & giải thích)

**Làm gì:** LLM nhận mô tả + bối cảnh bệnh nhân (tuổi, tiền sử) → sinh ra đánh giá rủi
ro kèm lý do từng bước.

- Gọi qua **API** (*Application Programming Interface* — giao diện lập trình để gọi
  dịch vụ từ xa) của **OpenAI** (model `gpt-4o-mini`) hoặc **Anthropic** (model Claude).
- Đầu ra dạng **JSON** (*JavaScript Object Notation* — định dạng dữ liệu có cấu trúc):
  `{"risk": "High", "reasons": [...], "confidence": ...}`.
- Có **fallback** (phương án dự phòng): nếu không có **API key** (chuỗi bí mật để xác
  thực khi gọi API), hệ thống dùng quy tắc nội bộ để vẫn sinh lý giải → demo không bao
  giờ chết.

### Bước 5 — Faithfulness check (kiểm tra tính trung thực của lời giải thích)

**Làm gì:** Kiểm tra lý do LLM đưa ra có *bám vào dữ kiện thật* không (tránh LLM "bịa").
- **Faithfulness** (tính trung thành): lời giải thích có phản ánh đúng cái mô hình thực
  sự thấy không.
- Cách làm hiện tại: quy tắc đơn giản — kiểm tra lý giải có nhắc đến đúng loại nhịp đã
  dự đoán không.

---

## Phần C — Huấn luyện & đánh giá (làm như thế nào)

### Huấn luyện (training)
- **Training**: quá trình cho mô hình xem dữ liệu có nhãn để nó tự điều chỉnh **trọng
  số** (weights — các con số bên trong mạng) sao cho đoán đúng hơn.
- **Epoch**: một lượt mô hình xem hết toàn bộ dữ liệu huấn luyện (ta chạy 8 epoch).
- **Batch**: một nhóm nhỏ mẫu xử lý cùng lúc (ta dùng 256).
- **Loss** (hàm mất mát): con số đo "mức sai" của mô hình; huấn luyện = giảm dần loss.
  Ta dùng **Cross-Entropy Loss** (mất mát entropy chéo — chuẩn cho bài toán phân loại).
- **Optimizer Adam**: thuật toán điều chỉnh trọng số để giảm loss (Adam là biến thể phổ
  biến, hội tụ nhanh).
- **Learning rate** (tốc độ học): mỗi bước điều chỉnh trọng số lớn hay nhỏ (0.001).
- **Class balancing (cân bằng lớp):** vì lớp N chiếm ~90%, ta lấy mẫu cân bằng (mỗi lớp
  tối đa 2000 beat khi train) + **class weights** (trọng số lớp — phạt nặng hơn khi
  đoán sai lớp hiếm). Nếu không, mô hình chỉ việc luôn đoán "N".
- **Inter-patient split (chia theo bệnh nhân):** không để cùng một bệnh nhân ở cả tập
  train lẫn test.
  - Dùng **de Chazal DS1/DS2** — cách chia chuẩn (DS = *DataSet*; DS1 để train, DS2 để
    test) do nhà nghiên cứu de Chazal đề xuất.
  - Tránh **data leakage** (rò rỉ dữ liệu): nếu cùng bệnh nhân ở 2 bên, mô hình "học
    thuộc" đặc điểm cá nhân → điểm ảo cao.
- **Checkpoint**: file lưu trọng số đã huấn luyện (`artifacts/checkpoint.pt`) để dùng
  lại không cần train lại.

### Đánh giá (evaluation) — các chỉ số
- **Accuracy** (độ chính xác): tỉ lệ đoán đúng / tổng số. *Nhược điểm:* bị đánh lừa khi
  mất cân bằng (luôn đoán N cũng được ~90%).
- **F1-score**: trung bình điều hòa của **Precision** và **Recall**:
  - **Precision** (độ chính xác dương): trong những ca *đoán* là lớp X, bao nhiêu % đúng.
  - **Recall** (độ bao phủ): trong những ca *thật sự* là lớp X, bắt được bao nhiêu %.
  - **Macro-F1**: tính F1 cho từng lớp rồi lấy **trung bình không trọng số** (macro =
    mỗi lớp quan trọng như nhau, dù hiếm). Vì thế nó *không* bị lớp N áp đảo → phản ánh
    trung thực.
- **AUROC** (*Area Under the Receiver Operating Characteristic curve* — diện tích dưới
  đường cong ROC): đo khả năng *phân tách* các lớp, từ 0.5 (đoán mò) đến 1.0 (hoàn
  hảo). Ta đạt **0.91** → phân tách rất tốt.
  - **OvR** (*One-vs-Rest*): tính AUROC cho từng lớp so với phần còn lại, rồi trung bình.
- **Confusion matrix** (ma trận nhầm lẫn): bảng cho thấy lớp thật vs lớp đoán — nhìn ra
  mô hình hay nhầm lớp nào với lớp nào.

### Trực quan hóa embedding (bằng chứng RQ1)
- **t-SNE** (*t-distributed Stochastic Neighbor Embedding*): kỹ thuật ép vector nhiều
  chiều (768) xuống 2 chiều để vẽ lên mặt phẳng, giữ cấu trúc "gần nhau". Nếu các lớp
  tạo cụm riêng → embedding đã học được cấu trúc.
- **PCA** (*Principal Component Analysis* — phân tích thành phần chính): kỹ thuật giảm
  chiều tuyến tính (ở đây dùng để khởi tạo t-SNE).
- **RQ** (*Research Question* — câu hỏi nghiên cứu): RQ1 (tokenization có giúp hiểu tín
  hiệu?), RQ2 (gióng đa modality?), RQ3 (LLM có tăng tính diễn giải?).

---

## Phần D — Một số điểm thiết kế quan trọng

**Text-bridge (cầu nối bằng văn bản):** Vì LLM gọi qua API là *closed-weights* (không
truy cập được trọng số bên trong), ta **không thể** nhồi embedding 768 chiều trực tiếp
vào không gian token của LLM. Thay vì giả vờ làm điều bất khả thi, ta chuyển embedding
→ mô tả văn bản để LLM suy luận. Đây là lựa chọn trung thực với giới hạn công cụ. Việc
nhồi embedding thật sự cần một LLM *mã nguồn mở* nhỏ + huấn luyện *soft-prompt adapter*
— là hướng nghiên cứu tương lai.

**Projector (lớp chiếu):** `src/models/projector.py` chiếu embedding 768 → 4096 chiều,
hướng tới việc nhồi embedding vào không gian token của LLM (future work). RQ2 (gióng đa
modality) đã được hiện thực hóa thật bằng *framework đa modality* mô tả ở Phần E.

---

## Phần E — Framework đa modality & forecasting (vì sao đây là "a framework")

Mô tả đề tài nói "develop ... **a new framework**". Điểm mấu chốt: dự án không phải một
pipeline cứng, mà là một **framework mở rộng được**.

**Khái niệm "modality" (phương thức dữ liệu):** mỗi modality = (1) bộ *trích đặc trưng
theo từng beat* + (2) một *encoder* biến đặc trưng đó thành embedding. Mỗi modality được
**đăng ký (register)** vào một *registry* (sổ đăng ký). Bộ nạp dữ liệu và mô hình chỉ
việc duyệt registry — nên **thêm một modality mới chỉ ~10 dòng, không sửa code lõi**.
- **Registry:** danh sách trung tâm ánh xạ tên modality → cách trích đặc trưng + encoder.
- File: `src/framework.py` (lõi registry), `src/modalities.py` (các modality cụ thể).

**3 modality thật, đúng 3 nhóm trong mô tả:**
| Nhóm trong mô tả | Tên | Dữ liệu |
|---|---|---|
| physiological signal (tín hiệu sinh lý) | `ecg` | sóng ECG (1D-CNN) |
| wearable sensor data (thiết bị đeo) | `rr` | chuỗi RR/nhịp tim trích từ ECG |
| clinical records (hồ sơ lâm sàng) | `clinical` | tuổi, giới, số thuốc (đọc thật từ header MIT-BIH) |

**Multimodal alignment (gióng đa modality):** mỗi modality được *chiếu (project)* về một
không gian chung 128 chiều, rồi *fuse (hợp nhất)* để phân loại. Mô hình tổng quát cho
**N modality** bất kỳ trong danh sách `config.MODALITIES`.

**Cross-modal (liên modality):** framework cho phép *đo* đóng góp từng modality. Kết quả
(macro-F1): `ecg` 0.50 → `ecg+rr` 0.67 (+0.17, lợi ích cross-modal rõ) → `ecg+rr+clinical`
0.60 (clinical *không* giúp phân loại beat — kết quả âm trung thực, vì feature tĩnh theo
bệnh nhân; nó hữu ích ở tầng reasoning của LLM).

**Forecasting (dự báo):** ngoài phân loại, một mô hình **GRU** (*Gated Recurrent Unit* —
mạng hồi tiếp xử lý chuỗi) nhận 10 khoảng RR gần nhất để dự báo *có beat bất thường (S/V)
trong 5 beat kế tiếp không* (AUROC ~0.85). File: `src/forecast.py`.

**Cách thêm modality mới (điểm mở rộng):** viết `prepare` + `beat_feature` + chọn
encoder, gọi `register(ModalitySpec(...))`, thêm tên vào `config.MODALITIES` — xong.

---

## Phần F — Bảng tra cứu thuật ngữ & viết tắt

| Viết tắt / Thuật ngữ | Nghĩa đầy đủ | Giải thích ngắn |
|---|---|---|
| ECG / EKG | Electrocardiogram | Điện tâm đồ — đồ thị điện hoạt động của tim |
| LLM | Large Language Model | Mô hình ngôn ngữ lớn (GPT, Claude…) |
| MIT-BIH | MIT–Beth Israel Hospital DB | Bộ dữ liệu ECG chuẩn về rối loạn nhịp |
| PhysioNet | — | Kho dữ liệu y sinh công khai |
| wfdb | WaveForm DataBase | Thư viện Python đọc tín hiệu y tế |
| AAMI | Assoc. for the Advancement of Medical Instrumentation | Tổ chức đặt chuẩn nhóm nhịp |
| N / S / V | Normal / Supraventricular / Ventricular | 3 lớp nhịp ta phân loại |
| R-peak | — | Đỉnh nhọn cao nhất của một nhịp |
| QRS | — | Phức bộ sóng thể hiện tâm thất co bóp |
| RR interval | — | Khoảng cách giữa 2 R-peak (suy ra nhịp tim) |
| MLII, V5 | — | Tên 2 kênh (đạo trình) đo ECG |
| Hz | Hertz | Số mẫu mỗi giây (360 Hz) |
| bpm | beats per minute | Nhịp tim/phút |
| Embedding | Vector nhúng | Dãy số biểu diễn cô đọng dữ liệu (768 chiều) |
| Tokenizer | — | Bộ chuyển dữ liệu thô thành token |
| CNN / 1D-CNN | (1D) Convolutional Neural Network | Mạng tích chập dò mẫu hình cục bộ |
| Convolution | Tích chập | Trượt bộ lọc dọc tín hiệu để dò đặc trưng |
| Pooling | — | Gộp nhiều vector (mean-pooling = trung bình) |
| Backbone / Head | Xương sống / Đầu | Phần tạo embedding / lớp làm nhiệm vụ cụ thể |
| Linear layer | Lớp tuyến tính | Phép `W·x + b` |
| Logits | — | Điểm số thô trước softmax |
| Softmax | — | Biến logits thành xác suất (tổng = 1) |
| Confidence | Độ tin cậy | Xác suất của lớp được đoán |
| Projector | — | Lớp chiếu embedding 768 → 4096 chiều (stub cho RQ2) |
| Text-bridge | Cầu nối văn bản | Chuyển kết quả số thành mô tả chữ cho LLM đọc |
| API | Application Programming Interface | Giao diện gọi dịch vụ từ xa (OpenAI/Anthropic) |
| API key | — | Chuỗi bí mật xác thực khi gọi API |
| JSON | JavaScript Object Notation | Định dạng dữ liệu có cấu trúc |
| Fallback | Phương án dự phòng | Dùng quy tắc nội bộ khi không có API key |
| Faithfulness | Tính trung thành | Lời giải thích có bám dữ kiện thật không |
| Training / Epoch / Batch | Huấn luyện / Lượt / Lô | Cho mô hình học; 1 lượt hết dữ liệu; nhóm mẫu |
| Loss / Cross-Entropy | Hàm mất mát | Đo mức sai; Cross-Entropy cho phân loại |
| Adam / Learning rate | Optimizer / Tốc độ học | Thuật toán & độ lớn mỗi bước cập nhật trọng số |
| Weights | Trọng số | Các số bên trong mạng, được học |
| Class balancing / weights | Cân bằng / Trọng số lớp | Xử lý mất cân bằng (N ~90%) |
| Inter-patient | Chia theo bệnh nhân | Không trộn 1 bệnh nhân ở train & test |
| de Chazal DS1/DS2 | DataSet 1 / 2 | Cách chia train/test chuẩn |
| Data leakage | Rò rỉ dữ liệu | Thông tin test lọt vào train → điểm ảo |
| Checkpoint | — | File lưu trọng số đã train |
| Accuracy | Độ chính xác | % đoán đúng (dễ bị đánh lừa khi mất cân bằng) |
| Precision / Recall | Độ chính xác dương / Độ bao phủ | Đoán-đúng-trong-số-đoán / bắt-được-trong-số-thật |
| F1 / Macro-F1 | — | Cân bằng Precision & Recall; macro = mỗi lớp như nhau |
| AUROC / OvR | Area Under ROC / One-vs-Rest | Khả năng phân tách lớp (0.5–1.0); từng-lớp-vs-còn-lại |
| Confusion matrix | Ma trận nhầm lẫn | Bảng lớp thật vs lớp đoán |
| t-SNE / PCA | — | Kỹ thuật giảm chiều để vẽ embedding |
| MPS | Metal Performance Shaders | Tăng tốc GPU của Apple (máy Mac) |
| RQ | Research Question | Câu hỏi nghiên cứu (RQ1/2/3) |
| Framework | Khung mở rộng | Hệ thống có điểm mở rộng rõ ràng (đăng ký là dùng được), không phải script cứng |
| Registry | Sổ đăng ký | Danh sách trung tâm ánh xạ tên modality → cách trích đặc trưng + encoder |
| Modality | Phương thức dữ liệu | Một nguồn dữ liệu (ECG / RR / clinical) + cách mã hoá nó |
| ModalitySpec | — | Khai báo 1 modality (tên, trích đặc trưng, encoder) để đăng ký |
| Fusion | Hợp nhất | Ghép embedding nhiều modality (concat) trước khi phân loại |
| Projection | Phép chiếu | Đưa embedding các modality về cùng một chiều (128) để gióng |
| Cross-modal | Liên modality | Mẫu hình học được nhờ *kết hợp* nhiều modality |
| GRU | Gated Recurrent Unit | Mạng hồi tiếp xử lý chuỗi, dùng cho forecasting |
