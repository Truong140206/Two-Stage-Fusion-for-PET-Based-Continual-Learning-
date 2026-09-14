# Đặc tả đầy đủ để vẽ lại pipeline

Tài liệu này mô tả đúng pipeline của bản thảo hiện tại. Hình mới phải cho thấy
hai pha chính:

1. học mô-đun PET và cập nhật đầu chiếu ngẫu nhiên sau mỗi tác vụ;
2. suy luận hai tầng bằng cùng một vector điểm RP.

Thông điệp trung tâm của hình:

> Một đầu RP cố định tạo ra vector điểm lớp \(s\), sau đó cùng \(s\) được tái sử
> dụng để hỗ trợ cả định tuyến tác vụ và phân lớp cuối.

## 1. Ký hiệu

| Ký hiệu | Ý nghĩa |
|---|---|
| \(x,y\) | ảnh và nhãn của mẫu |
| \(\mathcal D_t\) | dữ liệu huấn luyện của tác vụ hiện tại \(t\) |
| \(p_t\) | LoRA của tác vụ \(t\) |
| \(p_1\) | LoRA của tác vụ đầu tiên; chỉ số 0 trong mã |
| \(g_\omega\) | đầu TII phụ dùng để đề xuất task ban đầu |
| \(g\) | đầu phân lớp dùng chung của nhánh PET |
| \(u\) | điểm TII trần |
| \(t_0\) | task ban đầu suy ra từ \(u\) |
| \(r\) | logit từ lượt truyền với \(p_{t_0}\) |
| \(\ell\) | logit được giữ lại sau DRM/CRM |
| \(f\) | đặc trưng 768 chiều từ ViT với \(p_1\) |
| \(W\) | ma trận chiếu ngẫu nhiên cố định |
| \(h\) | đặc trưng mở rộng 10.000 chiều sau ReLU |
| \(G,C\) | thống kê cộng dồn dùng cho nghiệm ridge |
| \(W_{\mathrm{out}}\) | trọng số đầu RP, cập nhật bằng nghiệm giải tích |
| \(s\) | điểm lớp của đầu RP |
| \(w\) | trọng số nguồn PET tại tầng hợp nhất task |
| \(\beta_i\) | trọng số RP theo từng mẫu tại tầng hợp nhất lớp |

## 2. Panel A — học PET và cập nhật đầu RP

~~~mermaid
flowchart LR
    D["Dữ liệu tác vụ hiện tại D_t"] --> B["ViT-B/16<br/>FROZEN"]
    B --> PET["LoRA p_t + đầu g, g_ω<br/>TRAINABLE bằng gradient"]
    PET --> FZ["Đóng băng sau khi học xong tác vụ"]

    D --> P1["ViT + LoRA p_1<br/>FROZEN"]
    P1 --> F["Đặc trưng f ∈ R^768"]
    F --> RP["h = ReLU(Wᵀf)<br/>W ∈ R^(768×10000), FROZEN"]
    RP --> ST["Cập nhật giải tích<br/>G ← G + hhᵀ<br/>C ← C + he_yᵀ"]
    ST --> RIDGE["Giải ridge<br/>(G + λI)W_out = C"]
    RIDGE --> WO["W_out<br/>ANALYTIC, không gradient"]
~~~

### 2.1. Học mô-đun PET

Với dữ liệu tác vụ hiện tại \(\mathcal D_t\):

- backbone ViT-B/16 được đóng băng;
- LoRA \(p_t\), đầu dùng chung \(g\) và đầu TII \(g_\omega\) được huấn luyện
  bằng gradient;
- các adapter cũ \(p_{<t}\) được giữ nguyên;
- sau khi học xong PET, các mô-đun này được đóng băng trước khi khớp RP.

### 2.2. Trích xuất đặc trưng cố định cho RP

Mỗi ảnh huấn luyện hiện tại được chạy qua:

\[
x\longrightarrow \text{ViT}+p_1
\longrightarrow f\in\mathbb R^{768}.
\]

Phải dùng adapter của tác vụ đầu tiên \(p_1\), không dùng adapter hiện tại
\(p_t\). Việc dùng cùng \(p_1\) giữ không gian đặc trưng nhất quán qua các tác
vụ.

### 2.3. Chiếu ngẫu nhiên

\[
h=\operatorname{ReLU}(W^\top f),\qquad
W\in\mathbb R^{768\times10^4}.
\]

- \(W\) được sinh một lần từ phân phối Gaussian với seed 1993;
- \(W\) luôn được đóng băng;
- kích thước mở rộng \(M=10^4\);
- đặc trưng \(f\) không được chuẩn hóa bổ sung trước phép chiếu.

### 2.4. Cập nhật thống kê và nghiệm ridge

Với từng mẫu \((x,y)\):

\[
G\leftarrow G+hh^\top,
\qquad
C\leftarrow C+h\mathbf e_y^\top.
\]

Sau khi xử lý dữ liệu của tác vụ hiện tại:

\[
(G+\lambda I)W_{\mathrm{out}}=C,
\qquad \lambda=10^4.
\]

Đây là cập nhật giải tích:

- không backpropagation;
- không lưu ảnh cũ;
- không lưu đặc trưng của từng mẫu;
- chỉ cộng dồn \(G,C\) và giải lại \(W_{\mathrm{out}}\);
- \(G,C\) chứa đóng góp của mọi tác vụ đã quan sát.

## 3. Panel B — suy luận hai tầng

~~~mermaid
flowchart LR
    X["Ảnh kiểm thử x<br/>không biết task ID"]

    X --> BARE["Bare ViT + g_ω<br/>FROZEN"]
    BARE --> U["Điểm TII u"]
    U --> T0["t₀ = task(argmax u)"]
    T0 --> PASS0["ViT + adapter p_t₀ + g"]
    PASS0 --> R["Logit ban đầu r"]

    X --> P1["ViT + adapter p₁<br/>một lượt truyền thêm"]
    P1 --> F["f ∈ R^768"]
    F --> H["h = ReLU(Wᵀf)"]
    H --> S["s = W_outᵀh<br/>điểm lớp RP"]

    R --> RT["Max điểm lớp theo từng task<br/>r_task"]
    S --> ST["Max điểm lớp theo từng task<br/>s_task"]
    RT --> ZR["Chuẩn hóa z"]
    ST --> ZS["Chuẩn hóa z"]
    ZR --> MIX1["Tầng 1<br/>hợp nhất task theo w"]
    ZS --> MIX1
    MIX1 --> TDRM["Task đề xuất t_DRM"]
    TDRM --> DRM["Conditional adapter pass<br/>DRM"]
    R -. "Entropy và lớp đứng thứ hai" .-> CRM["CRM kế thừa"]
    DRM --> CRM
    CRM --> L["Logit sau re-matching ℓ"]

    L --> GATE["Softmax ℓ<br/>biên top-1/top-2 → βᵢ"]
    L --> ZL["Chuẩn hóa z(ℓ)"]
    S --> ZS2["Chuẩn hóa z(s)"]
    GATE --> MIX2["Tầng 2<br/>hợp nhất lớp có cổng"]
    ZL --> MIX2
    ZS2 --> MIX2
    MIX2 --> RESCALE["Đưa về thang của ℓ"]
    RESCALE --> Y["ŷ = argmax ℓ̂"]
~~~

Toàn bộ mô-đun trong panel suy luận đều được giữ cố định. Không có cập nhật
gradient hoặc dùng nhãn kiểm thử.

## 4. Nhánh PET chính

### 4.1. Đề xuất task ban đầu bằng TII

\[
u=g_\omega(\phi_{\mathrm{bare}}(x)),
\qquad
t_0=\mathcal T\left(\arg\max_c u_c\right).
\]

\(u\) là điểm từ đầu TII phụ. Nó chỉ tạo task ban đầu \(t_0\), không phải nguồn
điểm PET được trộn ở Tầng 1.

### 4.2. Lượt truyền thích nghi ban đầu

\[
r=g(\phi(x;p_{t_0})).
\]

\(r\) là logit lớp dưới adapter được TII chọn và là nguồn PET của Tầng 1.

## 5. Nhánh RP cố định

Cùng ảnh \(x\) được chạy thêm một lượt với adapter \(p_1\):

\[
f=\phi(x;p_1),
\qquad
h=\operatorname{ReLU}(W^\top f),
\qquad
s=W_{\mathrm{out}}^\top h.
\]

\(s\) là một vector điểm trên toàn bộ lớp đã thấy. Hình phải cho \(s\) tách
thành đúng hai nhánh:

1. lên Tầng 1 để hỗ trợ đề xuất task;
2. sang Tầng 2 để hỗ trợ dự đoán lớp cuối.

## 6. Tầng 1 — hợp nhất ở cấp task

Đưa điểm lớp về điểm task bằng phép lấy cực đại:

\[
r_t^{\mathrm{task}}
=\max_{c\in\mathcal C_t}r_c,
\qquad
s_t^{\mathrm{task}}
=\max_{c\in\mathcal C_t}s_c.
\]

Chỉ task và lớp đã thấy được tham gia.

Chuẩn hóa độc lập từng nguồn:

\[
z(v)=
\frac{v-\operatorname{mean}(v)}
{\max(\operatorname{std}(v),10^{-6})}.
\]

Sau đó hợp nhất:

\[
q_t=
w\,z(r_t^{\mathrm{task}})
+(1-w)\,z(s_t^{\mathrm{task}}),
\]

\[
\hat t_{\mathrm{DRM}}=\arg\max_t q_t.
\]

Cấu hình báo cáo dùng \(w=0.7\): 70% nguồn PET và 30% nguồn RP. \(w\) là trọng
số chung được cố định trước, không được mô tả là giá trị tối ưu.

Nếu \(\hat t_{\mathrm{DRM}}\ne t_0\), hệ thống chạy adapter do DRM đề xuất.
Nếu hai task trùng nhau, kết quả lượt truyền ban đầu có thể được tái sử dụng.

## 7. DRM và CRM kế thừa

Đây là phần kế thừa từ pipeline PET, không phải mô-đun mới:

- Tầng 1 chỉ thay đổi đề xuất task đưa vào DRM;
- DRM thử adapter được đề xuất khi cần;
- CRM dùng generalized-entropy confidence của \(r\);
- ứng viên CRM là task chứa lớp đứng thứ hai của \(r\);
- các logit ứng viên được giữ/chọn theo log-sum-exp;
- CRM không chạy ở tác vụ 1.

Logit được giữ lại sau DRM/CRM được ký hiệu là \(\ell\).

## 8. Tầng 2 — hợp nhất ở cấp lớp

Tầng 2 nhận:

- logit chính sau re-matching \(\ell\);
- điểm lớp RP \(s\).

### 8.1. Cổng biên xác suất

Tính softmax của \(\ell\), rồi lấy hai xác suất lớn nhất
\(p_{(1)}\ge p_{(2)}\):

\[
\beta_i=
\beta\left(
1-\frac{p_{(1)}-p_{(2)}}{p_{(1)}+p_{(2)}}
\right),
\qquad \beta=0.5.
\]

- nếu dự đoán chính rất tự tin, \(\beta_i\to0\);
- nếu hai lớp đầu gần nhau, \(\beta_i\to\beta\);
- đây là cổng liên tục, không phải quyết định nhị phân;
- cổng được tính từ \(\ell\), không được tính từ \(r\) hoặc \(s\).

### 8.2. Hợp nhất và trả về thang logit chính

\[
m_i=(1-\beta_i)z(\ell_i)+\beta_i z(s_i),
\]

\[
\hat\ell_i=
m_i\,\sigma_\epsilon(\ell_i)
+\operatorname{mean}(\ell_i).
\]

Dự đoán cuối:

\[
\hat y=\arg\max_c\hat\ell_{i,c}.
\]

Khối “Rescale” chỉ đưa hỗn hợp về thang đo của \(\ell\); không được mô tả như
một bước hiệu chuẩn xác suất.

## 9. Trạng thái mô-đun và quy ước hình

| Trạng thái | Thành phần |
|---|---|
| **F — Frozen** | backbone, adapter cũ, \(p_1\), \(W\), toàn bộ mô-đun khi suy luận |
| **T — Trainable** | \(p_t,g,g_\omega\) khi học PET |
| **A — Analytic** | cập nhật \(G,C\) và giải \(W_{\mathrm{out}}\) |

Quy ước màu đề xuất:

- xanh dương: backbone và nhánh PET cố định;
- cam: thành phần trainable bằng gradient;
- xanh lá: đặc trưng và thống kê RP;
- tím: đầu RP và cập nhật giải tích;
- đỏ hồng: hai tầng hợp nhất.

Quy ước mũi tên:

- nét liền: luồng ảnh, đặc trưng hoặc logit;
- nét đứt: nhãn huấn luyện hoặc tín hiệu điều khiển;
- một nút rẽ rõ ràng tại \(s\), không kéo dây xuyên qua các hộp;
- mũi tên từ \(\ell\) tới cổng \(\beta_i\) phải tách khỏi luồng điểm lớp.

## 10. Bố cục hình khuyến nghị

### Panel (a): Update on task \(t\)

- phía trên: học PET bằng gradient;
- phía dưới: dùng \(\mathcal D_t\) để cập nhật \(G,C,W_{\mathrm{out}}\);
- đặt nhãn “current-task data only” và “no replay”;
- đặt nhãn “no gradient” ngay dưới khối ridge.

### Panel (b): Inference

- nhánh PET chính nằm ở hàng trên;
- nhánh RP cố định nằm ở hàng dưới;
- Tầng 1 nằm sau \(r\) và trước DRM;
- DRM/CRM nằm giữa hai tầng;
- Tầng 2 nằm sát đầu ra;
- vector \(s\) rẽ lên Tầng 1 và sang Tầng 2;
- đặt callout “one extra \(p_1\) forward, shared by both stages”.

## 11. Checklist chống vẽ sai

- [ ] Backbone luôn được đánh dấu frozen.
- [ ] Chỉ \(p_t,g,g_\omega\) được train bằng gradient.
- [ ] \(p_1\) được dùng ở mọi tác vụ để tạo đặc trưng RP.
- [ ] \(W\) là random và frozen.
- [ ] \(W_{\mathrm{out}}\) được giải ridge, không train bằng gradient.
- [ ] \(u\) chỉ sinh \(t_0\).
- [ ] Tầng 1 trộn \(r^{\mathrm{task}}\) với \(s^{\mathrm{task}}\).
- [ ] Tầng 1 nằm trước DRM.
- [ ] CRM nhận tín hiệu điều khiển từ \(r\).
- [ ] Tầng 2 dùng \(\ell\) sau DRM/CRM, không dùng \(r\).
- [ ] Cổng \(\beta_i\) được tính từ top-2 softmax của \(\ell\).
- [ ] Cùng một \(s\) được tái sử dụng ở cả hai tầng.
- [ ] Không có test-time learning.
- [ ] Không có replay buffer hoặc lưu ảnh cũ.
- [ ] Chỉ lớp và task đã thấy tham gia chuẩn hóa, max và argmax.

## 12. Caption gợi ý

**Vòng đời tham số và hợp nhất hai tầng.** Trên tác vụ hiện tại, pipeline PET
kế thừa chỉ huấn luyện adapter và các đầu dự đoán, trong khi backbone được giữ
cố định. Sau đó, dữ liệu tác vụ hiện tại được ánh xạ bằng adapter \(p_1\) và
ma trận chiếu ngẫu nhiên cố định để cộng dồn \(G,C\) và cập nhật
\(W_{\mathrm{out}}\) bằng nghiệm ridge, không dùng gradient hay lưu ảnh cũ.
Khi suy luận, Tầng 1 tái sử dụng điểm RP để sửa đề xuất task trước DRM; sau
DRM/CRM, Tầng 2 dùng cổng biên để hợp nhất cùng điểm RP với logit lớp cuối.
Mọi mô-đun đều được giữ cố định tại thời điểm suy luận và chỉ các task/lớp đã
quan sát được tham gia.
