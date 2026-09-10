# Checkpoint review — 2026-09-10

## Kết luận

Có thể lưu làm mốc phát triển; CHƯA nên gọi là bản sẵn sàng nộp.
Không phát hiện sai công thức cốt lõi trong nhánh inference dùng cho paper.
Còn lỗi diễn giải ablation, số retention CIFAR chưa cập nhật, và thiếu thông tin
để kiểm chứng độc lập một số kết quả lịch sử. Không thay nội dung paper hoặc số
liệu trong lượt review này; không chạy thêm thí nghiệm trên lab.

Nguồn chuẩn: reports/lncs_method_en.tex và PDF cùng tên. Các số dòng dưới đây
tham chiếu nguồn tại commit 9db0dbae4e581aac6af7cf742603c106ccdd9b9a.
Checkpoint Git: paper-checkpoint-2026-09-10. Tag lưu cả nguồn, ảnh gốc,
PDF đang được kiểm tra và tài liệu này; không tạo folder/copy paper mới.

SHA-256 trước review:

| Tệp | SHA-256 |
|---|---|
| lncs_method_en.tex | 53588586DFFE8A3942F5B52C49217F05C1F7ED24B052F9F34FEF2C5414D47BE5 |
| lncs_method_en.pdf | 82D54D0B94211295557A6751F50F614BFB33534FC737D46F44CDFEF5F832F218 |
| figure_assets/mallard.jpg | 04B95BDE9B65ECD95931B2FA8310714EDFB12DFC70B3E4F19905969D40363B8A |

## Việc cần sửa trước khi gọi là bản hoàn chỉnh

### R1 — Số retention CIFAR cũ (ưu tiên cao)

Vị trí: Table 1, dòng 653–654; phần retention, dòng 681–691.
Log xác minh do tác giả cung cấp, tag maskfix_verify_v2, đủ bốn seed cho CIFAR:

| Paired full minus baseline | Đang viết | Tính lại từ log sau sửa |
|---|---|---|
| Forgetting, mean ± sample SD | -0.156 ± 0.098 | -0.150025 ± 0.095366 |
| Backward, mean ± sample SD | +0.142 ± 0.097 | +0.136100 ± 0.094869 |
| Forgetting, CI95 | [-0.312, +0.001] | [-0.301773, +0.001723] |
| Backward, CI95 | Không ghi ở đoạn này | [-0.014858, +0.287058] |

Full seed42/45 có Forgetting mới-cũ +0.0111, Backward mới-cũ -0.0111.
Phải cập nhật cả mean/SD tuyệt đối ở cột Proposed từ bốn final rows đầy đủ;
không suy ngược SD mới từ số đã làm tròn trong bảng.

Quan trọng: bản paper hiện tại ĐÃ nói cả sáu CI retention chứa 0. Vì vậy đây
là sửa số liệu, không phải đảo ngược một kết luận significant đang có trong paper.
Ghi chép bàn giao lịch sử nói 5/6 CI chứa 0 không còn là kết luận chuẩn.
Acc@1/Acc@task cuối của IMR và CIFAR không đổi qua lần sửa mask đã kiểm chứng.

### R2 — Kết luận về gate trên CUB mạnh hơn bằng chứng (ưu tiên cao)

Vị trí: abstract dòng 70–74; ablation dòng 771–778; conclusion dòng 841–846.
Gate trên CUB có ước lượng -0.284 pp nhưng CI95 [-0.570, +0.002] chứa 0.
Câu gate làm giảm accuracy và câu phương pháp thua trên một dataset cần:

- Ghi đó là xu hướng/ước lượng âm, chưa phân biệt được với 0 ở CI đang dùng.
- Nêu đúng đối chứng: full so với ungated class-only, KHÔNG phải HRM-PET.
- Không đánh đồng với kết quả chính: Table 1 vẫn báo full tốt hơn HRM-PET
  trên cả ba dataset.

Cách diễn đạt đề xuất: The gate has a negative estimated contribution on CUB-200,
but its confidence interval includes zero; the full method still improves over
HRM-PET.

### R3 — Hai nghĩa khác nhau của routing alone (ưu tiên cao)

Vị trí: abstract; Sect. 4.5 dòng 759–778; Sect. 4.7 dòng 828–835.
Phép so sánh bốn seed thực tế được mô tả là:

| Contrast | Điều kiện giữ nguyên | Ý nghĩa đúng |
|---|---|---|
| full - class+gate | beta=.5, gate=margin | Đóng góp routing khi đã có class fusion và gate |
| class+gate - class-only | w=1, beta=.5 | Đóng góp gate khi chưa có routing fusion |
| routing-only - baseline | beta=0 | Tác dụng routing riêng; không phải contrast thứ nhất |

Không gọi contrast thứ nhất là routing alone hoặc hai stage chạy riêng.
Đây là ablation có điều kiện theo đường thêm module, không phải phân rã đóng góp
duy nhất khi có tương tác giữa routing và gate.
Limitations nói conditional routing ablation là single-seed nhưng đoạn ngay
trước có contrast bốn seed; cần chỉ rõ single-seed là phần năm backbone.
Câu four seeds resolve its sign cũng không đúng cho CIFAR:
-0.030, CI [-0.085, +0.025].

### R4 — Căn cứ chọn w và phạm vi kiểm chứng chưa rõ (ưu tiên cao)

Vị trí: Sect. 4.6 dòng 807–823; sanity check dòng 618–624.
Câu chọn w=.7 vì cả sáu metric cải thiện ở mọi seed cần nêu rõ điều kiện
ROUTING-ONLY và đối chứng tại thời điểm chọn w, kèm bảng/log gốc.
Không được hiểu là full method: IMR seed43 có Forgetting +0.3796 và
Backward -0.4550 so với baseline sau sửa.
Chưa có raw logs cho tiêu chí lịch sử này trong lượt kiểm tra; không kết luận
tiêu chí sai, nhưng hiện chưa đủ truy vết.

Grid seed42 có max 75.61 tại w=.6 so với 75.51 tại w=.7. Câu cost nothing
measurable nên đổi thành chưa phân biệt được ở phép so sánh bốn seed
(Acc@1 +0.108, CI [-0.064,+0.280]); single-seed grid tự nó không chứng minh tương đương.
Các số Acc@task đã làm tròn 80.39/79.97/79.37 cho khoảng 1.02, trong khi văn
ghi 1.01; cần kiểm lại số chưa làm tròn trước khi sửa.

Sanity check nên ghi rõ seed, dataset và tên ba metric của ví dụ
77.7914/74.0191/1.2305 (IMR42: Acc@task/Acc@1/Loss), độ chính xác log
và phạm vi stage. Identity PASS không tự chứng minh mọi kết quả lịch sử,
mọi cấu hình ablation hoặc tính đúng đắn toàn bộ evaluator.

### R5 — Bằng chứng sau sửa chưa bao phủ toàn bộ paper

| Phạm vi | Bằng chứng đã xem | Giới hạn |
|---|---|---|
| IMR seeds42–45 | Metadata, identity all stages, historical final baseline/full PASS | Intermediate Loss và một số Acc@1 thay đổi; không nói toàn bộ log không đổi |
| CIFAR seeds42–45 | Các kiểm tra trên PASS | Retention full thay đổi nhỏ như R1 |
| CUB seed42 | PASS; Acc@1 full-baseline +1.2956 pp | Không phải mean bốn seed |
| CUB seeds43–45 | Seed43 identity log bị ngắt | Chưa có đủ bằng chứng hoàn tất sau sửa |
| 20-cell grid, ablation, weight sweep, diagnostics | Văn bản lịch sử và công cụ phân tích còn lưu | Chưa tái đối chiếu toàn bộ raw logs/provenance sau sửa |

Không được coi log CUB43 dở là kết quả hoàn chỉnh. Không có bằng chứng đủ để
gán lần ngắt đó cho OOM chỉ từ các dòng kernel mà không đối chiếu thời điểm/process.
Các kết quả lịch sử chưa tái xác minh không đồng nghĩa chúng đã sai.

### R6 — Thiếu chi tiết tái lập và baseline gần phương pháp nhất

Vị trí: Setup dòng 588–615; Sect. 4.3–4.5.

- Nêu explicit Acc@1/Acc@5/Acc@task là trung bình không trọng số qua các task
  đã thấy ở final stage: evaluator lấy sum(stat_matrix)/(task_id+1).
  Không phải pooled accuracy trên toàn bộ ảnh; khác biệt đáng lưu ý với 5-Datasets.
- Bổ sung số lớp/task, nguồn split và thứ tự lớp/domain theo seed, checkpoint
  pretrained cụ thể, preprocessing, LoRA insertion/depth/scaling, lora_type,
  thông số train/CTIRD hoặc trỏ tới cấu hình tái lập cố định.
- RP fit dùng ảnh TRAIN nhưng transform evaluation; không dùng nhãn test.
  Cần mô tả rõ thay vì chỉ nói features/fit chung chung.
- Câu retrained from scratch chỉ nên chỉ adapter/TII mới khởi tạo theo seed;
  backbone vẫn pretrained và frozen.
- Bảng 20 ô và bảng ablation hiện bị thay bằng khoảng số trong văn xuôi:
  cần bổ sung bảng/supplement hay file kết quả được dẫn rõ, cùng commands,
  checkpoint/config hashes và provenance. Không chỉ ghi released code mà
  thiếu đường dẫn/commit tái lập.
- RanPAC và APER được thảo luận nhưng không có đối chứng định lượng cùng
  protocol trong Table 2. Ít nhất đưa RP-only trên chính feature/checkpoint đang
  dùng; nếu thêm RanPAC chuẩn phải phân biệt với RP head mượn LoRA của HRM-PET.
  Đây là thí nghiệm tăng sức thuyết phục, không phải kết quả được phép tự điền.
- Grid đã được khảo sát trên dữ liệu phát triển/test; Limitations hiện có
  thừa nhận selection bias. Không đổi tên thành validation độc lập nếu không có.

### R7 — Vài câu khái quát quá rộng hoặc chưa định lượng

Vị trí: Intro dòng 87–110; Method dòng 495–510; Cost dòng 568–580.

- Không phải mọi PET method đều có task-specific pool/route; CODA/LAE/InfLoRA
  trong chính bảng đối chiếu là ví dụ cần phân biệt. Thu hẹp phát biểu về họ
  phương pháp đang xây trên HRM-PET, không coi đó là định nghĩa mọi PET.
- changes no training procedure nên là không đổi huấn luyện LoRA/TII,
  vì RP vẫn cần fit trên current-task train data.
- Ridge joint-fit/order invariance là tính chất toán học khi feature map cố định,
  không phải cam kết bitwise equality của phép cộng float64 theo mọi thứ tự.
- Fixed p1 trong code là cố định chỉ số adapter khi nạp checkpoint từng task;
  lượt này chưa so sánh tensor p1/backbone giữa toàn bộ checkpoint thực trên lab.
  Nên có kiểm tra invariant này bằng CPU/hash trước khi chứng nhận rộng hơn.
- 800 MB Gram đúng theo 10^8 float64 entries. 2.4 GB là ước lượng ba ma trận
  lớn, không phải measured peak của toàn pipeline (còn model/workspace/tensor).
  283 s cần nguồn log, điều kiện đo và không được coi là overhead so baseline.
- Diagnostic 74.31 ở Sect. 2.1 khớp số routing-only lịch sử, khác baseline
  74.02; cần gắn cấu hình của diagnostic để người đọc không lẫn với main table.

## Đối chiếu công thức và code: các phần đã khớp

Phạm vi: nhánh paper mặc định, không phải mọi thử nghiệm có trong repository.

| Nội dung | Code kiểm tra | Kết luận |
|---|---|---|
| TII chọn t0; initial adapted logits r | engines/hrm_lora_wtp_and_tap_engine.py:1254 | Khớp phân biệt u và r |
| Stage1 max theo seen task, population std, w=.7 | cùng file:730,1312 | Đối số tên tii_logits nhưng call site truyền r, không phải u |
| DRM khi proposal khác t0; CRM lấy confidence/candidate từ r | cùng file:1320–1376 | Khớp; CRM không chạy task1 |
| CRM so sánh log-sum-exp, mask trước so sánh | cùng file:1340–1365 | Khớp bản đã sửa |
| Gate top-two từ ell, beta=.5, z-score và affine map-back | cùng file:783–941 | Khớp Eqs.4–8 với sharpen=1, no ramp |
| RP features fixed adapter index0, extra pass, no gradient | cùng file:1300,2549; trainers/lora_trainer.py:311 | Khớp cấu hình; train=True chọn cách index LoRA, không bật tối ưu |
| ReLU, Gaussian W seed1993, G/C float64, ridge solve | engines/random_projection_head.py:108–175,430 | Khớp Eqs.1–3 |
| Forgetting/Backward chia T-1 | engine:2109–2111 | Khớp Eqs.9–10 |
| Acc@task là DRM proposal, không final CRM adapter | engine:1372,1423 | Paper đã giải thích đúng |

Code còn comment cũ: fuse_routers gọi log-probabilities/TII; fuse_class_scores
nói mixture unit variance. Hành vi code không theo các comment đó; paper hiện
mô tả đúng hơn comment. Nên dọn COMMENT riêng, không đổi công thức thực thi.
Không lấy lượt review này làm chứng nhận toàn bộ train/inference/options:
chưa chạy torch/GPU tests hay retrain trên máy hiện tại.

## References

Kiểm tra cấu trúc: 21 bibitems, 21 cited keys; không thiếu citation key,
không bibitem bỏ quên, không ref/label chưa định nghĩa hoặc label trùng.
Đã tra sự tồn tại và thông tin định danh của 21 tài liệu từ nguồn gốc dưới đây.
Không thấy reference bịa hoặc lỗi tên/venue rõ ràng trong các mục đã tra.
Không đồng nghĩa đã đọc và xác minh mọi khẳng định trong toàn văn cả 21 bài.

| Ref | Nguồn đối chiếu | Nhận xét |
|---|---|---|
| 1 HRM-PET | [NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/a978bdfeb195e4a574c0def98806346a-Abstract-Conference.html) | Tên, tác giả, năm khớp |
| 2 RanPAC | [NeurIPS paper](https://papers.nips.cc/paper_files/paper/2023/file/2793dc35e14003dd367684d93d236847-Paper-Conference.pdf) | Khớp; projection/second-order head phải tiếp tục được ghi công |
| 3 HiDe-Prompt | [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d9f8b5abc8e0926539ecbb492af7b2f1-Abstract-Conference.html) | Năm 2023 trong bản ta đúng; không chép năm 2024 bị ghi ở bib HRM-PET |
| 4 L2P | [CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Learning_To_Prompt_for_Continual_Learning_CVPR_2022_paper.html) | Khớp; có thể bổ sung pp.139–149 |
| 5 DualPrompt | [ECCV paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860617.pdf) | Khớp nhận diện; chuẩn hóa đầy đủ metadata |
| 6 LoRA | [Author preprint](https://arxiv.org/abs/2106.09685) | Nhận diện khớp; OpenReview trực tiếp bị browser challenge trong lượt này |
| 7 CODA-Prompt | [CVPR paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Smith_CODA-Prompt_COntinual_Decomposed_Attention-Based_Prompting_for_Rehearsal-Free_Continual_Learning_CVPR_2023_paper.pdf) | Khớp |
| 8 APER | [Springer IJCV](https://link.springer.com/article/10.1007/s11263-024-02218-0) | 133,1012–1032(2025) đúng; online 2024 khác issue 2025; không nhầm APER với APER† dùng exemplar |
| 9 Survey | [Author preprint](https://arxiv.org/abs/2010.15277) | Tên/tác giả khớp; pagination TPAMI chưa xác minh lại trực tiếp từ publisher |
| 10 iCaRL | [CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.html) | Khớp |
| 11 BiC | [CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Wu_Large_Scale_Incremental_Learning_CVPR_2019_paper.html) | Khớp |
| 12 WA | [CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhao_Maintaining_Discrimination_and_Fairness_in_Class_Incremental_Learning_CVPR_2020_paper.html) | Khớp |
| 13 LUCIR | [CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Hou_Learning_a_Unified_Classifier_Incrementally_via_Rebalancing_CVPR_2019_paper.html) | Khớp |
| 14 MoTE | [Elsevier](https://www.sciencedirect.com/science/article/pii/S095070512500841X) | Còn thiếu volume324/article113795; DOI10.1016/j.knosys.2025.113795 |
| 15 DLEPEM | [MDPI](https://www.mdpi.com/2076-3417/16/12/6153) | Khớp tác giả,16(12),6153(2026); có thật, xuất bản17/06/2026 |
| 16 S-Prompts | [NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/25886d7a7cf4e33fd44072a0cd81bf30-Abstract-Conference.html) | Khớp; chú thích ++ là variant trong benchmark HRM-PET |
| 17 LAE | [ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Gao_A_Unified_Continual_Learning_Framework_with_General_Parameter-Efficient_Tuning_ICCV_2023_paper.html) | Khớp |
| 18 CPrompt | [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Gao_Consistent_Prompting_for_Rehearsal-Free_Continual_Learning_CVPR_2024_paper.html) | Khớp |
| 19 InfLoRA | [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Liang_InfLoRA_Interference-Free_Low-Rank_Adaptation_for_Continual_Learning_CVPR_2024_paper.html) | Khớp |
| 20 ACIL | [NeurIPS 2022](https://proceedings.neurips.cc/paper/2022/hash/4b74a42fc81fc7ee252f6bcb6e26c8be-Abstract.html) | Khớp |
| 21 DS-AL | [AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/29670) | Khớp; bổ sung38(15),17237–17244, DOI10.1609/aaai.v38i15.29670 |

Đã đối chiếu đủ 32 giá trị trích dẫn (8 phương pháp × 4 datasets) ở Table 2
với Sup-21K/Table 1 trong [HRM-PET gốc](https://proceedings.neurips.cc/paper_files/paper/2025/file/a978bdfeb195e4a574c0def98806346a-Paper-Conference.pdf):
không thấy sai chép số. Hai hàng tự chạy không được coi là đã chạy lại cùng
protocol với tám hàng trích dẫn; caption hiện đã nêu khác seed/implementation.
Nên chỉ bold best trong từng nhóm, và ghi rõ HiDe-LoRA là bản dùng LoRA của
HiDe-Prompt trong benchmark đó, tránh ngầm tuyên bố thắng có kiểm soát.

Chưa có citation trực tiếp cho datasets, ViT và các pretrained settings
MoCo-v3/iBOT/DINO trong Setup. Bổ sung nguồn đúng checkpoint/dataset đã dùng,
không chỉ citation gián tiếp qua HRM-PET. Hoàn thiện DOI/pages/venue thống nhất;
References không tính vào giới hạn nội dung.

## PDF, trình bày và yêu cầu hội nghị

Đã render và xem lần lượt toàn bộ14 trang, không chỉ trang hình.
Đọc log build đang có và kiểm tra font/cross-reference; không recompile hay
thay PDF trong lượt review read-only này.

- 12 trang nội dung, gồm acknowledgment ở cuối trang12; References ở13–14.
  Đúng giới hạn12 trang KHÔNG tính references và yêu cầu hiện tên tác giả của
  [SOICT2026](https://soict.org/submission/paper-submission/) tại ngày kiểm tra.
- Hội nghị yêu cầu Springer CCIS; bản này dùng macro llncs thuộc bộ template
  Springer Computer Science. Không nhầm tên file lncs với một venue khác;
  đối chiếu sample CCIS cuối cùng trước submission/camera-ready, không sửa lề
  hoặc font size để lách giới hạn.
- Đủ bố cục: Intro, Related Work, Method (problem statement ở đầu),
  Experiments (main/prior/grid/ablation/weights/limits), Conclusion.
  Baseline HRM-PET nằm ở Related Work; Method chỉ giải thích phần ghép thêm.
- Không thấy ảnh mất, công thức bị cắt, glyph lỗi, chữ/nhãn chồng rõ rệt.
  Các font trong PDF đều được embed; không có missing glyph/undefined
  reference/overfull/underfull trong log hiện tại. Còn warning amsmath vec,
  không phải lỗi làm hỏng PDF.
- Hình ở trang5 hiện không còn lỗi neo mũi tên và nhãn đè mà người dùng đã chỉ.
  Tuy nhiên nhãn chính khoảng7pt, vài subscript khoảng5pt; nên tăng cỡ khi
  tái thiết kế. Caption còn gánh nhiều logic DRM/CRM; độc giả mới cần đọc cả
  caption/Method, không nên gọi hình hoàn toàn tự giải thích.
- Table1 khá dày; Loss CIFAR hiển thị -0.000 và SD0.00 che mất mức thay đổi.
  Dùng đủ chữ số cho Loss/delta, đơn vị pp rõ, không suy diễn negative zero.
- Table3 ở trang12, cách lời dẫn ở11; chưa phải lỗi nhưng có thể đặt gần hơn.
  Ref21 đứng riêng ở trang14: điểm dàn trang nên chỉnh, không vượt limit.
- Chưa thấy lỗi chính tả lớn qua đọc toàn văn. Nên giảm lối viết hội thoại
  (bought, costs nothing measurable, loses to the gate), ưu tiên đối chứng rõ.
- Disclosure of Interests mới là comment trong source, chưa hiện trong PDF.
  Hoàn thiện theo bộ hướng dẫn camera-ready; không tự điền tuyên bố thay tác giả.
- Credit ảnh có trong caption; lượt này chưa kiểm độc lập chain nguồn/license
  của chính file ảnh cục bộ. Cần giữ source URL/license record khi đóng bản nộp.

## Các kiểm tra thực hiện và phần chưa thực hiện

- Đọc nguồn paper, code RP/routing/gate/metric/dataloader và tài liệu hệ thống;
  ưu tiên code thực thi hơn comment hoặc bàn giao lịch sử.
- Đọc lại output verifier người dùng cung cấp, tính độc lập sample SD và CI
  retention CIFAR từ bốn paired deltas.
- 26 tests của test_verification_tools.py chạy bằng unittest trên CPU: PASS.
  Đây là tests công cụ kiểm chứng, KHÔNG phải full torch/GPU test suite.
- Không chạy lab/GPU, không tái huấn luyện, không so tensor checkpoint thực,
  không xác nhận mọi con số lịch sử từ raw logs, không test live Overleaf.
- Source/PDF/ảnh được giữ nguyên; review nêu lỗi để sửa ở lượt kế tiếp.

## Dữ liệu cần tác giả/Claude cung cấp, ưu tiên tận dụng log đã có

1. Bốn final rows baseline/full CIFAR sau maskfix, đủ sáu metric để cập nhật
   cả absolute mean/SD, không chỉ paired delta.
2. Khi có thể: trạng thái/log hoàn tất CUB43–45. Chỉ chạy phần thiếu khi lab rảnh
   và được đồng ý, không mặc định chạy lại cả hệ thống.
3. CSV/log cho20 cells và các arms routing-only, class-only, class+gate,
   full; đủ seeds/config/commit/provenance để phân biệt lịch sử với sau sửa.
4. Bằng chứng tiêu chí chọn w=.7 trước Stage2: từng seed, sáu metric,
   đối chứng cụ thể và split đã dùng chọn hyperparameter.
5. Run manifest/checkpoint URLs/preprocessing/class order cùng invariant
   backbone+p1 qua các task; việc so tensor/hash có thể làm trên CPU.
6. Nguồn log283s, measured memory nếu đã có, RP-only cùng protocol.
   Chưa biết thì ghi chưa đo, không tạo số.

Checkpoint này được lưu LOCAL; không push remote trong lượt này.
