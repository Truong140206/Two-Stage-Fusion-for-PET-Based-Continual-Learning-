# Kiểm tra sau sửa mask — 2026-09-10

## Bản lưu và phạm vi sửa

Bản code lúc viết báo cáo vẫn nằm ở commit `069dfff` trong
`<workspace>/Hybrid_ReMatching`. Không reset hoặc sửa cây đó.
Bản phát hành đầu `5195aac` có cùng cấu trúc mã của 62 file Python xuất bản.
Các thay đổi hiện tại nằm trực tiếp trong `<workspace>/_verify`;
không tạo bản sao repo, không sửa checkpoint hoặc số liệu trong bài.

Đã sửa:
- Mask lớp chưa học trên từng đầu ra DRM/CRM, trước so sánh energy.
- Đo `ClsRouted` trước stage 2, kể cả khi beta khác 0.
- Truyền đúng dataset cho TII trong script train tổng quát.
- Audit log: bỏ run thiếu task/metric; giữ tất cả trường cấu hình khi ghép cặp;
  khoảng tin cậy dùng bậc tự do n-1; không gọi RP-only là RanPAC tương đương.
- Lệnh README và mô tả chi phí; log mới có hậu tố riêng.

**Số liệu Forgetting/Backward hiện có trong bài vẫn là số lịch sử, chưa được
chứng nhận lại bằng evaluator đã sửa.** Không tự thay số trước khi có log mới.

## Đưa thay đổi lên máy 4090

Thay `/path/to/projects` bằng thư mục dự án thực tế trên máy của bạn;
`user@your-server` và `C:\path\to\repo` cũng là giá trị mẫu, không phải địa chỉ kết nối.

Đồng bộ bản sửa từ nhánh `main` vào repo hiện có trên máy 4090. Trước tiên
kiểm tra cây trên máy đích để giữ nguyên thay đổi đang làm của người khác:

```bash
cd /path/to/projects/clean-check
git status --short
git pull --ff-only origin main
```

Nếu Git báo thay đổi chồng lấn hoặc nhánh đã phân kỳ, dừng để merge; không dùng
reset hoặc force. Sau khi pull thành công, chuyển thẳng sang bước 1 bên dưới.

Phương án thay thế khi không dùng Git: sau khi xác nhận các tệp đích không có
thay đổi cần giữ, đồng bộ đúng các tệp dưới đây từ Windows bằng PowerShell.
Không cần tạo thư mục repo mới. Không cần chạy cách này nếu đã pull thành công.

```powershell
$repoPath = 'C:\path\to\repo'
$remotePath = 'user@your-server:/path/to/projects/clean-check'
$filesToSync = @(
  'README.md',
  'engines/hrm_lora_wtp_and_tap_engine.py',
  'tools/audit_weight_evidence.py',
  'tools/verify_paper_results.py',
  'tests/test_verification_tools.py',
  'tests/test_evaluator_regressions.py',
  'training_scripts/train_any_4090.sh',
  'training_scripts/eval_rp_head_any_4090.sh',
  'training_scripts/eval_imagenet_r_conventional_4090.sh',
  'training_scripts/eval_cifar100_conventional_4090.sh',
  'training_scripts/eval_cub200_conventional_4090.sh',
  'reports/verification_after_mask_fix.md'
)
foreach ($relativePath in $filesToSync) {
  scp (Join-Path $repoPath $relativePath) "${remotePath}/${relativePath}"
  if ($LASTEXITCODE -ne 0) { throw "Sync failed: $relativePath" }
}
```

## 1. Kiểm thử code trước, chưa dùng GPU

Trên máy 4090:

```bash
cd /path/to/projects/clean-check
PY=/path/to/projects/Hybrid_ReMatching/.venv/bin/python
"$PY" -m pytest tests/ -q
```

Không chạy eval nếu kiểm thử fail. Các test mới gọi evaluator thật với đầu ra
mô hình giả lập trên CPU; chúng kiểm tra mask DRM, mask trước CRM energy,
và tính đúng `ClsRouted` khi bật fusion. Đây không thay thế đánh giá dataset thật.

## 2. Xác minh checkpoint CUB trước khi nghĩ đến train lại

```bash
"$PY" tools/verify_paper_results.py \
  --dataset cub200 --seeds 42 43 44 45 \
  --output-root /path/to/projects/hrm-pet-output \
  --data-root /path/to/projects/datasets \
  --checkpoints-only
```

Kiểm tra đủ 10 checkpoint TII và 10 checkpoint LoRA của từng seed: dataset,
seed, số task, backbone, LoRA rank và cờ/memory vi phạm exemplar-free.
Công cụ tính SHA-256 để ghi nguồn gốc checkpoint. Chỉ dùng với checkpoint của
chính dự án mà bạn tin cậy (chúng chứa argparse Namespace).

Nếu báo TII `dataset='Split-Imagenet-R'` thay vì `Split-CUB200`: dừng, không
đổi tên thư mục/file để bỏ qua. Cần truy đúng checkpoint CUB hoặc lập kế hoạch
train lại phần bị sai. Việc kiểm tra metadata không chứng minh toàn bộ lịch sử
huấn luyện; nó phát hiện sai cấu hình được lưu và memory bị cấm.

## 3. Pilot: ImageNet-R, seed 42, ba nhánh cùng checkpoint

ImageNet-R được kiểm tra trước khi đọc checkpoint hoặc dùng GPU: phải có
`train/` và `test/`, mỗi bên đủ 200 thư mục lớp cùng tên và có tệp ảnh. File nén
hoặc thư mục lồng rỗng không được dùng để đoán đường dẫn. Không tự giải nén,
di chuyển ảnh hoặc tạo lại train/test vì như vậy có thể đổi split lịch sử.
Đường dẫn đã chọn được in ở dòng `EVALUATION_DATA_PATH=...`.

Nếu tìm thấy hai split hợp lệ, chỉ định `--data-path` bằng đường dẫn cha của
thư mục `imagenet-r` đã dùng lúc chạy báo cáo. Nếu không tìm thấy split hợp lệ,
công cụ dừng và liệt kê vị trí đã kiểm tra; không tự tải lại dữ liệu.

Nếu lượt trước đã lỗi đường dẫn với tag `maskfix_verify_v1`, giữ nguyên log
lỗi và dùng `--tag maskfix_verify_v2` cho lượt chạy mới sau khi cập nhật code.

```bash
"$PY" tools/verify_paper_results.py \
  --dataset imr --seeds 42 \
  --output-root /path/to/projects/hrm-pet-output \
  --data-root /path/to/projects/datasets \
  --tag maskfix_verify_v1 --run
```

Không train lại TII/LoRA. Ba lượt eval tuần tự:
1. Baseline trực tiếp, không đầu RP.
2. Identity: w=1, beta=0.
3. Full: w=0.7, beta=0.5, gate margin.

Đầu RP vẫn phải được tích lũy/giải ridge từ dữ liệu train hiện có trong các
lượt có RP. Đây là phần fitting của phương pháp, không phải retrain HRM-PET.
Công cụ dừng nếu GPU có dưới 5 GiB trống; không tự dừng tiến trình người khác.

Log nằm ngay trong thư mục kết quả cũ, hậu tố `__maskfix_verify_v1.log`.
Không ghi đè file đã có. Nếu log hoàn tất và khớp nguồn code/checkpoint/lệnh,
lượt chạy tiếp dùng lại nó. Nếu log dở dang hoặc nguồn khác: giữ nguyên,
chọn tag mới, ví dụ `--tag maskfix_verify_v2`.

## 4. Chạy đủ ba dataset × bốn seed sau khi pilot đạt

```bash
for ds in imr cifar100 cub200; do
  "$PY" tools/verify_paper_results.py \
    --dataset "$ds" --seeds 42 43 44 45 \
    --output-root /path/to/projects/hrm-pet-output \
    --data-root /path/to/projects/datasets \
    --tag maskfix_verify_v1 --run || break
done
```

Pilot hoàn tất cùng tag sẽ được dùng lại; không chạy trùng. Nếu muốn chỉ đọc
và đối chiếu lại log đã có, bỏ `--run`.

## Đọc kết quả và tiêu chí

- `CHECKPOINT_DATASET_SEED_MODEL_PROTOCOL=PASS`: metadata/memory đã kiểm đạt.
- `IDENTITY_ALL_STAGES=PASS`: baseline và identity khớp cả 10 stage trên
  Acc@task, Acc@1, Acc@5, Loss và các metric retention có mặt, ngưỡng 1e-4.
- `HISTORICAL_FINAL_baseline/full=PASS`: bốn metric tại stage 10 khớp log cũ.
  Không yêu cầu retention khớp vì nó phụ thuộc các stage trung gian đã sửa.
- `HISTORICAL_COVERAGE=INCOMPLETE`: thiếu log lịch sử theo đúng tên; không được
  diễn giải thành đã tái lập toàn bộ số cũ. Không tự chọn log gần giống.
- `Intermediate deltas`: thay đổi Acc@1/Loss ở từng stage.
- `PAIRED_CORRECTED_SUMMARY`: chênh lệch full - baseline mới theo seed, mean,
  sample SD và CI95 khi có ít nhất hai seed.
- `CONSISTENCY_CHECKS=PASS` chỉ nói các phép so sánh đã thực hiện khớp,
  không khẳng định phương pháp tốt hơn hoặc mọi bảng trong bài đã được kiểm.

Giữ số cuối Acc@1/Acc@5/Loss nếu kiểm chứng đúng; cập nhật Forgetting/Backward
và lập luận liên quan từ log sửa. Nếu stage cuối lệch, dừng để tìm nguyên nhân,
không tự ép số hoặc nới tolerance. Lưới trọng số, ablation và SSL ngoài ba nhánh
trên cần được đối chiếu riêng nếu muốn chứng nhận lại toàn bộ bài.

## Audit log trọng số hiện có

```bash
"$PY" tools/audit_weight_evidence.py \
  /path/to/projects/hrm-pet-output
```

Log lịch sử và log sửa được tách theo suffix, không gộp vào cùng cấu hình.
Audit này không suy ra checkpoint giống nhau chỉ từ tên file; bộ xác minh ba
nhánh mới mới ghi SHA-256 cùng lệnh thực chạy để kiểm tra nguồn gốc.

## Giới hạn kiểm chứng tại Windows

Có thể chạy bộ kiểm tra công cụ bằng Python chuẩn:
`python -B -m unittest discover -s tests -p test_verification_tools.py -v`.
Máy này chưa có PyTorch/pytest; chưa chạy lại mô hình hoặc các test tensor.
Không cài thêm bộ ML lớn vào máy Windows chỉ để kiểm tra, không tạo bản sao repo.
