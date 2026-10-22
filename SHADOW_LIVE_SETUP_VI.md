# Deployment shadow-live — Cloudflare Cron + GitHub Actions

Thiết kế này dùng Cloudflare Cron làm đồng hồ ổn định. Worker chỉ gọi GitHub
`workflow_dispatch`; API key OpenAQ và model inference vẫn chạy trong private GitHub
repository. Không có secret nào được ghi vào source code hoặc forecast ledger.

## Phạm vi chiến dịch

- Arms: frozen `gth10`, `gsea15/global15`, Local LightGBM theo station và Persistence; feature contract `v1.3`.
- Forecast output: 24 thời điểm tương lai kể từ lúc issue; model lead tự dịch theo độ
  trễ của quan sát PM2.5 gần nhất.
- Shadow stations: Bangkok SERVE_V1 cùng Bạch Mai, HCMC CMT8, Kuala Lumpur Setia,
  Vientiane UNICEF và Yangon AirGradient.
- Thời lượng: 7 ngày kể từ lần issue thành công đầu tiên; scoring tiếp tục thêm 4 ngày.
- Forecast và observation ledger là append-only trong `data/shadow/deployment-shadow-v1/`.

## 1. Đưa repo sạch lên GitHub

Tạo một **private repository** rồi commit toàn bộ repo này. Cần bảo đảm các file sau
có mặt trên GitHub:

- `.github/workflows/shadow-forecast.yml`
- `.github/workflows/shadow-score.yml`
- Các bundle được allowlist trong `.gitignore`: G-TH10, G-SEA15 global, Local Target5 và Local10 Bangkok.

Model bundle khoảng 9 MB và đã được `.gitignore` cho phép track có chủ đích. Các
artifacts nghiên cứu khác vẫn bị ignore.

Trong GitHub repository, tạo Actions secret tên chính xác `OPENAQ_API_KEY` tại
`Settings → Secrets and variables → Actions`. Dùng cùng OpenAQ v3 key đang được lưu
trong macOS Keychain; không tạo `.env` và không commit key.

Chạy thủ công workflow `PM25 - deployment shadow forecast` một lần. Chỉ khi job tạo được
forecast thì `data/shadow/deployment-shadow-v1/campaign.json` mới được tạo và đồng hồ bảy
ngày mới bắt đầu.

## 2. Tạo Cloudflare Worker trigger

Tạo một fine-grained GitHub personal access token giới hạn ở đúng private repository,
với quyền **Actions: Read and write**. Token này chỉ dùng để dispatch workflow; nó
không phải OpenAQ key.

Tại thư mục `cloudflare/`:

```bash
cp wrangler.toml.example wrangler.toml
```

Điền `GITHUB_OWNER`, `GITHUB_REPO`, và `GITHUB_REF` trong `wrangler.toml`, rồi chạy:

```bash
npm install
npx wrangler login
npx wrangler secret put GITHUB_TOKEN
npx wrangler deploy
```

Cloudflare schedules đều dùng UTC:

- `17 * * * *`: dispatch forecast mỗi giờ.
- `43 4 * * *`: dispatch ground-truth scoring mỗi ngày.

GitHub workflow có gate chống trùng: một issue đủ 144 forecasts trong cùng giờ UTC sẽ
không được chạy lại. Forecast và scoring dùng chung concurrency group để không push
ledger cùng lúc.

## 3. Kiểm tra sau khi bật

Một run forecast thành công phải có:

- `latest_run.json` báo tối đa `forecast_count = 576`;
- status theo từng station/model arm;
- mỗi forecast có `origin_time_utc`, `issued_at_utc`, `valid_at_utc`,
  `observation_age_hours`, `horizon`, `model_family`, `model_run` và `feature_values_sha256`;
- source responses nằm trong GitHub Actions artifact, không nằm trong Git history.

Sau khi target đã trôi qua 12 giờ, workflow scoring ghi:

- append-only exact-time observation ledgers;
- `artifacts/shadow/deployment-shadow-v1/REPORT.md`;
- metrics theo station, horizon, ngày và paired model comparisons.

Chỉ so khớp observation có đúng `sensor_id` và đúng interval-end UTC. Không nội suy
ground truth. Các forecast issue theo giờ bị chồng lấn, vì vậy không được coi từng sai
số là một mẫu thống kê độc lập.

## 4. Kết thúc

Sau 7 ngày, forecast gate ngừng gọi API. Scoring tiếp tục tối đa 4 ngày để nhận đủ
ground truth trễ. Sau đó disable Cron Triggers của Worker hoặc xóa Worker để tránh
dispatch các workflow đã hết hạn.

Kết quả shadow không tự động sửa `config/serve_v1.csv` và không tự động kích hoạt
SERVE_V2. Việc promote station/model phải dựa trên issuance coverage, RMSE/MAE so với
persistence theo từng station và độ ổn định ở các horizon.
