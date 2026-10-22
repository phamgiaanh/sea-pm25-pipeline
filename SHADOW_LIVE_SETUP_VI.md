# G-TH10 shadow-live — Cloudflare Cron + GitHub Actions

Thiết kế này dùng Cloudflare Cron làm đồng hồ ổn định. Worker chỉ gọi GitHub
`workflow_dispatch`; API key OpenAQ và model inference vẫn chạy trong private GitHub
repository. Không có secret nào được ghi vào source code hoặc forecast ledger.

## Phạm vi chiến dịch

- Model: `gth10`, 30 LightGBM models cho h1–h30, feature contract `v1.3`.
- Forecast output: 24 thời điểm tương lai kể từ lúc issue; model lead tự dịch theo độ
  trễ của quan sát PM2.5 gần nhất.
- Shadow stations: Bangkok SERVE_V1 cùng Bạch Mai, HCMC CMT8, Kuala Lumpur Setia,
  Vientiane UNICEF và Yangon AirGradient.
- Thời lượng: 7 ngày kể từ lần issue thành công đầu tiên; scoring tiếp tục thêm 4 ngày.
- Forecast ledger là append-only trong `data/shadow/gth10-shadow-v1/runs/`.

## 1. Đưa repo sạch lên GitHub

Tạo một **private repository** rồi commit toàn bộ repo này. Cần bảo đảm các file sau
có mặt trên GitHub:

- `.github/workflows/shadow-forecast.yml`
- `.github/workflows/shadow-score.yml`
- `artifacts/gth10/models/h01.txt` … `h30.txt`
- `artifacts/gth10/run_manifest.json` và `fit_manifest.csv`

Model bundle khoảng 9 MB và đã được `.gitignore` cho phép track có chủ đích. Các
artifacts nghiên cứu khác vẫn bị ignore.

Trong GitHub repository, tạo Actions secret tên chính xác `OPENAQ_API_KEY` tại
`Settings → Secrets and variables → Actions`. Dùng cùng OpenAQ v3 key đang được lưu
trong macOS Keychain; không tạo `.env` và không commit key.

Chạy thủ công workflow `PM25 - G-TH10 shadow forecast` một lần. Chỉ khi job tạo được
forecast thì `data/shadow/gth10-shadow-v1/campaign.json` mới được tạo và đồng hồ bảy
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

- `latest_run.json` báo `forecast_count = 144`;
- sáu status `FORECAST_ISSUED`;
- mỗi forecast có `origin_time_utc`, `issued_at_utc`, `valid_at_utc`,
  `observation_age_hours`, `model_horizon`, dự báo G-TH10 và persistence;
- source responses nằm trong GitHub Actions artifact, không nằm trong Git history.

Sau khi target đã trôi qua 12 giờ, workflow scoring ghi:

- `metrics/latest_global.csv`;
- `metrics/latest_by_station.csv`;
- `metrics/latest_by_station_horizon.csv`;
- `metrics/latest_summary.json`.

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
