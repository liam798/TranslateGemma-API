# TranslateGemma API

TranslateGemma API 是一个基于 FastAPI、Transformers 和单 GPU 的私有化翻译服务。它加载本地 `google/translategemma-4b-it` 模型，并提供 Google Cloud Translate v2 风格接口。

## 特性

- 私有部署，不依赖外部翻译平台
- Google Cloud Translate v2 风格 API
- API Key 鉴权
- 单 GPU 常驻模型推理
- 4bit 量化加载，降低显存占用
- 动态合批，提高并发吞吐
- 按语言方向和文本长度分桶，降低混合流量长尾延迟
- 长文本切块翻译
- 返回 `detectedSourceLanguage`
- 保护 URL、`{{变量}}`、`<标签>`
- 提供 `/health` 和 `/metrics`
- 提供 systemd 自启动模板

## 文件结构

```text
.
├── app.py
├── requirements.txt
├── translate-gemma.service
├── .env.example
├── .gitignore
└── README.md
```

仓库不包含模型权重、API Key、服务器环境文件或 HuggingFace Token。

## 部署要求

- Linux 服务器
- NVIDIA GPU
- 可用 CUDA 环境
- Python 3.10+
- 已下载好的 TranslateGemma 模型目录

默认部署路径：

```text
/data/translate-gemma
```

默认模型路径：

```text
/data/translate-gemma/model
```

## 快速部署

创建目录：

```bash
mkdir -p /data/translate-gemma
cd /data/translate-gemma
```

准备代码文件：

```bash
cp app.py requirements.txt .env.example translate-gemma.service /data/translate-gemma/
```

创建 Python 环境：

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

创建环境文件：

```bash
cp .env.example /data/translate-gemma/translate-gemma.env
```

编辑 `/data/translate-gemma/translate-gemma.env`，至少配置：

```bash
TRANSLATE_GEMMA_MODEL_DIR=/data/translate-gemma/model
TRANSLATE_GEMMA_API_KEY=change-me
```

## 启动

前台启动：

```bash
cd /data/translate-gemma
set -a
. ./translate-gemma.env
set +a
./venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000
```

systemd 启动：

```bash
cp /data/translate-gemma/translate-gemma.service /etc/systemd/system/translate-gemma.service
systemctl daemon-reload
systemctl enable --now translate-gemma.service
```

查看状态：

```bash
systemctl status translate-gemma.service
```

## 接口

### 健康检查

```bash
curl http://127.0.0.1:8000/health
```

响应：

```json
{
  "status": "ok",
  "model": "google/translategemma-4b-it"
}
```

### 翻译

POST `/language/translate/v2`

```bash
curl -X POST http://127.0.0.1:8000/language/translate/v2 \
  -H "Content-Type: application/json" \
  -d '{
    "q": "Hello",
    "source": "en",
    "target": "vi",
    "key": "change-me"
  }'
```

响应：

```json
{
  "data": {
    "translations": [
      {
        "translatedText": "Xin chào",
        "detectedSourceLanguage": "en"
      }
    ]
  }
}
```

批量请求：

```json
{
  "q": ["Hello", "Good morning"],
  "source": "en",
  "target": "vi",
  "key": "change-me"
}
```

### 运行指标

```bash
curl http://127.0.0.1:8000/metrics
```

常见字段：

| 字段 | 说明 |
| --- | --- |
| `requests_total` | 请求次数 |
| `jobs_total` | 翻译任务数 |
| `batches_total` | 实际模型推理批次数 |
| `batch_items_avg` | 平均每批文本条数 |
| `batch_chars_avg` | 平均每批字符数 |
| `generate_seconds_avg` | 平均每批模型生成耗时 |
| `errors_total` | 模型推理错误数 |
| `queue_full_total` | 队列满次数 |
| `queue_timeout_total` | 请求等待结果超时次数 |
| `queue_size` | 当前队列长度 |

## 请求字段

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `q` | `string` 或 `string[]` | 是 | 待翻译文本 |
| `source` | `string` | 否 | 来源语言。可传 `auto` |
| `target` | `string` | 是 | 目标语言 |
| `key` | `string` | 否 | API Key |
| `api_key` | `string` | 否 | API Key 兼容字段 |
| `format` | `string` | 否 | 保留字段，默认 `text` |

也支持使用 Header 传 API Key：

```text
x-api-key: <key>
Authorization: Bearer <key>
```

## 语言码

服务启动时会优先从 TranslateGemma 的 chat template 中读取模型支持的语言列表。

内置语言码别名：

| 输入 | 标准化后 |
| --- | --- |
| `zh` | `zh-Hans` |
| `zh-CN` | `zh-Hans` |
| `zh-CHS` | `zh-Hans` |
| `zh-CHT` | `zh-TW` |
| `fil` | `tl` |

当 `source=auto` 时，服务端使用轻量规则检测：

| 文本特征 | 检测结果 |
| --- | --- |
| 日文假名 | `ja` |
| 中文字符 | `zh-Hans` |
| 西语重音符号或常见词 | `es` |
| 其他 | `en` |

注意：当前 `auto` 不是完整语言检测模型，只适合作为基础兜底。

## 内容保护

翻译前会保护以下内容，翻译后再恢复：

- `{{user}}` 这类变量
- `<START>` 这类尖括号标签
- `http://` 和 `https://` URL

当前不会结构化解析 JSON。若要翻译 JSON 内容，建议调用方先解析 JSON，只把需要翻译的 value 传入服务，避免模型翻译 key 或破坏结构。

## 性能机制

服务内部使用单个后台 batch worker 进行模型推理：

1. HTTP 请求进入内存队列。
2. worker 短暂等待同一波请求。
3. 按来源语言、目标语言、文本长度分桶。
4. 同桶文本合并成 batch。
5. 单次调用 `model.generate()`。
6. 将 batch 结果回填给对应请求。

这样可以减少多请求下的单条推理开销，并避免长文本拖慢短文本。

## 配置项

| 配置 | 默认值 | 说明 |
| --- | --- | --- |
| `TRANSLATE_GEMMA_MODEL_DIR` | `/data/translate-gemma/model` | 模型目录 |
| `TRANSLATE_GEMMA_API_KEY` | 空 | 服务端 API Key |
| `TRANSLATE_GEMMA_MAX_NEW_TOKENS` | `2048` | 单次最大生成 token |
| `TRANSLATE_GEMMA_CHUNK_CHARS` | `900` | 长文本切块字符数 |
| `TRANSLATE_GEMMA_CHUNK_MAX_NEW_TOKENS` | `768` | 单块生成 token 上限 |
| `TRANSLATE_GEMMA_MAX_BATCH_SIZE` | `24` | 最大 batch 条数 |
| `TRANSLATE_GEMMA_LONG_BATCH_SIZE` | `4` | 长文本切块 batch 上限 |
| `TRANSLATE_GEMMA_LOCK_WAIT_SECONDS` | `60` | 入队等待时间 |
| `TRANSLATE_GEMMA_BATCH_WAIT_SECONDS` | `0.08` | 普通 batch 聚合等待时间 |
| `TRANSLATE_GEMMA_BATCH_MIN_WAIT_SECONDS` | `0.005` | 高压力或长文本最小等待时间 |
| `TRANSLATE_GEMMA_BATCH_MAX_CHARS` | `6000` | 单批最大字符数 |
| `TRANSLATE_GEMMA_QUEUE_MAX_SIZE` | `256` | 队列最大任务数 |
| `TRANSLATE_GEMMA_QUEUE_RESULT_TIMEOUT_SECONDS` | `180` | 请求等待结果超时 |

## 运维命令

查看服务：

```bash
systemctl status translate-gemma.service
```

查看日志：

```bash
journalctl -u translate-gemma.service -f
```

重启：

```bash
systemctl restart translate-gemma.service
```

查看 GPU：

```bash
nvidia-smi
```

## 安全说明

- 不要把真实 API Key 写入仓库。
- 不要把 HuggingFace Token 写入仓库。
- 不要提交模型权重。
- 建议只把服务监听在 `127.0.0.1`，公网入口放在反向代理或隧道层，并在入口层继续做访问控制。
