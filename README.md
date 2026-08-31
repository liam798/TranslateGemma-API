# TranslateGemma API

基于 FastAPI、Transformers 和单 GPU 的 TranslateGemma 翻译服务。接口兼容 Google Cloud Translate v2 风格，并额外提供 LibreTranslate 风格的 `/translate` 兼容入口。

## 功能

- FastAPI HTTP 服务
- Google Cloud Translate v2 风格接口：`/language/translate/v2`
- API Key 鉴权
- TranslateGemma 本地模型加载
- 4bit 量化加载，降低显存占用
- 单 GPU 推理锁，避免并发 generate 冲突
- 跨请求动态合批，提高 GPU 利用率
- 长文本切块翻译
- `detectedSourceLanguage` 返回
- URL、`{{变量}}`、`<标签>` 保护与恢复
- systemd 自启动模板

## 目录

```text
.
├── app.py
├── requirements.txt
├── translate-gemma.service
├── .env.example
└── README.md
```

## 安装

示例部署目录：

```bash
mkdir -p /data/translate-gemma
cd /data/translate-gemma
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

把模型放到：

```text
/data/translate-gemma/model
```

创建环境文件：

```bash
cp .env.example /data/translate-gemma/translate-gemma.env
```

然后修改 `TRANSLATE_GEMMA_API_KEY` 和模型路径。

## 启动

直接启动：

```bash
set -a
. /data/translate-gemma/translate-gemma.env
set +a
/data/translate-gemma/venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000
```

systemd 启动：

```bash
cp translate-gemma.service /etc/systemd/system/translate-gemma.service
systemctl daemon-reload
systemctl enable --now translate-gemma.service
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

## Google v2 风格接口

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

也支持批量：

```json
{
  "q": ["Hello", "Good morning"],
  "source": "en",
  "target": "vi",
  "key": "change-me"
}
```

## 兼容接口

POST `/translate`

```json
{
  "q": "Hello",
  "source": "en",
  "target": "vi",
  "api_key": "change-me"
}
```

## 鉴权

支持以下方式传入 API Key：

- JSON 字段：`key`
- JSON 字段：`api_key`
- Header：`x-api-key`
- Header：`Authorization: Bearer <key>`

## 语言码

服务会从 TranslateGemma 的 chat template 中读取支持语言列表。额外保留少量语言码别名：

- `zh`、`zh-CN`、`zh-CHS` -> `zh-Hans`
- `zh-CHT` -> `zh-TW`
- `fil` -> `tl`

当 `source=auto` 时，服务端会做轻量检测：

- 日文假名 -> `ja`
- 中文字符 -> `zh-Hans`
- 西语特征 -> `es`
- 其他默认 `en`

## 性能参数

核心参数：

- `TRANSLATE_GEMMA_MAX_BATCH_SIZE`：短文本动态合批上限，默认 `16`
- `TRANSLATE_GEMMA_BATCH_WAIT_SECONDS`：等待同一波请求进入 batch 的时间，默认 `0.05`
- `TRANSLATE_GEMMA_BATCH_MAX_CHARS`：单个 batch 最大总字符数，默认 `6000`
- `TRANSLATE_GEMMA_LONG_BATCH_SIZE`：长文本切块后的批量上限，默认 `4`
- `TRANSLATE_GEMMA_QUEUE_RESULT_TIMEOUT_SECONDS`：请求等待队列结果的超时时间，默认 `180`

## 注意

仓库不包含模型权重、API Key、服务器环境文件或 HuggingFace Token。部署时需要自行准备模型目录和环境文件。
