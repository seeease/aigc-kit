# aigc-kit

统一 AIGC 能力工具包，封装图片生成、Chat Completions 和存储后端，提供一致的调用接口。

## 安装

```bash
# monorepo 内引用
uv add ../aigc-kit

# 或 git 仓库引用
uv add git+https://github.com/yourorg/aigc-kit.git
```

## 支持的 Provider

| Provider | 文生图 | 图生图 | 返回格式 | 备注 |
|----------|--------|--------|----------|------|
| volcengine (豆包 Seedream) | ✅ | ✅ | URL | 最小尺寸 2048x2048，自动 clamp |
| gemini (Vertex AI) | ✅ | ✅ | base64 | 需配合 Storage 上传获取 URL |

## 支持的 Storage

| Storage | 说明 |
|---------|------|
| R2 (Cloudflare) | S3 兼容，用于 base64 → URL 转换 |

## 快速开始

```python
from aigc_kit import ImageClient
from aigc_kit.storage import R2Storage

# 1. 创建存储（Gemini 需要，volcengine 可选）
storage = R2Storage(
    access_key_id="...",
    access_key_secret="...",
    endpoint="https://xxx.r2.cloudflarestorage.com",
    bucket="my-bucket",
    public_domain="https://assets.example.com",
)

# 2. 创建客户端
client = ImageClient(
    provider="gemini",          # 或 "volcengine"
    storage=storage,
    storage_key_prefix="aigc/images",
    model="gemini-3-pro-image-preview",  # provider 特定参数直接传
)

# 3. 调用
result = client.text_to_image("一只橘猫在阳光下打盹", size="1024x1024")
print(result.url)
```

## 使用文档

### 文生图

根据文字描述生成图片。

```python
result = client.text_to_image(
    "赛博朋克风格的未来城市夜景，霓虹灯倒映在雨水中",
    size="2048x2048",
)
print(result.url)       # 图片 URL
print(result.provider)  # "gemini" 或 "volcengine"
```

**参数说明：**
- `prompt`: 图片描述文本
- `size`: 图片尺寸，格式 `WxH`
  - volcengine: 最小 2048x2048，传小尺寸会自动放大并保持比例
  - gemini: 自动转换为最接近的 aspect_ratio + resolution

### 图生图

基于参考图片 + 文字描述生成新图片，适合角色场景照等需要保持一致性的场景。

```python
result = client.image_to_image(
    "角色站在樱花树下，微风吹过，花瓣飘落",
    reference_images=["https://assets.example.com/avatar.png"],
    size="2048x2048",
)
print(result.url)
```

**参数说明：**
- `prompt`: 场景描述
- `reference_images`: 参考图片 URL 列表
- `size`: 同文生图

### ImageResult 结构

```python
@dataclass
class ImageResult:
    url: str = ""           # 图片 URL（volcengine 直接返回，gemini 经 storage 上传后填充）
    base64: str = ""        # base64 编码（gemini 原始返回）
    mime_type: str = ""     # MIME 类型
    provider: str = ""      # 使用的 provider 名称
    metadata: dict = {}     # 额外元数据
```

## 环境变量

### Volcengine

```env
VOLCENGINE_API_KEY=your-api-key
VOLCENGINE_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
VOLCENGINE_IMAGE_MODEL=doubao-seedream-4-5-251128
VOLCENGINE_TIMEOUT_SECS=120
```

### Gemini (Vertex AI)

```env
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
GOOGLE_CLOUD_PROJECT=your-project
GOOGLE_CLOUD_LOCATION=global
GOOGLE_GENAI_USE_VERTEXAI=true
GEMINI_IMAGE_MODEL=gemini-3-pro-image-preview
```

### Cloudflare R2

```env
R2_ACCESS_KEY_ID=your-key-id
R2_ACCESS_KEY_SECRET=your-secret
R2_ENDPOINT=https://xxx.r2.cloudflarestorage.com
R2_BUCKET_NAME=your-bucket
R2_PUBLIC_DOMAIN=https://assets.example.com
```

## 添加新 Provider

1. 在 `src/aigc_kit/image/` 下新建文件，继承 `ImageProvider`
2. 实现 `text_to_image` 和 `image_to_image` 方法
3. 在 `client.py` 的 `_ensure_registered` 中注册

```python
class MyProvider(ImageProvider):
    @property
    def name(self) -> str:
        return "my_provider"

    def text_to_image(self, prompt, *, size="1024x1024", **kwargs):
        ...

    def image_to_image(self, prompt, *, reference_images, size="1024x1024", **kwargs):
        ...
```

---

## Chat Completions

统一 Chat 客户端，支持所有 OpenAI API 兼容后端。

### 支持的 Provider

| Provider | 说明 | Structured Output | Tool Calling | Streaming |
|----------|------|-------------------|--------------|-----------|
| openai | OpenAI 官方 | ✅ json_schema | ✅ | ✅ |
| deepseek | DeepSeek | ⚠️ 自动降级 json_object | ✅ | ✅ |
| glm / zhipu | 智谱 GLM | ✅ json_schema | ✅ | ✅ |
| volcengine / doubao | 火山引擎豆包 | ✅ json_schema | ✅ | ✅ |
| gemini | Google Gemini (Vertex AI) | ✅ response_json_schema | ✅ | ✅ |
| bedrock | Amazon Bedrock (Converse API) | ❌ | ✅ | ✅ |

### 快速开始

```python
from aigc_kit import ChatClient

# OpenAI 兼容后端 (deepseek / glm / volcengine / openai)
client = ChatClient(
    provider="deepseek",
    api_key="sk-xxx",
    model="deepseek-chat",
)

# Google Gemini (Vertex AI)
client = ChatClient(
    provider="gemini",
    model="gemini-2.5-flash",
    # project/location 从环境变量读取，或显式传入
)

# Amazon Bedrock
client = ChatClient(
    provider="bedrock",
    model="anthropic.claude-sonnet-4-20250514-v1:0",
    region="us-east-1",
)

# 同步调用
result = client.chat([{"role": "user", "content": "你好"}])
print(result.content)
```

### 流式输出

```python
for chunk in client.chat_stream([{"role": "user", "content": "讲个故事"}]):
    if chunk.delta_content:
        print(chunk.delta_content, end="", flush=True)
```

### Structured Output (JSON Schema)

不支持 `json_schema` 的模型（如 deepseek）会自动降级为 `json_object` + system prompt 引导。

```python
result = client.chat(
    [{"role": "user", "content": "生成一个角色"}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "character",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "role": {"type": "string"},
                },
                "required": ["name", "role"],
            },
        },
    },
)
```

### Tool Calling

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    }
]

# 单次调用
result = client.chat(messages, tools=tools)
if result.has_tool_calls:
    for tc in result.tool_calls:
        print(tc.name, tc.arguments)

# 自动循环（自动执行工具直到 LLM 返回最终文本）
def executor(name: str, args_json: str):
    return {"temperature": 22}

result = client.chat_with_tools(
    messages, tools,
    tool_executor=executor,
    max_iterations=10,
)
print(result.content)
```

### 非标准参数透传

通过 `**kwargs` 传递 provider 特有参数（如豆包的 thinking）：

```python
result = client.chat(
    messages,
    thinking={"type": "enabled"},  # 透传到 extra_body
)
```

### 添加新 Chat Provider

所有 OpenAI 兼容后端共用 `OpenAICompatProvider`，只需在 `client.py` 的 `compat_providers` 集合中注册名称即可。

如需完全自定义，继承 `ChatProvider` 并实现 `chat()` 和 `chat_stream()` 方法。
