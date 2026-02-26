# aigc-kit

统一 AIGC 能力工具包，封装图片生成、LLM Chat、Embedding 和存储后端，提供一致的调用接口。

## 安装

```bash
uv add git+https://github.com/yourorg/aigc-kit.git
```

## 模块概览

| 模块 | 说明 | Provider |
|------|------|----------|
| Image | 文生图 / 图生图 | volcengine, gemini, dashscope |
| LLM | Chat Completions | deepseek, dashscope, gemini, bedrock, volcengine, zhipu, moonshot |
| Embedding | 文本向量化 | bedrock (Titan), dashscope, stub |
| Storage | 对象存储 | Cloudflare R2 |

## 快速开始

### 图片生成

```python
from aigc_kit import ImageClient
from aigc_kit.storage import R2Storage

storage = R2Storage(
    access_key_id="...",
    access_key_secret="...",
    endpoint="https://xxx.r2.cloudflarestorage.com",
    bucket="my-bucket",
    public_domain="https://assets.example.com",
)

client = ImageClient(provider="gemini", storage=storage, model="gemini-3-pro-image-preview")
result = client.text_to_image("一只橘猫在阳光下打盹", size="1024x1024")
print(result.url)
```

### LLM Chat

```python
from aigc_kit.llm.providers.deepseek import DeepSeekProvider

llm = DeepSeekProvider(api_key="sk-xxx", model="deepseek-chat")
result = llm.chat([{"role": "user", "content": "你好"}])
print(result.content)

# 流式
for chunk in llm.chat_stream([{"role": "user", "content": "讲个故事"}]):
    if chunk.delta_content:
        print(chunk.delta_content, end="", flush=True)
```

### Embedding

```python
from aigc_kit.embedding.providers.dashscope import DashScopeEmbeddingProvider

emb = DashScopeEmbeddingProvider(api_key="sk-xxx")
result = emb.embed(["你好世界"])
print(len(result.vectors[0]))  # 1024
```

## 详细文档

参见 [llms.txt](./llms.txt) 获取完整 API 参考。

## License

MIT
