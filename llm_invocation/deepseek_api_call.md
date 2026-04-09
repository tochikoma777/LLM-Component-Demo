### 代码：

```python
pip install openai  # DeepSeek 兼容 OpenAI SDK，直接复用
```

```
from openai import OpenAI

# 通过Kaggle Secrets，安全管理API Key
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("DEEPSEEK_API_KEY")


# ① 初始化客户端（指向 DeepSeek，也可换成任何兼容接口）
client = OpenAI(
    api_key=secret_value_0,         # 替换成你的 API Key
    base_url="https://api.deepseek.com"  # 换成 https://api.openai.com/v1 即为 GPT
)

# ② 构建对话消息（system 设定角色，user 是用户输入）
messages = [
    {
        "role": "system",
        "content": "你是一个幽默风趣的AI助手，擅长用生活比喻解释复杂的技术概念。"
    },
    {
        "role": "user",
        "content": "请用一个有趣的比喻，解释什么是神经网络的'权重'。"
    }
]

# ③ 调用 API（就像发送一条消息）
response = client.chat.completions.create(
    model="deepseek-chat",  # 模型名称
    messages=messages,
    temperature=0.8,        # 控制随机性：0=保守稳定，1=富有创意
    max_tokens=300          # 最大输出长度（约150个中文字）
)

# ④ 提取并打印回答
answer = response.choices[0].message.content
print("=" * 50)
print("🥰 大模型的回答：")
print("=" * 50)
print(answer)

# ⑤ 查看本次消耗的 Token 数（了解计费方式）
usage = response.usage
print(f"\n📊 Token 用量：输入 {usage.prompt_tokens}，输出 {usage.completion_tokens}")
```



### 输出：

```
==================================================
🥰 大模型的回答：
==================================================
想象一下，你是个刚入职的“火锅品鉴师”，任务是调配出完美的火锅底料。

**神经网络**就像一口大火锅，**数据**是各种食材（毛肚、牛肉、土豆…），而**权重**就是你手里那排神秘的调料罐——每个罐子对应一种调料（花椒量、辣度、盐比例…）。

刚开始你手忙脚乱：  
👉 花椒罐拧太猛 → 权重过大 → 锅底麻到怀疑人生  
👉 盐罐只抖两下 → 权重过小 → 食材煮出来像在泡温泉澡  

每次顾客（训练数据）皱眉，你就偷偷调整调料罐（反向传播算法）。三个月后：  
✅ 毛肚下锅时自动多转三下花椒罐  
✅ 煮土豆时把辣度调到“微微微辣”  
✅ 甚至能根据天气湿度微调麻油比例  

最后你成了“火锅仙人”——那些被调教到出神入化的调料罐，就是让神经网络突然开窍的**权重**。它们本质上就是：“面对不同食材时，该用多大力气拧哪个罐子”的肌肉记忆。  

（此刻某个正在吃火锅的AI工程师，偷偷把锅里的藕片想象成了欠训练的神经元…）

📊 Token 用量：输入 34，输出 268
```



### 总结：

#### 1、API KEY使用安全准则

**必须遵守的安全规范**

1. **永远不要硬编码**：绝对不要在代码中直接写入API Key
2. **使用Secrets优先**：Kaggle Secrets是首选方案
3. **定期轮换密钥**：每3-6个月更新一次API Key
4. **最小权限原则**：只授予API Key必要的最小权限
5. **监控使用情况**：定期检查API调用日志和费用

**常见风险与防范**

| 风险类型        | 错误做法                | 正确做法                            |
| --------------- | ----------------------- | ----------------------------------- |
| **代码泄露**    | `api_key = "sk-abc123"` | 使用`secrets.get_secret("API_KEY")` |
| **Git提交泄露** | 将`.env`文件提交到仓库  | 将`.env`添加到`.gitignore`          |
| **分支泄露**    | 认为分支不会继承Secrets | 确认分支后重新配置Secrets           |
| **日志泄露**    | 在日志中打印完整API Key | 只打印前几位：`sk-abc...`           |

#### 2、DeepSeek API 调用方法

**初始化客户端**

```python
from openai import OpenAI
client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"  # DeepSeek专用接口
)
```

 **构建对话消息**

```python
messages = [
    {"role": "system", "content": "设定AI角色"},
    {"role": "user", "content": "用户问题"}
]
```

- **system**: 设定AI行为风格
- **user**: 用户输入
- **assistant**: AI历史回复（多轮对话）

**调用API**

```python
response = client.chat.completions.create(
    model="deepseek-chat",  # 模型名称
    messages=messages,
    temperature=0.8,        # 创意度：0-2
    max_tokens=300,         # 最大输出长度
    stream=False            # 是否流式输出
)
```

**解析响应**

```python
answer = response.choices[0].message.content
usage = response.usage  # token用量统计
```

