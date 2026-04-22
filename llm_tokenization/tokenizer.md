

---

## 主流大模型分词器



#### 1. GPT系列 (Byte-level BPE) - 使用 `tiktoken`

**代码**：

```python
import tiktoken

# 初始化不同模型的编码器（映射关系参考OpenAI官方文档）
encoders = {
    "gpt2": tiktoken.get_encoding("r50k_base"),           # GPT-2
    "gpt-3.5-turbo": tiktoken.get_encoding("cl100k_base"),# GPT-3.5/4
    "gpt-4": tiktoken.get_encoding("cl100k_base"),
    "gpt-4o": tiktoken.get_encoding("o200k_base"),        # GPT-4o
}

def gpt_tokenize(text, model="gpt-4"):
    """GPT系列分词（支持自动选择模型编码器）"""
    # 1. 根据模型选择编码器（容错处理）
    if model not in encoders:
        raise ValueError(f"不支持的模型: {model}，可选模型: {list(encoders.keys())}")
    encoder = encoders[model]
    
    # 2. 编码
    token_ids = encoder.encode(text)
    
    # 3. 解码验证
    decoded = encoder.decode(token_ids)
    
    # 4. 获取可读的token（用于分析）
    tokens = []
    for tid in token_ids:
        try:
            token_bytes = encoder.decode_single_token_bytes(tid)
            tokens.append(token_bytes.decode('utf-8', errors='replace'))
        except:
            tokens.append(f"<token_{tid}>")
    
    return {
        "token_ids": token_ids,
        "tokens": tokens,
        "count": len(token_ids),
        "decoded": decoded,
        "lossless": decoded == text
    }

# 使用示例
text = "Hello, 你好! 🎉 Tokenization is crucial."
result = gpt_tokenize(text, model="gpt-4o")  # 可切换模型测试
print(f"编码结果: {result['token_ids']}")
print(f"Token数量: {result['count']}")
print(f"Tokens: {result['tokens']}")
print(f"解码验证: {result['decoded']}")
print(f"无损还原: {result['lossless']}")
```

**输出：**

```bash
编码结果: [13225, 11, 220, 177519, 0, 139786, 231, 17951, 2860, 382, 19008, 13]
Token数量: 12
Tokens: ['Hello', ',', ' ', '你好', '!', ' �', '�', ' Token', 'ization', ' is', ' crucial', '.']
解码验证: Hello, 你好! 🎉 Tokenization is crucial.
无损还原: True
```

**关键特性：**

- 基于字节级BPE，可处理任意Unicode字符
- 无OOV（Out-of-Vocabulary）问题
- 词表包含256字节基础token + 合并子词

---

#### 2. BERT系列 (WordPiece) - 使用 `transformers`

**代码：**

```python
from transformers import BertTokenizer

# 加载预训练分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def bert_tokenize(text):
    """BERT WordPiece分词"""
    
    # 编码（自动添加[CLS]和[SEP]）
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded['input_ids'][0].tolist()
    
    # 获取token字符串
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    # 解码（去除特殊标记）
    decoded = tokenizer.decode(input_ids, skip_special_tokens=True)
    
    return {
        "token_ids": input_ids,
        "tokens": tokens,
        "count": len(input_ids),
        "decoded": decoded,
        "has_special_tokens": "[CLS]" in tokens or "[SEP]" in tokens
    }

# 使用示例
text = "Tokenization is crucial for GPT-4."
result = bert_tokenize(text)
print(f"编码结果: {result['token_ids']}")
print(f"Token数量: {result['count']}")
print(f"Tokens示例: {result['tokens'][:15]}")
print(f"解码验证: {result['decoded']}")
print(f"包含特殊标记: {result['has_special_tokens']}")
```

**输出：**

```bash
编码结果: [101, 19204, 3989, 2003, 10232, 2005, 14246, 2102, 1011, 1018, 1012, 102]
Token数量: 12
Tokens示例: ['[CLS]', 'token', '##ization', 'is', 'crucial', 'for', 'gp', '##t', '-', '4', '.', '[SEP]']
解码验证: tokenization is crucial for gpt - 4.
包含特殊标记: True
```

**关键特性：**

- 使用 `##` 标记子词延续（如 `tokenization` → `token`, `##ization`）
- 保留词根信息，利于理解词形变化
- 自动添加 `[CLS]`（开头）和 `[SEP]`（结尾）

---

#### 3. T5系列 (SentencePiece) - 使用 `transformers`

**代码：**

```python
from transformers import T5Tokenizer

# 加载T5分词器
tokenizer = T5Tokenizer.from_pretrained("t5-base")

def t5_tokenize(text):
    """T5 SentencePiece分词"""
    
    # 编码
    input_ids = tokenizer.encode(text)
    
    # 解码
    decoded = tokenizer.decode(input_ids)
    
    # SentencePiece特点：使用 ▁ 标记词首
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    return {
        "token_ids": input_ids,
        "tokens": tokens,
        "count": len(input_ids),
        "decoded": decoded,
        "lossless": decoded == text
    }

# 使用示例
text = "你好，世界！这是一个测试。"
result = t5_tokenize(text)
print(f"Token数量: {result['count']}")
print(f"Tokens: {result['tokens']}")  # 注意 ▁ 标记
```

**输出：**（中文支持不是很好）

```bash
Token数量: 3
Tokens: ['▁', '<unk>', '</s>']
```

**换个T5分词器，代码：**

```python
from transformers import T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained("Langboat/mengzi-t5-base")

def t5_tokenize(text):
    """T5分词（与GPT/BERT完全同格式）"""
    
    # 核心：用 __call__ 方式，最稳定
    # tokenizer.encode方法在部分版本会自动加空格、乱拆中文！
    tokens = tokenizer.tokenize(text)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    # 解码
    decoded = tokenizer.decode(input_ids, skip_special_tokens=True)

    return {
        "token_ids": input_ids,
        "tokens": tokens,
        "count": len(input_ids),
        "decoded": decoded,
        "lossless": decoded.strip() == text
    }

text = "你好，世界！这是一个测试。"
result = t5_tokenize(text)

print(f"Token数量: {result['count']}")
print(f"Tokens: {result['tokens']}")
print(f"解码验证: {result['decoded']}")
```

**输出：**（中文效果好些）：

```
Token数量: 8
Tokens: ['▁', '你好', '，', '世界', '！', '这是一个', '测试', '。']
解码验证: 你好世界这是一个测试。
```

**关键特性：**

- 使用 `▁` (U+2581) 标记词的开头
- 语言无关，直接处理原始文本（无需预分词）
- 支持可逆解码（detokenization无歧义）

---

#### 4. LLaMA/CodeLlama 系列

```python
from transformers import LlamaTokenizer

# 加载LLaMA风格分词器（使用CodeLlama作为开源示例）
tokenizer = LlamaTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")

def llama_tokenize(text):
    """LLaMA风格分词（SentencePiece BPE）"""
    
    input_ids = tokenizer.encode(text)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    decoded = tokenizer.decode(input_ids)
    
    return {
        "token_ids": input_ids,
        "tokens": tokens,
        "count": len(input_ids),
        "decoded": decoded,
        "lossless": decoded == text
    }

# 关键差异
text = "The number is 12345"
result = llama_tokenize(text)
# 注意：LLaMA会将连续数字拆分为单独数字："1", "2", "3", "4", "5"
print(f"Token数量: {result['count']}")
print(f"Tokens: {result['tokens']}")
print(f"解码验证: {result['decoded']}")
```

**输出：**

```bash
Token数量: 10
Tokens: ['<s>', '▁The', '▁number', '▁is', '▁', '1', '2', '3', '4', '5']
解码验证: <s> The number is 12345
```

**关键特性：**

- 显式拆分连续数字（提升数学计算能力）
- 针对代码优化（空格、缩进、标点特殊处理）
- 基于SentencePiece，但采用BPE模式

#### 5、评估指标代码

```python
import re
from transformers import T5Tokenizer

def evaluate_tokenizer_detailed(tokenizer, test_cases):
    """全面评估分词器性能"""
    results = {}
    
    for name, text in test_cases.items():
        # 1. 编码
        ids = tokenizer.encode(
            text,
            truncation=True,
            max_length=512
        )
        
        # 2. 获取分词列表
        tokens = tokenizer.convert_ids_to_tokens(ids)
        
        # 3. 解码
        decoded = tokenizer.decode(ids, skip_special_tokens=True).strip()
        is_lossless = decoded == text.strip()
        
        # 4. 统计
        token_count = len(ids)
        word_count = len(text.split())
        fertility = token_count / word_count if word_count > 0 else 0
        
        # 5. 表情支持
        has_special = any(ord(c) > 127 for c in text)
        emoji_handling = "支持" if has_special and token_count > 0 else "不支持"
        
        # 6. 数字拆分
        digits = re.findall(r'\d+', text)
        digit_tokens = sum(len(tokenizer.encode(d, truncation=True)) for d in digits)
        digit_splitting = len(digits) > 0 and digit_tokens > len(digits)

        results[name] = {
            "original_text": text,
            "decoded_text": decoded,
            "tokens": tokens,  # 分词内容
            "token_ids": ids,
            "token_count": token_count,
            "word_count": word_count,
            "fertility": round(fertility, 2),
            "lossless": is_lossless,
            "emoji_support": emoji_handling,
            "digit_splitting": digit_splitting,
        }
    
    return results

# 测试用例
test_cases = {
    "short_english": "Hello world",
    "long_english": "Natural language processing is fascinating",
    "chinese": "自然语言处理很重要",
    "mixed": "GPT-4和人工智能",
    "code": "def hello_world():\n    print('Hello')",
    "emoji": "Hello 🎉",
    "numbers": "The answer is 12345"
}
```

---

```python
# 加载分词器（你可以换成任何分词器：BERT/GPT/T5）
tokenizer_gpt = tiktoken.get_encoding("cl100k_base")

# 执行评估
results = evaluate_tokenizer_detailed(tokenizer, test_cases)

# ===================== 超详细打印（含分词内容） =====================
print("=" * 90)
print("🔍 分词器详细评估报告")
print("=" * 90)

for case_name, res in results.items():
    print(f"\n📌 测试用例: {case_name}")
    print(f"├─ 原始文本: {res['original_text']}")
    print(f"├─ 解码结果: {res['decoded_text']}")
    print(f"├─ 分词结果: {res['tokens']}")  
    print(f"├─ Token ID: {res['token_ids']}")
    print(f"├─ Token数量: {res['token_count']}")
    print(f"├─ 单词数量: {res['word_count']}")
    print(f"├─ 平均Token/词: {res['fertility']}")
    print(f"├─ 无损还原: {res['lossless']}")
    print(f"├─ 表情支持: {res['emoji_support']}")
    print(f"└─ 数字拆分: {res['digit_splitting']}")
    print("-" * 90)
```

**输出：**

```bash
==========================================================================================
🔍 分词器详细评估报告
==========================================================================================

📌 测试用例: short_english
├─ 原始文本: Hello world
├─ 解码结果: Hello world
├─ 分词结果: ['<s>', '▁Hello', '▁world']
├─ Token ID: [1, 15043, 3186]
├─ Token数量: 3
├─ 单词数量: 2
├─ 平均Token/词: 1.5
├─ 无损还原: True
├─ 表情支持: 不支持
└─ 数字拆分: False
------------------------------------------------------------------------------------------

📌 测试用例: long_english
├─ 原始文本: Natural language processing is fascinating
├─ 解码结果: Natural language processing is fascinating
├─ 分词结果: ['<s>', '▁Natural', '▁language', '▁processing', '▁is', '▁fasc', 'in', 'ating']
├─ Token ID: [1, 18385, 4086, 9068, 338, 21028, 262, 1218]
├─ Token数量: 8
├─ 单词数量: 5
├─ 平均Token/词: 1.6
├─ 无损还原: True
├─ 表情支持: 不支持
└─ 数字拆分: False
------------------------------------------------------------------------------------------

📌 测试用例: chinese
├─ 原始文本: 自然语言处理很重要
├─ 解码结果: 自然语言处理很重要
├─ 分词结果: ['<s>', '▁', '自', '然', '语', '言', '处', '理', '<0xE5>', '<0xBE>', '<0x88>', '重', '要']
├─ Token ID: [1, 29871, 30688, 31516, 31505, 31243, 31548, 30687, 232, 193, 139, 30908, 30698]
├─ Token数量: 13
├─ 单词数量: 1
├─ 平均Token/词: 13.0
├─ 无损还原: True
├─ 表情支持: 支持
└─ 数字拆分: False
------------------------------------------------------------------------------------------

📌 测试用例: mixed
├─ 原始文本: GPT-4和人工智能
├─ 解码结果: GPT-4和人工智能
├─ 分词结果: ['<s>', '▁G', 'PT', '-', '4', '和', '人', '工', '智', '能']
├─ Token ID: [1, 402, 7982, 29899, 29946, 30503, 30313, 31041, 31676, 30815]
├─ Token数量: 10
├─ 单词数量: 1
├─ 平均Token/词: 10.0
├─ 无损还原: True
├─ 表情支持: 支持
└─ 数字拆分: True
------------------------------------------------------------------------------------------

📌 测试用例: code
├─ 原始文本: def hello_world():
    print('Hello')
├─ 解码结果: def hello_world():
    print('Hello')
├─ 分词结果: ['<s>', '▁def', '▁hello', '_', 'world', '():', '<0x0A>', '▁▁▁', '▁print', "('", 'Hello', "')"]
├─ Token ID: [1, 822, 22172, 29918, 11526, 7295, 13, 1678, 1596, 877, 10994, 1495]
├─ Token数量: 12
├─ 单词数量: 3
├─ 平均Token/词: 4.0
├─ 无损还原: True
├─ 表情支持: 不支持
└─ 数字拆分: False
------------------------------------------------------------------------------------------

📌 测试用例: emoji
├─ 原始文本: Hello 🎉
├─ 解码结果: Hello 🎉
├─ 分词结果: ['<s>', '▁Hello', '▁', '<0xF0>', '<0x9F>', '<0x8E>', '<0x89>']
├─ Token ID: [1, 15043, 29871, 243, 162, 145, 140]
├─ Token数量: 7
├─ 单词数量: 2
├─ 平均Token/词: 3.5
├─ 无损还原: True
├─ 表情支持: 支持
└─ 数字拆分: False
------------------------------------------------------------------------------------------

📌 测试用例: numbers
├─ 原始文本: The answer is 12345
├─ 解码结果: The answer is 12345
├─ 分词结果: ['<s>', '▁The', '▁answer', '▁is', '▁', '1', '2', '3', '4', '5']
├─ Token ID: [1, 450, 1234, 338, 29871, 29896, 29906, 29941, 29946, 29945]
├─ Token数量: 10
├─ 单词数量: 4
├─ 平均Token/词: 2.5
├─ 无损还原: True
├─ 表情支持: 不支持
└─ 数字拆分: True
------------------------------------------------------------------------------------------
```



## 比较和选型

### **对比表**

| 维度                | GPT系列（tiktoken/BPE） | BERT系列（WordPiece） | T5系列（SentencePiece-Unigram） | LLaMA系列（SentencePiece-BPE）    | RoBERTa（BPE） |
| ------------------- | ----------------------- | --------------------- | ------------------------------- | --------------------------------- | -------------- |
| **核心算法**        | 字节级BPE               | WordPiece             | Unigram 语言模型                | BPE（数字/代码优化）              | 改进版BPE      |
| **词表大小**        | 50K~200K（GPT-4o最大）  | ~30K                  | ~32K                            | 32K~128K                          | ~50K           |
| **子词标记**        | 无特殊标记              | `##` 表子词延续       | `▁` 表单词开头                  | `▁` 表单词开头                    | 无特殊标记     |
| **OOV问题**         | 完全无（字节兜底）      | 极少（可能出[UNK]）   | 完全无                          | 完全无                            | 极少           |
| **中文效果**        | 一般（单字拆分多）      | 一般                  | 优秀（原生支持无空格）          | 优秀                              | 一般           |
| **特殊字符**        | 完美支持（Emoji/符号）  | 差（易出[UNK]）       | 良好                            | 优秀                              | 良好           |
| **数字处理**        | 常规拆分                | 常规拆分              | 常规拆分                        | **强制拆单数字**（数学/代码专用） | 常规拆分       |
| **解码保真**        | 无损解码                | 无损（去特殊标记）    | 完全可逆无损                    | 完全可逆无损                      | 无损           |
| **自动加特殊Token** | 否                      | 是（[CLS]/[SEP]）     | 是（</s>）                      | 否                                | 否             |
| **代表场景**        | 通用对话、多语言        | 文本分类、理解任务    | 翻译、摘要、多语言              | 代码、数学、长文本                | 鲁棒文本理解   |

---

### 优缺点

1. GPT系列（最通用、最稳）

- **优点**：无OOV、Unicode全支持（Emoji/特殊符号无敌）、无损解码、生态最成熟
- **缺点**：中文拆分偏碎、无专门代码/数学优化
- **适用**：90%通用场景、多语言混合文本、API开发

2. BERT系列（纯文本理解专用）

- **优点**：子词带词根信息、适合语义理解、轻量化
- **缺点**：特殊字符支持差、必须加特殊Token、中文一般
- **适用**：**仅文本分类/情感分析/命名实体识别**（生成式任务不推荐）

3. T5系列（多语言/无空格语言王者）

- **优点**：语言无关、中文/日文等无空格语言最优、可逆解码
- **缺点**：词表偏小、长文本token数略多
- **适用**：翻译、跨语言任务、纯中文NLP、文本生成

4. LLaMA系列（代码/数学/长文本天花板）

- **优点**：代码结构优化、数字强制拆分（数学极强）、大词表长上下文友好
- **缺点**：模型权重门槛高、中文优化略逊于T5
- **适用**：代码生成、数学计算、编程助手、长文档处理

5. RoBERTa（鲁棒性文本理解）

- **优点**：BPE更稳定、训练效果优于原生BERT
- **缺点**：无特殊亮点、通用性不如GPT
- **适用**：企业级文本理解、需要高鲁棒性的NLP任务

---

### 选型规则

🔥通用首选方案

**选 GPT系列（cl100k_base/o200k_base）**

- 理由：兼容性拉满、无坑、支持所有字符、文档最全，新手/工程化首选。

🔥 纯中文NLP首选

**选 T5系列**

- 理由：原生支持无空格语言，中文token利用率最高，无乱码/拆分异常。

🔥 代码/数学/编程任务

**必选 LLaMA/CodeLlama**

- 理由：数字拆分、代码符号处理是行业最优，数学计算准确率显著更高。

🔥 文本分类/语义理解任务

**选 BERT 或 RoBERTa**

- 理由：专门为理解任务设计，轻量、效果稳定。

🔥 跨语言/多语言混合

**选 GPT-4o 或 T5**

- 理由：多语言token压缩率高、无OOV、解码无损。

🔥 资源受限/轻量部署

**选 GPT-2（r50k_base）**

- 理由：词表最小、计算开销低，效果依然够用。

---

### 总结

- **做通用大模型/多语言** → **GPT系列**

- **做纯中文NLP** → **T5系列**

- **做代码/数学** → **LLaMA系列**

- **做文本理解/分类** → **BERT/RoBERTa**

- **所有场景怕踩坑** → 直接用 **GPT分词器**

  

- **GPT**：全能通用，无短板，90%场景直接用
- **BERT**：只做文本理解，不做生成
- **T5**：中文/多语言最强
- **LLaMA**：代码/数学专用
- **RoBERTa**：高鲁棒性文本理解

