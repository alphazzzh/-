import logging
import json
import re
import hashlib
import pickle
import asyncio
from typing import TypedDict, Optional, Dict, Any, Tuple, List
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langgraph.graph import StateGraph, END

# ============= 新增离线验证工具导入 =============
try:
    from rbloom import Bloom
except ImportError:
    try:
        from pybloom_live import BloomFilter as Bloom
    except ImportError:
        Bloom = None

try:
    import spacy
    try:
        # 优化: 禁用不需要的组件，只保留 NER (用于实体识别)
        # 显式添加 sentencizer 用于分句
        nlp_en = spacy.load("en_core_web_sm", disable=['tagger', 'parser', 'attribute_ruler', 'lemmatizer'])
        nlp_en.add_pipe('sentencizer')
    except:
        nlp_en = None
    try:
        # 中文同理
        nlp_zh = spacy.load("zh_core_web_sm", disable=['tagger', 'parser', 'attribute_ruler', 'lemmatizer'])
        nlp_zh.add_pipe('sentencizer')
    except:
        nlp_zh = None
except ImportError:
    spacy = None
    nlp_en = None
    nlp_zh = None

try:
    import stdnum.isbn
    import stdnum.issn
    HAS_STDNUM = True
except ImportError:
    HAS_STDNUM = False

try:
    from gibberish_detector import detector as gibberish_detector
    HAS_GIBBERISH = True
except ImportError:
    HAS_GIBBERISH = False

# Optional: NLI (ONNX)
try:
    import numpy as np
    import onnxruntime as ort
    from transformers import AutoTokenizer
except Exception:
    np = None
    ort = None
    AutoTokenizer = None

# 导入sentencetransformer
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    HAS_EMBEDDING = True
except ImportError:
    HAS_EMBEDDING = False
    SentenceTransformer = None
    np = None


# ================= [新增模块] 离线拦截器与 NLTK 补丁 =================

# 1. 强制离线环境变量
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# 2. PyTorch 版本伪装
import torch
torch.__version__ = "2.6.0"

# 3. Transformers 路径拦截
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from config import AppConfig, LOCAL_SAFE_PATH

orig_config_load = AutoConfig.from_pretrained
orig_model_load = AutoModelForSequenceClassification.from_pretrained
orig_tok_load = AutoTokenizer.from_pretrained

def is_target_model(path):
    return any(k in str(path) for k in [
        "roberta-large", "MiniCheck", "lytang",
        "Bespoke", "bespokelabs"
    ])

def mocked_config_load(cls, path, *args, **kwargs):
    if is_target_model(path): kwargs['local_files_only'] = True
    return orig_config_load.__func__(cls, LOCAL_SAFE_PATH if is_target_model(path) else path, *args, **kwargs)

def mocked_model_load(cls, path, *args, **kwargs):
    if is_target_model(path): kwargs['local_files_only'] = True
    return orig_model_load.__func__(cls, LOCAL_SAFE_PATH if is_target_model(path) else path, *args, **kwargs)

def mocked_tok_load(cls, path, *args, **kwargs):
    if is_target_model(path): kwargs['local_files_only'] = True
    return orig_tok_load.__func__(cls, LOCAL_SAFE_PATH if is_target_model(path) else path, *args, **kwargs)

# 激活拦截
AutoConfig.from_pretrained = classmethod(mocked_config_load)
AutoModelForSequenceClassification.from_pretrained = classmethod(mocked_model_load)
AutoTokenizer.from_pretrained = classmethod(mocked_tok_load)

#---------------- vllm拦截 ----------------
try:
    import vllm
    orig_vllm_init = vllm.LLM.__init__

    def mocked_vllm_init(self, model, *args, **kwargs):
        if model == 'Bespoke-MiniCheck-7B' or 'Bespoke' in str(model):
            print(f"🛡️ [vLLM 拦截器] 正在将模型重定向到本地 -> {LOCAL_SAFE_PATH}")
            model = LOCAL_SAFE_PATH
            kwargs['trust_remote_code'] = True
            kwargs['dtype'] = "bfloat16"
            kwargs['gpu_memory_utilization'] = 0.6
        
        return orig_vllm_init(self, model, *args, **kwargs)

    vllm.LLM.__init__ = mocked_vllm_init
    print("✅ vLLM 拦截器已激活 (vLLM Interceptor Activated)")

except ImportError:
    print("⚠️ 未检测到 vLLM 库,跳过 vLLM 拦截 (可能正在使用 Torch 模式)")

# 4. NLTK 离线补丁
import nltk.tokenize
def apply_nltk_patch():
    pickle_path = os.path.expanduser("~/nltk_data/tokenizers/punkt/english.pickle")
    if not os.path.exists(pickle_path):
        pickle_path = "/root/nltk_data/tokenizers/punkt/english.pickle"
    
    if os.path.exists(pickle_path):
        try:
            with open(pickle_path, 'rb') as f:
                tokenizer = pickle.load(f)
            nltk.tokenize.sent_tokenize = lambda t, language='english': tokenizer.tokenize(t)
        except Exception as e:
            logging.warning(f"NLTK patch failed: {e}")
apply_nltk_patch()

try:
    from minicheck.minicheck import MiniCheck
except ImportError:
    MiniCheck = None

# ================= 1. 配置 (Configuration) =================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


llm = None
_MINICHECK_SCORER = None
_BLOOM_FILTER = None
_GIBBERISH_DETECTOR = None


# ================= Dynamic Few-Shot Selector (适配你的 JSON) =================

_FEW_SHOT_SELECTOR = None

# 修改 DynamicFewShotSelector 类如下：

class DynamicFewShotSelector:
    def __init__(self, examples_path, model_path_or_name):
        if not HAS_EMBEDDING:
            raise ImportError("请先安装依赖: pip install sentence-transformers onnxruntime")
            
        # ============ [修改点] 动态加载逻辑 ============
        if AppConfig.EMBEDDING_USE_ONNX:
            # 模式 A: 使用 ONNX (本地路径)
            self.model = OnnxBgeEmbedder(model_path_or_name)
        else:
            # 模式 B: 使用 PyTorch (本地路径 或 HuggingFace ID)
            logger.info(f"正在加载 PyTorch Embedding 模型: {model_path_or_name} ...")
            self.model = SentenceTransformer(model_path_or_name, device='cpu')
        # ============================================
        
        with open(examples_path, 'r', encoding='utf-8') as f:
            self.examples = json.load(f)
            
        logger.info(f"正在为 {len(self.examples)} 条判例构建索引...")
        
        # 1. 构建语料库
        self.corpus = []
        for ex in self.examples:
            # 适配你的 json 字段
            ref = ex.get("reference", "")
            q = ex.get("question", "")
            ans = ex.get("answer", "")
            # 拼接 Ref+Q+A 以获得最佳检索效果
            text_to_embed = f"Ref: {ref}\nQuestion: {q}\nAnswer: {ans}"
            self.corpus.append(text_to_embed)
            
        # 2. 计算向量 (encode 接口已统一)
        # 注意：SentenceTransformer 默认返回 Tensor 或 numpy，ONNX 返回 numpy
        self.embeddings = self.model.encode(self.corpus)
        
        # 确保转为 numpy 且归一化 (防止 SentenceTransformer 未归一化)
        if hasattr(self.embeddings, "detach"): # 如果是 Tensor
             self.embeddings = self.embeddings.detach().cpu().numpy()
             
        # 再次强制归一化 (双重保险)
        norm = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / (norm + 1e-9)
        
        logger.info("判例库索引构建完成！")

    def retrieve(self, current_ref, current_q, current_ans, k=2):
        """检索最相似的 k 个例子"""
        query = f"Ref: {current_ref}\nQuestion: {current_q}\nAnswer: {current_ans}"
        
        # 向量化查询
        query_vec = self.model.encode([query])
        
        # 确保格式统一
        if hasattr(query_vec, "detach"):
            query_vec = query_vec.detach().cpu().numpy()
            
        # 归一化
        query_vec = query_vec / (np.linalg.norm(query_vec, axis=1, keepdims=True) + 1e-9)
        
        # 计算相似度
        scores = np.dot(query_vec, self.embeddings.T)[0]
        top_indices = np.argsort(scores)[::-1][:k]
        
        return [self.examples[i] for i in top_indices]

# 修改加载函数，传入新的 PATH 配置
def _ensure_selector_loaded():
    global _FEW_SHOT_SELECTOR
    if not AppConfig.ENABLE_DYNAMIC_FEW_SHOT:
        return False
    
    if _FEW_SHOT_SELECTOR is not None:
        return True
        
    try:
        # 这里传入 AppConfig.EMBEDDING_MODEL_PATH (本地路径)
        _FEW_SHOT_SELECTOR = DynamicFewShotSelector(
            AppConfig.FEW_SHOT_EXAMPLES_PATH,
            AppConfig.EMBEDDING_MODEL_PATH 
        )
        return True
    except Exception as e:
        logger.error(f"Few-Shot Selector 初始化失败: {e}", exc_info=True)
        return False

# ================= [新增] ONNX Embedding 包装器 =================

class OnnxBgeEmbedder:
    """
    轻量级 ONNX 推理包装器，接口兼容 SentenceTransformer
    专门用于加载本地 bge-m3.onnx
    """
    def __init__(self, model_dir: str, model_filename: str = "model.onnx"):
        if not ort or not AutoTokenizer:
            raise ImportError("请安装 onnxruntime 和 transformers: pip install onnxruntime transformers")
        
        logger.info(f"正在加载本地 ONNX 模型: {model_dir} ...")
        
        # 1. 加载 Tokenizer (负责把文本转成 ID)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # 2. 加载 ONNX Session (负责推理)
        model_path = os.path.join(model_dir, model_filename)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX 模型文件未找到: {model_path}")
            
        # 使用 CPU 推理 (Embedding 通常很快，CPU 足够)
        providers = ['CPUExecutionProvider']
        # 如果你有 GPU 且装了 onnxruntime-gpu，可以把 'CUDAExecutionProvider' 放前面
        
        self.session = ort.InferenceSession(model_path, providers=providers)
        
    def encode(self, sentences: list, batch_size: int = 4, **kwargs):
        """
        兼容 SentenceTransformer 的 encode 接口
        """
        # 确保输入是列表
        if isinstance(sentences, str):
            sentences = [sentences]
            
        all_embeddings = []
        
        # 批量处理
        for i in range(0, len(sentences), batch_size):
            batch_texts = sentences[i : i + batch_size]
            
            # 1. Tokenization (BGE-M3 支持长文本，这里设 8192 或 1024 均可)
            inputs = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=1024, 
                return_tensors="np" # 直接返回 numpy 数组给 ONNX 用
            )
            
            # 2. 构造 ONNX 输入
            # 注意：BGE-M3 (XLM-RoBERTa) 通常只需要 input_ids 和 attention_mask
            ort_inputs = {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64)
            }
            
            # 3. ONNX 推理
            # output[0] 通常是 last_hidden_state (Batch, SeqLen, Dim)
            outputs = self.session.run(None, ort_inputs)
            last_hidden_state = outputs[0]
            
            # 4. Pooling & Normalization
            # BGE-M3 的 Dense Embedding 使用 CLS token (索引 0)
            embeddings = last_hidden_state[:, 0, :]
            
            # 归一化 (L2 Norm) - 这一步对于计算余弦相似度至关重要
            norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norm + 1e-9)
            
            all_embeddings.append(embeddings)
            
        if not all_embeddings:
            return np.array([])
            
        # 合并所有 batch
        return np.vstack(all_embeddings)

# ================= 稳定哈希函数 (必须与 build_filter.py 一致) =================
def stable_hash(val):
    """
    必须与 build_filter.py 中的哈希函数完全一致
    """
    if isinstance(val, str):
        val = val.encode('utf-8')
    h_int = int(hashlib.md5(val).hexdigest(), 16)
    if h_int >= 1 << 127:
        h_int -= 1 << 128
    return h_int


def _ensure_bloom_loaded():
    """延迟加载 Bloom Filter (带哈希函数修复)"""
    global _BLOOM_FILTER
    if not AppConfig.ENABLE_BLOOM_FILTER or Bloom is None:
        return False
    if _BLOOM_FILTER is not None:
        return True
    
    bloom_path = Path(AppConfig.BLOOM_FILTER_PATH)
    if not bloom_path.exists():
        logger.warning(f"Bloom filter file not found: {bloom_path}")
        return False
    
    try:
        # 【核心修复】: 传入 hash_func=stable_hash
        if 'rbloom' in str(Bloom):
            _BLOOM_FILTER = Bloom.load(str(bloom_path), hash_func=stable_hash)
        else:
            # pybloom_live 通常把 hash 存进去了,直接加载即可
            with open(bloom_path, 'rb') as f:
                _BLOOM_FILTER = pickle.load(f)
                
        logger.info(f"✅ Bloom Filter loaded from {bloom_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to load Bloom Filter: {e}")
        return False


def _ensure_gibberish_loaded():
    """延迟加载 Gibberish Detector"""
    global _GIBBERISH_DETECTOR
    if not AppConfig.ENABLE_GIBBERISH_CHECK or not HAS_GIBBERISH:
        return False
    if _GIBBERISH_DETECTOR is not None:
        return True
    
    try:
        _GIBBERISH_DETECTOR = gibberish_detector.Detector()
        logger.info("✅ Gibberish Detector loaded")
        return True
    except Exception as e:
        logger.error(f"Failed to load Gibberish Detector: {e}")
        return False


def _ensure_minicheck_loaded():
    global _MINICHECK_SCORER
    if not AppConfig.MINICHECK_ENABLED or MiniCheck is None:
        return False
    if _MINICHECK_SCORER is not None:
        return True
    try:
        _MINICHECK_SCORER = MiniCheck(
            model_name=AppConfig.MINICHECK_MODEL_NAME,
            enable_prefix_caching=AppConfig.MINICHECK_ENABLE_PREFIX_CACHING
        )
        logger.info(f"MiniCheck loaded successfully: {AppConfig.MINICHECK_MODEL_NAME}")
        return True
    except Exception as e:
        logger.error(f"MiniCheck Init Failed: {e}", exc_info=True)
        return False


# ================= 2. 智能预处理 (Smart Preprocessing) =================

def smart_extract(text: str) -> Tuple[str, Optional[str]]:
    """
    [升级版] 三级引用提取策略:
    Level 1: 显式标签 ([Reference]...)
    Level 2: 代码块劫持 (```...```)
    Level 3: 自然语言前缀 (Based on...)
    Level 4: 结构化兜底 (长文切分)
    """
    if not text:
        return "", None
    clean_text = text.strip()

    # --- Level 1: 显式标签匹配 (优先级最高) ---
    patterns_brackets = [
        (r"\[Reference\]", r"\[Question\]"),
        (r"Reference:", r"Question:"),
        (r"Context:", r"Query:"),
        (r"【参考资料】", r"【问题】"),
        (r"资料：", r"问题："),
    ]
    for ref_tag, q_tag in patterns_brackets:
        ref_match = re.search(ref_tag, clean_text, re.IGNORECASE)
        q_match = re.search(q_tag, clean_text, re.IGNORECASE)
        if ref_match and q_match:
            # 支持 Ref 在前 或 Ref 在后
            if ref_match.start() < q_match.start():
                q = clean_text[q_match.end():].strip()
                ref = clean_text[ref_match.end(): q_match.start()].strip()
                return q, ref
            else:
                q = clean_text[q_match.end(): ref_match.start()].strip()
                ref = clean_text[ref_match.end():].strip()
                return q, ref

    # --- Level 2: 代码块劫持 ---
    # 匹配三种常见标记：```, ''', ~~~，并利用 \1 确保首尾标记一致
    # findall 返回的是 [(标记, 内容), (标记, 内容)...] 的列表
    code_matches = re.findall(r"(```|'''|~~~)([\s\S]*?)\1", clean_text)
    
    if code_matches:
        # 找到内容最长的那一组匹配 (m[1] 是内容)
        longest_match = max(code_matches, key=lambda m: len(m[1]))
        delimiter = longest_match[0]   # 获取该段代码用的标记 (比如 ''')
        code_content = longest_match[1] # 获取中间的代码内容
        
        # 依然保持 50 字符的阈值判断
        if len(code_content) > 50:
            # 组装出完整的块字符串（标记 + 内容 + 标记），以便从原文中精确移除
            full_block_str = delimiter + code_content + delimiter
            
            # 从原文中移除这段代码
            q = clean_text.replace(full_block_str, "").strip()
            
            return q, code_content.strip()

    # --- Level 3: 自然语言触发词 (新增) ---
    nl_patterns = [
        r"^(?:Based on|According to|Given|Refer to|Read) (?:the following|the text|the article|the passage)[:,\s]",
        r"^Here is (?:a|the) (?:text|article|passage)[:,\s]",
        r"^(?:基于|根据|参考|阅读|依据)(?:以下|下文|上述)?(?:资料|内容|文章|段落|文本)?[:：,\s]",
    ]
    for pat in nl_patterns:
        if re.match(pat, clean_text, re.IGNORECASE):
            # 尝试寻找最后的问题分隔符
            splitters = [r"\nQuestion:", r"\nQuery:", r"\n问题：", r"\n任务："]
            for sp in splitters:
                parts = re.split(sp, clean_text, flags=re.IGNORECASE)
                if len(parts) >= 2:
                    return parts[-1].strip(), parts[0].strip()

            # 尾部切分法: 假设最后一段是问题
            segments = re.split(r'\n\s*\n', clean_text)
            if len(segments) >= 2:
                q = segments[-1].strip()
                ref = "\n".join(segments[:-1]).strip()
                if len(ref) > len(q) and len(ref) > 20:
                    return q, ref

    # --- Level 4: 结构化兜底 ---
    # 如果没有标签，但文本很长且结尾像问题
    if len(clean_text) > 300:
        lines = clean_text.split('\n')
        last_line = lines[-1].strip()
        if len(last_line) < 150 and (last_line.endswith('?') or "总结" in last_line):
             return last_line, "\n".join(lines[:-1]).strip()

    return clean_text, None


def clean_and_check_refusal(
    question: str, answer: str, strict_mode: bool
) -> Tuple[Optional[dict], str]:
    """Refusal 处理逻辑"""
    q_lower = (question or "").lower()

    code_triggers = [
        "write code", "function", "script", "javascript", "python", "java", "c++", "algorithm",
        "program", "implement", "develop", "html", "css", "sql", "query", "render", "draw",
        "canvas", "excel formula", "regex", "regular expression",
        "写代码", "函数", "脚本", "算法", "编程", "程序", "实现", "开发", "渲染", "绘制", "正则",
    ]
    code_triggers = [t.lower() for t in code_triggers]
    if any(t in q_lower for t in code_triggers):
        return None, answer

    scan = (answer or "")[: AppConfig.REFUSAL_SCAN_CHARS]

    patterns = [
        r"^\s*((?:as an ai|as a language model|i cannot|i can't|unable to|i don't have|i do not have|i don't know|i have no|my knowledge)[\s\S]{0,320}?)(?:\s*(?:however|but|here is|below is|i can provide)\b[\s\S]{0,120}.*)$",
        r"^\s*((?:我是ai|人工智能|语言模型|无法|不能|不知道|我没有|我无法|我不能)[\s\S]{0,320}?)(?:\s*(?:但是|不过|然而|以下是|我可以提供)\b[\s\S]{0,120}.*)$",
    ]

    CONTEXT_KEYWORDS = [
        "text", "image", "context", "input", "provide", "access", "browse", "internet",
        "real-time", "realtime", "file", "see", "passage", "reference", "given", "mentioned",
        "data", "article", "local date", "system time", "camera",
        "文本", "图片", "上下文", "输入", "提供", "访问", "浏览", "联网", "实时", "文件", "看",
        "文章", "参考", "给定", "提及", "资料", "本地时间", "系统时间",
    ]

    OPINION_KEYWORDS = [
        "opinion", "opinions", "belief", "beliefs", "feel", "personal view", "standpoint",
        "sentience", "consciousness",
        "观点", "看法", "立场", "个人认为", "信仰", "感受", "意识",
    ]

    clean_answer = answer or ""

    for pat in patterns:
        m = re.match(pat, scan, re.IGNORECASE | re.DOTALL)
        if not m:
            continue

        waste_words = m.group(1) or ""
        refusal_part = waste_words.lower()

        is_context_refusal = any(kw in refusal_part for kw in CONTEXT_KEYWORDS)
        is_opinion_refusal = any(kw in refusal_part for kw in OPINION_KEYWORDS)

        if is_opinion_refusal:
            cleaned = clean_answer.replace(waste_words, "", 1).strip()
            cleaned = re.sub(r"^(however|but|但是|不过|然而)[,,, \s]*", "", cleaned, flags=re.IGNORECASE).strip()
            return None, cleaned

        if is_context_refusal:
            if strict_mode:
                return (
                    {
                        "status": "FAIL",
                        "risk_level": "MEDIUM",
                        "reason": f"严格模式:检测到上下文/能力受限式拒答转折,疑似逻辑冲突或来源不可追溯: {refusal_part[:80]}...",
                    },
                    answer,
                )
            cleaned = clean_answer.replace(waste_words, "", 1).strip()
            cleaned = re.sub(r"^(however|but|但是|不过|然而)[,,, \s]*", "", cleaned, flags=re.IGNORECASE).strip()
            return None, cleaned

        if strict_mode:
            return (
                {
                    "status": "FAIL",
                    "risk_level": "LOW",
                    "reason": f"严格模式:检测到不明原因拒答转折,疑似逻辑冲突: {refusal_part[:80]}...",
                },
                answer,
            )
        cleaned = clean_answer.replace(waste_words, "", 1).strip()
        cleaned = re.sub(r"^(however|but|但是|不过|然而)[,,, \s]*", "", cleaned, flags=re.IGNORECASE).strip()
        return None, cleaned

    return None, clean_answer


# ================= 3. 原子化 Prompts =================

class RobustJsonParser(BaseOutputParser):
    def parse(self, text: str) -> dict:
        raw = text or ""
        clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        clean = clean.replace("```json", "").replace("```", "").strip()

        try:
            return json.loads(clean)
        except Exception:
            pass

        try:
            matches = re.findall(r"(\{[\s\S]*\})", clean, re.DOTALL)
            if matches:
                return json.loads(matches[-1])
        except Exception:
            pass

        return {
            "status": "FAIL",
            "reason": "Output parsing error (non-JSON response).",
            "raw": raw[:1500],
        }


def standardize_result(raw_data: dict) -> dict:
    data = {k.lower(): v for k, v in (raw_data or {}).items()}
    status = str(data.get("status", "FAIL")).upper()
    status = "PASS" if "PASS" in status else "FAIL"
    reason = str(data.get("step_3_verdict", data.get("reason", data.get("analysis", "No reason"))))
    risk = str(data.get("risk_level", "LOW")).upper()
    if risk not in {"LOW", "MEDIUM", "HIGH"}:
        risk = "LOW"
    return {"status": status, "risk_level": risk, "reason": reason, "trace": raw_data}


INTENT_PROMPT = """任务：分析用户指令的意图类型。
【用户指令】: {question}

请从以下 4 个类别中选择最匹配的一个，输出 JSON: {{"type": "CATEGORY"}}

1. CREATIVE (开放创作):
   - 关键词: 创作、写故事、写诗、假设、扮演、续写、邮件、文案。
   - 示例: "写一个科幻故事", "帮我写封请假条", "假设你是马斯克"。
2. CONSTRAINED (基于上文):
   - 强依赖给定的参考资料 (但注意：如果用户没有提供资料，绝对不要选这个！)。
3. CODE (代码编程):
   - 编写、解释或调试代码/SQL/正则。
4. QA (事实问答):
   - 询问客观事实、知识、概念定义。
   - 示例: "鲁迅是谁？", "苹果的股价是多少？", "推荐几部电影"。
"""

CONSTRAINED_AUDIT_PROMPT = """你是一个严苛的"事实核查法官"。
你的任务是基于【参考资料】,审判【模型回答】是否存在**任何形式的幻觉或不忠实**。

【参考资料】
{reference}

【用户问题】
{question}

【模型回答】
{answer}

请基于以下"三大铁律"进行审查。只要违反任何一条,即判为 FAIL:

1. **铁律一:严禁无中生有 (The "Not-Mentioned" Rule)**
   - **审查目标**:检查回答中是否出现了原文未提及的**具体实体**(人名、地名、机构)、**数值**(日期、百分比、金额)或**事件**。
   - **幻觉类型**:事实新增、细节编造、引用错配。

2. **铁律二:严禁逻辑扭曲 (The "Logic-Twist" Rule)**
   - **审查目标**:检查回答是否颠倒了原文的**因果关系**、**主被动关系**或**肯定/否定**。
   - **幻觉类型**:事实歪曲、因果补全。

3. **铁律三:严禁程度/范围漂移 (The "Scope-Drift" Rule)**
   - **审查目标**:检查回答是否将原文的"可能/部分"夸大为"一定/所有",或忽略了原文的前提条件。
   - **幻觉类型**:过度概括、语义漂移、选择性忽略。

请输出 JSON:
{{
    "analysis": "1. 实体核查: [有无新增]... 2. 逻辑核查: [有无矛盾]... 3. 程度核查: [有无夸大/范围错误]...",
    "status": "PASS" | "FAIL",
    "reason": "请明确指出违反了哪一条铁律"
}}
"""

AUGMENTED_CONSTRAINED_PROMPT = """你是一名资深的"事实仲裁法官"。
你的任务是基于【参考资料】,对【模型回答】进行终审。

我们使用初级自动化工具对回答进行了预扫描,并标记了一些疑似风险点。请参考以下**三大铁律**与**判例**进行最终裁决。

【参考资料】
{reference}

【用户问题】
{question}

【模型回答】
{answer}

【🔎 预扫描高亮提示】
(以下句子被工具标记为缺乏支撑,请重点复核,但不要盲从)
{nli_evidence}

### 审判法则 (The Iron Rules)
**只要违反任何一条，即判为 FAIL**：

1. **铁律一: 严禁无中生有 (The "Not-Mentioned" Rule)**
   - **红线**: 严禁引入原文未提及的**具体实体**(人名/头衔)、**数值**、**事件**或**动作**。
   - **例外**: 允许同义改写 (released -> launched) 和 常识性代词指代。
   - **豁免**: 如果模型回答是对原文的**总结(Summarization)**或**跨段落整合(Synthesis)**，只要核心事实不冲突，**允许**省略非关键细节，**允许**同义表述。不要仅仅因为“原文没原话”就判错。

2. **铁律二: 严禁逻辑扭曲 (The "Logic-Twist" Rule)**
   - **红线**: 严禁颠倒因果、主被动或肯定/否定关系，严禁时空错乱和数学计算错误。

3. **铁律三: 严禁程度漂移 (The "Scope-Drift" Rule)**
   - **红线**: 严禁将"可能"夸大为"一定"，或忽略前提条件。

### 判例示范 (Few-Shot Demonstrations)
学习以下两个案例：

{few_shot_examples}

---

【仲裁审查步骤】
1. **全文通读与消歧**: 首先通读全文,确认代词(他/它)指代的对象,并理解整段回答的逻辑链条。
2. **疑点语境复核**: 
   - 将"高亮提示"中的句子放回【参考资料】的**原始语境**中比对。
   - **关键判断**: 这是"无中生有"的错误?还是"同义转换"的合理改写?
3. **最终判决**:
   - 根据规则给出你的判断 (PASS 或 FAIL)。

输出 JSON:
{{
  "step_1_context": "简述全文核心逻辑...",
  "step_2_evidence_audit": "针对高亮句的复核分析(是误报还是实锤)...",
  "step_3_verdict": "最终判定 (PASS 或 FAIL)",
  "status": "PASS" | "FAIL",
  "reason": "..."
}}
"""

# ================= 核心修改: V5 融合版审计 Prompt =================
# 结合了"知识注入"的准确性 + "实体审计"的审判逻辑
KNOWLEDGE_RETRIEVAL_PROMPT = """你是一个百科全书。请根据你的内部知识，为回答用户的问题提供必要的知识支撑。

【问题】: {question}

请按以下策略输出：
1. **如果是事实类问题** (如时间、地点、人物、作品)：
   - 请直接列出关键事实（Key Facts）。
   - 示例："1994年上映；导演是罗伯特·泽米吉斯。"
2. **如果是解释/观点类问题** (如为什么、如何评价、原理解析)：
   - 请简述核心概念或公认的主流观点。
   - 示例："因为瑞利散射（Rayleigh scattering），短波长的蓝光更容易被散射..."

(保持客观，不要臆造，如果不知道就说不知道)
"""

FORENSIC_AUDIT_PROMPT = """你是一名铁面无私的"幻觉审判官"。
任务: 判断【模型回答】中是否包含**编造的实体**、**篡改的事实**或**错误的逻辑**。

为了辅助你的判断，我让你先在后台回忆了相关的正确知识（见【内部记忆快照】），请参考它，但核心是进行**可信度审计**。

【问题】: {question}

【内部记忆快照 (用于辅助核实)】: 
{internal_knowledge}

【模型回答 (被告)】: 
{answer}

【🔍 离线验证报告】
{offline_validation_report}

请执行以下**三步走**审计程序：

1. **场景定性 (Scenario Check)**:
   - 这是一个需要创造力的任务(写故事/代码)吗？ -> 如果是，且逻辑通顺，直接 PASS。
   - 这是一个严肃的事实问答吗？ -> 继续下一步。

2. **实体与细节审计 (Entity & Fact Audit)** - *核心环节*:
   - **扫描**: 找出回答中所有**具体的实体**(人名、作品名、年份、地点)。
   - **自我质询 (Self-Inquiry)**: 
     * 结合【内部记忆快照】问自己："这个细节(如1995年)与我记忆中的事实(如1994年)冲突吗？"
     * 结合【离线报告】问自己："这个生僻词(如'爬行卫矛')是真实存在的，还是模型在'一本正经地胡说八道'？"
   - **判定**: 只要发现一个**确凿的**事实冲突或编造实体 -> 标记为 **FAIL**。

3. **逻辑与指令审查 (Logic Audit)**:
   - 是否存在自相矛盾（前言不搭后语）？
   - 是否存在过度具体化（在模糊问题下编造精确到小数点的虚假数据）？
   - 是否主语偷换（问"张三"，答"李四的事迹"）？

**最终判决标准 (Verdict Criteria)**:
- **FAIL**: 
  - 包含**确凿的**事实错误（如年份错、张冠李戴）。
  - 包含极大概率是**编造的**细节（离线工具报警+内部记忆无此记录）。
  - 逻辑崩坏或答非所问。
- **PASS**: 
  - 所有关键事实（时间、地点、人物）均核实无误。
  - 或者是创作类任务且逻辑自洽。
  - 或者仅是表述方式不同（"首映于94年" vs "1994年"），但意思完全一致。

请输出 JSON:
{{
    "step_1_scenario": "...",
    "step_2_entity_audit": "分析关键实体：[实体A] -> [核实结果]...",
    "step_3_logic_audit": "...",
    "status": "PASS" | "FAIL", 
    "reason": "一针见血地指出哪里错了，或者为什么通过"
}}
"""

CODE_AUDIT_PROMPT = """角色:代码正确性与指令遵循审查员。

你需要判断助手的回答(包含代码)是否:
1. **满足需求**: 实现了用户请求的功能。
2. **无致命错误**: 没有明显的语法错误、虚构的库/函数、未定义的符号。
3. **无虚假执行**: 没有谎称"我已经运行了代码"或"读取了你的本地文件"。

【问题】: {question}
【回答】: {answer}

请判定以下布尔值:
- meets_requirement: 代码是否实现了用户请求?
- fatal_error: 是否存在致命阻断性错误?
- misleading_execution_claim: 是否存在虚假的执行/访问声明?

输出 JSON:
{{
  "meets_requirement": true | false,
  "fatal_error": true | false,
  "misleading_execution_claim": true | false,
  "reasons": ["简短理由1", "简短理由2"],
  "evidence": ["引用回答中的关键片段"]
}}
"""

CREATIVE_AUDIT_PROMPT = """你是一个创意写作审阅员。
任务：检查【开放式创作】的逻辑一致性。

【审查原则】
1. **允许虚构**：允许引入新细节、新设定，只要符合前文逻辑。
2. **禁止冲突**：禁止与用户给定的前提（背景）发生直接事实冲突。
3. **禁止拒答**：模型不应拒绝用户的创作请求。

【用户输入】: {question}
【模型创作】: {answer}

请输出 JSON: {{"analysis": "...", "status": "PASS" | "FAIL", "reason": "..."}}
"""

# ============= 离线验证器函数集 =============

def validate_with_bloom(text: str) -> List[Dict[str, Any]]:
    """
    [修正版] 使用 Bloom Filter 检测不存在的实体 (支持中英双语)
    """
    if not _ensure_bloom_loaded():
        return []
    
    findings = []
    
    # 策略: 收集需要运行的 SpaCy 模型
    docs_to_process = []
    if AppConfig.ENABLE_SPACY_NER:
        # 如果英文模型可用，跑一遍
        if nlp_en: 
            try:
                docs_to_process.append(nlp_en(text))
            except Exception:
                pass # 忽略模型报错
        
        # 【核心修复】如果中文模型可用，也跑一遍！
        if nlp_zh: 
            try:
                docs_to_process.append(nlp_zh(text))
            except Exception:
                pass
    
    seen_entities = set() # 用于去重，防止同一个实体被两个模型都抓出来报错两次

    for doc in docs_to_process:
        # 提取核心实体类型
        entities = [(ent.text, ent.label_) for ent in doc.ents 
                   if ent.label_ in {"PERSON", "ORG", "GPE", "WORK_OF_ART", "EVENT"}]
        
        for entity_text, entity_type in entities:
            # 清洗一下实体文本
            clean_text = entity_text.strip()
            
            # 1. 避免重复处理
            if clean_text in seen_entities: 
                continue 
            seen_entities.add(clean_text)
            
            # 2. 过滤掉太短的实体 (中文单字容易误报，如"李"、"王")
            if len(clean_text) < 2:
                continue

            # 3. Bloom Filter 查证
            # 注意：Bloom Filter 里的中文通常是简体，不需要额外处理，直接查
            if clean_text not in _BLOOM_FILTER:
                findings.append({
                    "type": "unknown_entity",
                    "entity": clean_text,
                    "entity_type": entity_type,
                    "risk": "HIGH",
                    "reason": f"实体 '{clean_text}' ({entity_type}) 不在知识库中，可能是幻觉"
                })
    
    return findings


def validate_isbn_doi(text: str) -> List[Dict[str, Any]]:
    """验证 ISBN/DOI 等标准格式"""
    if not HAS_STDNUM:
        return []
    
    findings = []
    
    # ISBN 检测
    isbn_pattern = r'\bISBN[:\s-]*([0-9\-Xx]{10,17})\b'
    for match in re.finditer(isbn_pattern, text, re.IGNORECASE):
        isbn_candidate = match.group(1).replace('-', '').replace(' ', '')
        try:
            stdnum.isbn.validate(isbn_candidate)
        except Exception:
            findings.append({
                "type": "invalid_isbn",
                "value": match.group(0),
                "risk": "HIGH",
                "reason": f"ISBN '{match.group(0)}' 校验位错误,可能是编造的"
            })
    
    # DOI 格式检测
    doi_pattern = r'\b(10\.\d{4,}/[^\s]+)'
    for match in re.finditer(doi_pattern, text):
        doi = match.group(1)
        if re.search(r'[<>"\{\}|\\^`\[\]]', doi):
            findings.append({
                "type": "invalid_doi",
                "value": doi,
                "risk": "MEDIUM",
                "reason": f"DOI '{doi}' 包含非法字符,格式可疑"
            })
    
    return findings


def check_gibberish(text: str) -> Optional[Dict[str, Any]]:
    """检测无意义文本/乱码"""
    if not _ensure_gibberish_loaded():
        return None
    
    try:
        is_gibberish = _GIBBERISH_DETECTOR.is_gibberish(text)
        if is_gibberish:
            return {
                "type": "gibberish_detected",
                "risk": "HIGH",
                "reason": "检测到大量无意义文本或乱码,模型可能已崩坏"
            }
    except Exception as e:
        logger.warning(f"Gibberish detection failed: {e}")
    
    return None


def check_future_dates(text: str) -> List[Dict[str, Any]]:
    """检测不合理的未来日期"""
    findings = []
    current_year = 2026
    
    year_pattern = r'\b(20[2-9][0-9]|2[1-9][0-9]{2})\b'
    for match in re.finditer(year_pattern, text):
        year = int(match.group(1))
        if year > current_year + 1:
            context_start = max(0, match.start() - 50)
            context_end = min(len(text), match.end() + 50)
            context = text[context_start:context_end].lower()
            
            past_indicators = ["was", "were", "had", "did", "已", "曾", "过去", "当时"]
            if any(indicator in context for indicator in past_indicators):
                findings.append({
                    "type": "future_date_in_past_context",
                    "value": match.group(0),
                    "risk": "HIGH",
                    "reason": f"年份 {year} 是未来时间,但语境暗示已发生"
                })
    
    return findings


def run_offline_validation(text: str) -> Dict[str, Any]:
    """运行所有离线验证器,返回汇总报告"""
    all_findings = []
    
    # 1. Gibberish 检测 (最快,优先级最高)
    if AppConfig.ENABLE_GIBBERISH_CHECK:
        gibberish_result = check_gibberish(text)
        if gibberish_result:
            return {
                "critical_issue": gibberish_result,
                "findings": [gibberish_result],
                "recommendation": "IMMEDIATE_FAIL"
            }
    
    # 2. Bloom Filter 实体验证
    if AppConfig.ENABLE_BLOOM_FILTER:
        bloom_findings = validate_with_bloom(text)
        all_findings.extend(bloom_findings)
    
    # 3. ISBN/DOI 校验
    if AppConfig.ENABLE_ISBN_CHECK:
        format_findings = validate_isbn_doi(text)
        all_findings.extend(format_findings)
    
    # 4. 未来日期检测
    date_findings = check_future_dates(text)
    all_findings.extend(date_findings)
    
    # 汇总风险等级
    high_risk_count = sum(1 for f in all_findings if f.get("risk") == "HIGH")
    
    recommendation = "NO_ISSUE"
    if high_risk_count >= 2:
        recommendation = "STRONG_FAIL"
    elif high_risk_count == 1:
        recommendation = "SUSPICIOUS"
    elif all_findings:
        recommendation = "MINOR_CONCERN"
    
    return {
        "findings": all_findings,
        "high_risk_count": high_risk_count,
        "total_findings": len(all_findings),
        "recommendation": recommendation
    }


def format_offline_validation_report(result: Dict[str, Any]) -> str:
    """将离线验证器的结果字典转换为 Prompt 可读的文本报告"""
    if not result or not result.get("findings"):
        return "【离线验证状态】: 通过 (未发现明显异常)"
    
    lines = ["【离线验证警报】: ⚠️ 发现潜在风险,请重点排查以下项目:"]
    
    findings = result.get("findings", [])
    for i, f in enumerate(findings, 1):
        f_type = f.get("type", "unknown")
        reason = f.get("reason", "")
        risk = f.get("risk", "LOW")
        
        icon = "❌" if risk == "HIGH" else "⚠️"
        lines.append(f"{i}. {icon} [{f_type}]: {reason}")
        
    lines.append("\n(请在终审判决中充分考虑上述验证结果,如果实体确实不存在,请判 FAIL)")
    return "\n".join(lines)


def format_nli_evidence_for_prompt(nli_results: List[dict]) -> str:
    """格式化 NLI 证据"""
    if not nli_results:
        return "无 NLI 风险句检测到。"

    lines = []
    for idx, item in enumerate(nli_results):
        sent = item.get("sentence", "").strip()
        prob = item.get("prob", 0.0)
        
        if prob > 0.9:
            risk_level = "高"
        elif prob > 0.5:
            risk_level = "中"
        else:
            risk_level = "低"
        
        lines.append(f"{idx+1}. 风险句: \"{sent}\" (风险等级: {risk_level})")

    return "\n".join(lines)


# ================= Graph Construction =================

class GraphState(TypedDict):
    question: str
    answer: str
    reference: Optional[str]
    intent: str
    nli: Optional[List[dict]]
    offline_validation: Optional[Dict[str, Any]]
    final_result: dict

parser = RobustJsonParser()


async def intent_node(state: GraphState):
    """
    三级防御路由节点:
    1. 物理锁: 有 Ref -> CONSTRAINED
    2. 关键词锁: 代码/创意词 -> CODE/CREATIVE
    3. LLM 路由: 处理剩余模糊情况
    """
    q = state.get("question", "") or ""
    ql = q.lower()
    ref = state.get("reference")

    # 清洗 ref
    if ref and len(str(ref).strip()) < 5:
        ref = None

    # --- 防御层 1: 物理锁 (Reference) ---
    # 只要有资料，强制走 CONSTRAINED
    if ref:
        return {"intent": "CONSTRAINED"}

    # --- 防御层 2: 关键词锁 (Heuristics) ---

    # 2.1 代码关键词
    code_kws = [
        "code", "function", "script", "program", "sql", "regex", "python", "java",
        "写代码", "函数", "脚本", "编程", "正则"
    ]
    if any(k in ql for k in code_kws):
        return {"intent": "CODE"}

    # 2.2 创意关键词 (新增)
    # 这些词出现时，直接跳过 LLM，防止被误判为 QA(Forensic) 导致误杀
    creative_kws = [
        "story", "poem", "novel", "fiction", "joke", "script", "act as", "pretend", "imagine",
        "email", "letter", "essay", "lyrics", "parody",
        "故事", "小说", "诗", "笑话", "剧本", "扮演", "假设", "想象", "编造", 
        "邮件", "信", "作文", "歌词", "续写", "扩写", "拟人"
    ]
    
    # 加question关键字排除
    question_indicators = ["为什么", "为何", "what", "why", "how", "who", "哪位", "哪个", "解释", "含义", "是啥", "是什么", "mean"]
    is_question = any(idx in ql for idx in question_indicators) or "?" in q or "？" in q

    if any(k in ql for k in creative_kws) and not is_question:
        return {"intent": "CREATIVE"}

    # --- 防御层 3: LLM 语义裁决 ---
    try:
        chain = ChatPromptTemplate.from_template(INTENT_PROMPT) | llm | parser
        res = await chain.ainvoke({"question": q})
        intent = str(res.get("type", "QA")).upper()

        # 兜底 A: LLM 误判 CONSTRAINED 但没 Ref -> 降级为 QA
        if intent == "CONSTRAINED":
            logger.warning(f"Router predicted CONSTRAINED but no ref. Forcing QA.")
            intent = "QA"

        # 兜底 B: 归一化
        valid_intents = {"CREATIVE", "CODE", "QA", "CONSTRAINED"}
        if intent not in valid_intents:
            intent = "QA"

        return {"intent": intent}

    except Exception as e:
        logger.warning(f"Intent router error: {e}, defaulting to QA")
        return {"intent": "QA"}


def router_node(state: GraphState):
    intent = state.get("intent", "QA")
    valid_intents = {"CONSTRAINED", "CODE", "QA", "CREATIVE", "GENERAL"}
    if intent not in valid_intents:
        logger.warning(f"Unknown intent: {intent}, defaulting to QA")
        return "QA"
    # 兼容旧代码: GENERAL -> QA
    if intent == "GENERAL":
        return "QA"
    return intent


def _invoke_with_retry(chain, inputs: dict, retries: int = 1) -> dict:
    """解析失败重试"""
    last = None
    for _ in range(retries + 1):
        last = chain.invoke(inputs)
        if isinstance(last, dict):
            reason = str(last.get("reason", "")).lower()
            status = str(last.get("status", "")).upper()
            if status == "FAIL" and "parsing error" in reason:
                continue
        break
    return last if isinstance(last, dict) else {"status": "FAIL", "reason": "Chain returned non-dict."}


async def _ainvoke_with_retry(chain, inputs: dict, retries: int = 1) -> dict:
    """异步解析失败重试"""
    last = None
    for _ in range(retries + 1):
        last = await chain.ainvoke(inputs)
        if isinstance(last, dict):
            reason = str(last.get("reason", "")).lower()
            status = str(last.get("status", "")).upper()
            if status == "FAIL" and "parsing error" in reason:
                continue
        break
    return last if isinstance(last, dict) else {"status": "FAIL", "reason": "Chain returned non-dict."}


async def forensic_chk(state: GraphState):
    question = state.get("question", "")
    answer = state.get("answer", "")
    
    # --- Parallel Task Launch ---
    
    # Task A: Start Knowledge Injection (Network I/O)
    # We use a helper coroutine or direct ainvoke to run this concurrently
    async def fetch_knowledge():
        try:
            from langchain_core.messages import HumanMessage
            k_msg = KNOWLEDGE_RETRIEVAL_PROMPT.format(question=question)
            # Use ainvoke for async
            raw_k_res = await llm.ainvoke([HumanMessage(content=k_msg)])
            k_text = raw_k_res.content.strip()
            if len(k_text) > 600:
                k_text = k_text[:600] + "..."
            return k_text
        except Exception as e:
            logger.warning(f"Knowledge injection failed: {e}")
            return "无内部知识"

    # Launch Task A
    knowledge_task = asyncio.create_task(fetch_knowledge())

    # Task B: Run Offline Validation (CPU Bound)
    # Since this is local CPU work, it runs immediately on the main thread.
    # While it runs, the LLM request (Task A) is waiting for network response.
    offline_result = run_offline_validation(answer)
    
    # Check for critical offline failure
    if offline_result.get("recommendation") == "IMMEDIATE_FAIL":
        # Cancel the LLM task if we fail early to save tokens/time
        knowledge_task.cancel()
        critical = offline_result.get("critical_issue", {})
        return {"final_result": {
            "status": "FAIL", 
            "risk_level": "HIGH", 
            "reason": f"离线验证熔断: {critical.get('reason')}",
            "trace": {"offline": offline_result}
        }}

    # Task A Join: Await the knowledge result
    generated_knowledge = await knowledge_task

    # --- Step 3: Call V5 Audit Prompt ---
    validation_report = format_offline_validation_report(offline_result)
    
    inputs = {
        "question": question,
        "answer": answer,
        "offline_validation_report": validation_report,
        "internal_knowledge": generated_knowledge # 作为"记忆快照"传入
    }
    
    # 使用更新后的 FORENSIC_AUDIT_PROMPT
    chain = ChatPromptTemplate.from_template(FORENSIC_AUDIT_PROMPT) | llm | parser
    # Use async helper
    raw = await _ainvoke_with_retry(chain, inputs, retries=1)
    
    result = standardize_result(raw)
    
    if "trace" not in result: result["trace"] = {}
    result["trace"]["generated_knowledge"] = generated_knowledge
    result["trace"]["offline_validation"] = offline_result
    
    return {"final_result": result}


# 修改 enhanced_forensic.py 中的 constrained_chk 函数

async def constrained_chk(state: GraphState):
    """仲裁节点:根据是否有 MiniCheck 证据,动态分流 Prompt"""
    inputs = {
        "question": state["question"], 
        "answer": state["answer"], 
        "reference": state.get("reference", "N/A")
    }
    
    evidence_list = state.get("nli", [])
    
    # ================= [新增] 动态判例检索逻辑 =================
    few_shot_text = "(暂无相关判例，请严格基于铁律判罚)"
    
    # 只有当 NLI 报警了，或者你希望全量都加判例时才检索
    # 这里建议: 只要开启了 ENABLE_DYNAMIC_FEW_SHOT 就检索
    if AppConfig.ENABLE_DYNAMIC_FEW_SHOT and _ensure_selector_loaded():
        try:
            # 检索 2 个最相似的例子
            # 传入 Ref, Q, A 确保全方位匹配
            # Note: _FEW_SHOT_SELECTOR.retrieve is synchronous (local embedding). 
            # In a true high-concurrency setup, we might wrap this in loop.run_in_executor, 
            # but for now, it's fast enough or acceptable to block briefly.
            examples = _FEW_SHOT_SELECTOR.retrieve(
                inputs['reference'], 
                inputs['question'], 
                inputs['answer'], 
                k=2
            )
            
            # 格式化检索到的例子
            formatted_exs = []
            for i, ex in enumerate(examples, 1):
                # 读取你的 JSON 字段
                # 注意: 你的 json 用的是 'label' 而不是 'status'
                label = ex.get('label', 'UNKNOWN') 
                reason = ex.get('reason', '无')
                # 截取一部分内容展示，防止 prompt 过长
                ref_text = ex.get('reference', '')[:150] + "..."
                q_text = ex.get('question', '')
                a_text = ex.get('answer', '')
                
                formatted_exs.append(
                    f"**案例 {i} [{label}]**:\n"
                    f"- 参考: {ref_text}\n"
                    f"- 问题: {q_text}\n"
                    f"- 回答: {a_text}\n"
                    f"- 判决: {label}\n"
                    f"- 理由: {reason}"
                )
            
            few_shot_text = "\n\n".join(formatted_exs)
            logger.info(f"已注入 {len(examples)} 条动态判例")
            
        except Exception as e:
            logger.error(f"Dynamic Few-Shot 检索失败: {e}", exc_info=True)
            
    # ================= End of Dynamic Logic =================

    if not evidence_list:
        logger.info("MiniCheck passed or disabled. Using STANDARD prompt.")
        # 如果你想给普通模式也加 Few-Shot，可以把 few_shot_examples 塞进 CONSTRAINED_AUDIT_PROMPT
        # 但目前我们只改了 AUGMENTED 版本
        prompt_template = ChatPromptTemplate.from_template(CONSTRAINED_AUDIT_PROMPT)
    else:
        logger.info(f"MiniCheck risks found ({len(evidence_list)}). Using AUGMENTED prompt.")
        evidence_str = format_nli_evidence_for_prompt(evidence_list)
        
        inputs["nli_evidence"] = evidence_str
        # 【关键】注入 few_shot_examples
        inputs["few_shot_examples"] = few_shot_text
        
        prompt_template = ChatPromptTemplate.from_template(AUGMENTED_CONSTRAINED_PROMPT)
    
    chain = prompt_template | llm | parser
    raw = await _ainvoke_with_retry(chain, inputs, retries=1)
    
    return {"final_result": standardize_result(raw)}


def _to_bool(x):
    if isinstance(x, bool):
        return x
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"true", "yes", "y", "1"}:
            return True
        if s in {"false", "no", "n", "0"}:
            return False
    return None


def _finalize_code_audit(raw: dict) -> dict:
    if not isinstance(raw, dict):
        return {"status": "FAIL", "risk_level": "LOW", "reason": "Code audit returned non-dict."}

    d = {k.lower(): v for k, v in raw.items()}
    meets = _to_bool(d.get("meets_requirement"))
    fatal = _to_bool(d.get("fatal_error"))
    mislead = _to_bool(d.get("misleading_execution_claim"))

    if meets is None and fatal is None and mislead is None:
        return raw

    is_fail = (fatal is True) or (mislead is True) or (meets is False)
    raw["status"] = "FAIL" if is_fail else "PASS"

    if "risk_level" not in d:
        if is_fail and (fatal is True or mislead is True):
            raw["risk_level"] = "HIGH"
        elif is_fail:
            raw["risk_level"] = "MEDIUM"
        else:
            raw["risk_level"] = "LOW"

    if "reason" not in d:
        reasons = d.get("reasons")
        if isinstance(reasons, list) and reasons:
            raw["reason"] = "; ".join(str(r) for r in reasons[:3])
        else:
            if fatal is True:
                raw["reason"] = "Code contains an obvious fatal issue."
            elif mislead is True:
                raw["reason"] = "Misleading execution/access claim detected."
            elif meets is False:
                raw["reason"] = "Code does not meet the user's requirement."
            else:
                raw["reason"] = "Code audit passed."

    return raw


async def code_chk(state: GraphState):
    chain = ChatPromptTemplate.from_template(CODE_AUDIT_PROMPT) | llm | parser
    raw = await _ainvoke_with_retry(chain, state, retries=1)
    raw = _finalize_code_audit(raw)
    return {"final_result": standardize_result(raw)}


async def creative_chk(state: GraphState):
    """创意类不查 Bloom Filter，只查逻辑自洽"""
    chain = ChatPromptTemplate.from_template(CREATIVE_AUDIT_PROMPT) | llm | parser
    raw = await _ainvoke_with_retry(chain, state, retries=1)
    return {"final_result": standardize_result(raw)}


# ================= NLI Sentence Probe =================

def _split_sentences_mixed(text: str) -> List[str]:
    """
    [升级版] 优先使用 SpaCy 进行语义分句，回退到正则
    """
    if not text:
        return []
    t = text.strip()
    
    # 1. 尝试使用 SpaCy (中英文兼容)
    # 英文优先
    if nlp_en:
        try:
            doc = nlp_en(t)
            # 过滤掉过短的碎片 (比如 "Yes.")，避免 NLI 误判，但保留 meaningful 的短句
            return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 3]
        except Exception as e:
            logger.warning(f"SpaCy EN split failed: {e}")

    # 中文备选
    if nlp_zh:
        try:
            doc = nlp_zh(t)
            return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 1]
        except Exception:
            pass

    # 2. 正则兜底 (保持原有逻辑作为 Fallback)
    # 修复: 避免把 "Reference 1." 这种引用切断
    t = re.sub(r'(Ref|Fig|No|Vol)\.\s+', r'\1<DOT> ', t)
    
    parts = re.split(r'(?<=[.!?。！？])\s+', t)
    
    out = []
    for p in parts:
        p = p.replace("<DOT>", ".").strip()
        if len(p) > 5: # 稍微提高一点最小长度阈值
            out.append(p)
            
    return out


def _is_meta_sentence(text: str) -> bool:
    """精细化元句过滤"""
    t = text.strip().lower()
    
    meta_phrases = [
        "based on the provided", "based on the given", "according to the passage",
        "answer to the question", "context provided", "given passages",
        "per passage", "refer to passage", "see passage", "described in passage",
        "reference passage", "from passage", "in passage",
        "根据提供的", "根据上文", "基于参考资料", "作为一个ai", "语言模型",
    ]
    if any(p in t for p in meta_phrases):
        return True

    if re.match(r'^[\(\[].{0,40}[\)\]][\.\s]*', t):
        return True
        
    if re.match(r'^(per\s+)?(passage|source|reference)\s+\d+(?:\s+(and|&)\s+\d+)?[\.:\s]*', t):
        return True
    
    return False


async def minicheck_node(state: GraphState):
    """
    [升级版] 双阈值漏斗机制 (Dual-Threshold Funnel)
    策略:
    1. Score > 0.7 -> FAST FAIL (高精度拦截，零 LLM 开销)
    2. Score > 0.02 -> NEEDS AUDIT (高召回筛选，交给 LLM 辩护)
    3. Score <= 0.02 -> SAFE (直接放行)
    """
    ref = state.get("reference", "")
    ans = state.get("answer", "")
    
    # 如果没有 Reference，或者 MiniCheck 未加载，直接跳过 (视为无风险)
    if not ref or len(str(ref)) < 10 or not _ensure_minicheck_loaded(): 
        return {"nli_verdict": "SAFE", "nli": []}
    
    # 1. 分句
    valid_sents = [s for s in _split_sentences_mixed(ans) if not _is_meta_sentence(s)]
    if not valid_sents: 
        return {"nli_verdict": "SAFE", "nli": []}

    try:
        # 2. 批量推理
        docs = [ref] * len(valid_sents)
        # Note: In a fully async pipeline, we could offload this to a thread,
        # but for now we keep it blocking as it's the primary task of this node.
        pred_label, probs, _, _ = _MINICHECK_SCORER.score(docs=docs, claims=valid_sents)
        
        # 3. 收集证据与最大风险
        evidence = []
        max_hallu_score = 0.0
        
        for i, prob in enumerate(probs):
            # 处理 list/float 格式差异
            prob_val = float(prob[0]) if isinstance(prob, (list, tuple, np.ndarray)) else float(prob)
            
            hallu_score = 1.0 - prob_val # 幻觉概率
            
            if hallu_score > max_hallu_score:
                max_hallu_score = hallu_score
            
            # 只要进入黄色预警区 (>0.02)，就记录案底，供 LLM 参考
            if hallu_score > 0.02:
                evidence.append({
                    "sentence": valid_sents[i],
                    "prob": hallu_score,
                    "risk": "HIGH" if hallu_score > 0.6 else "MEDIUM"
                })

        logger.info(f"🔍 MiniCheck Max Risk: {max_hallu_score:.4f} | Evidence Count: {len(evidence)}")

        # === 4. 漏斗分流逻辑 (核心修改) ===

        # 【红色区间】: 铁证如山 -> 直接判死刑
        if max_hallu_score > 0.7:
            logger.info("🛑 Fast-Fail triggered! Skipping LLM.")
            return {
                "nli_verdict": "FAST_FAIL",
                "nli": evidence,
                # 直接生成 Final Result
                "final_result": {
                    "status": "FAIL",
                    "risk_level": "HIGH",
                    "reason": f"NLI模型检测到确凿的幻觉 (置信度 {max_hallu_score:.2f})，触发快速拦截。",
                    "trace": {"nli_score": max_hallu_score}
                }
            }
            
        # 【黄色区间】: 疑罪从有 -> 交给 LLM 听证
        elif max_hallu_score > 0.02:
            return {
                "nli_verdict": "NEEDS_AUDIT",
                "nli": evidence
            }
            
        # 【绿色区间】: 安全 -> 直接通过
        else:
            return {
                "nli_verdict": "SAFE",
                "nli": [],
                # 直接生成 Final Result (Pass)
                "final_result": {
                    "status": "PASS",
                    "risk_level": "LOW",
                    "reason": "经NLI模型交叉验证，未发现明显事实冲突。",
                    "trace": {"nli_score": max_hallu_score}
                }
            }
        
    except Exception as e:
        logger.error(f"MiniCheck Error: {e}", exc_info=True)
        # 出错时默认回退到 LLM 检查，保证安全性
        return {"nli_verdict": "NEEDS_AUDIT", "nli": []}

# ================= MiniCheck Router 函数 =================
def minicheck_router(state: GraphState):
    verdict = state.get("nli_verdict", "NEEDS_AUDIT")
    
    if verdict == "NEEDS_AUDIT":
        # 只有存疑时，才去跑 LLM (constrained_chk)
        return "constrained_chk"
    else:
        # FAST_FAIL 或 SAFE 都已经生成了 final_result，直接结束
        return END


# ================= Workflow =================

workflow = StateGraph(GraphState)
workflow.add_node("intent_classifier", intent_node)
workflow.add_node("constrained_chk", constrained_chk)
workflow.add_node("forensic_chk", forensic_chk)
workflow.add_node("code_chk", code_chk)
workflow.add_node("creative_chk", creative_chk)
workflow.add_node("minicheck_node", minicheck_node)

workflow.set_entry_point("intent_classifier")

# 入口路由
workflow.add_conditional_edges(
    "intent_classifier",
    router_node,
    {
        "CONSTRAINED": "minicheck_node", # 1. 有 Reference -> 进 NLI
        "CODE": "code_chk",
        "QA": "forensic_chk",
        "CREATIVE": "creative_chk",
        "GENERAL": "forensic_chk"
    },
)

# 【修改点】NLI 节点的条件边
workflow.add_conditional_edges(
    "minicheck_node",
    minicheck_router,
    {
        "constrained_chk": "constrained_chk", # 存疑 -> LLM
        END: END                              # 确信 -> 结束
    }
)

# 其他节点直接结束
workflow.add_edge("constrained_chk", END)
workflow.add_edge("code_chk", END)
workflow.add_edge("forensic_chk", END)
workflow.add_edge("creative_chk", END)

app = workflow.compile()


# ================= API Wrapper =================

class RemoteHTTPGen:
    def __init__(self, host, model_name, timeout=120.0, api_key=None):
        global llm
        base_url = f"http://{host}/v1" if not host.startswith("http") else f"{host}/v1"
        AppConfig.API_BASE = base_url
        AppConfig.MODEL_NAME = model_name
        llm = ChatOpenAI(
            base_url=base_url,
            api_key=api_key or "sk-x",
            model=model_name,
            temperature=AppConfig.TEMPERATURE,
            timeout=timeout,
        )


def run_fast_validation(gen, question: str, raw_answer: str, **kwargs) -> Dict[str, Any]:
    """主入口函数:运行完整的验证流程"""
    global llm
    if llm is None:
        llm = ChatOpenAI(
            base_url=AppConfig.API_BASE,
            api_key=AppConfig.API_KEY or "sk-x",
            model=AppConfig.MODEL_NAME,
            temperature=AppConfig.TEMPERATURE,
        )

    # Step 1: refusal 检测/清洗
    pre_result, final_answer = clean_and_check_refusal(
        question, raw_answer, AppConfig.STRICT_REFUSAL_CHECK
    )
    if pre_result:
        out = standardize_result(pre_result)
        out["trace"] = out.get("trace", {})
        out["trace"]["intent"] = "PRE_REFUSAL_BLOCK"
        return out

    # Step 2: 准备输入(提取 reference)
    ref_input = kwargs.get("reference") or kwargs.get("knowledge")
    final_q, extracted_ref = smart_extract(question)
    final_ref = ref_input if ref_input else extracted_ref

    # Step 3: LLM/NLI 图审计 (Async Execution)
    try:
        # 使用 asyncio.run 运行异步图
        state = asyncio.run(app.ainvoke({
            "question": final_q, 
            "answer": final_answer, 
            "reference": final_ref
        }))
        
        final = state.get("final_result", {
            "status": "FAIL", 
            "risk_level": "LOW", 
            "reason": "Graph execution error"
        })
        
        if "trace" not in final or not isinstance(final["trace"], dict):
            final["trace"] = {}
        final["trace"]["intent"] = state.get("intent")
        if state.get("nli") is not None:
            final["trace"]["nli_detections"] = len(state.get("nli", []))
            final["trace"]["nli_details"] = state.get("nli", [])
        
        return final
        
    except Exception as e:
        logger.error(f"Graph execution error: {e}", exc_info=True)
        return {
            "status": "FAIL", 
            "risk_level": "LOW", 
            "reason": f"Graph execution failed: {str(e)}", 
            "trace": {"intent": "GRAPH_ERROR", "error": str(e)}
        }