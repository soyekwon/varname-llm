import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Iterable, Union
import codecs
from charset_normalizer import from_bytes

from tqdm import tqdm

# 맨 위 imports 근처에 추가
from pathlib import Path


#   pip install python-docx transformers datasets accelerate peft bitsandbytes scikit-learn
try:
    from docx import Document as DocxDocument
    from docx.document import Document as _Doc
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import Table
    from docx.text.paragraph import Paragraph
except Exception as e:
    DocxDocument = None

import datasets
from datasets import load_dataset

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling, BitsAndBytesConfig,
)

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except Exception:
    LoraConfig = None
    get_peft_model = None
    prepare_model_for_kbit_training = None


# =========================
# Helpers: DOCX block walk
# =========================

def iter_block_items(doc: _Doc) -> Iterable[Union[Paragraph, Table]]:
    """
    Iterate over paragraphs and tables in document order.
    """
    if doc is None:
        return
    body = doc.element.body
    for child in body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, doc)
        elif isinstance(child, CT_Tbl):
            yield Table(child, doc)


def para_text(p: Paragraph) -> str:
    t = p.text.strip()
    return t if t else ""


def table_text(t: Table) -> str:
    # Flatten table rows to tab-joined lines; skip all-empty rows
    lines = []
    for row in t.rows:
        cells = [c.text.strip() for c in row.cells]
        cells = [c for c in cells if c]
        if cells:
            lines.append("\t".join(cells))
    return "\n".join(lines).strip()


# =========================
# Classic small-doc reader
# =========================

def read_docx_text(path: str) -> str:
    """Extract text from a .docx file (paragraphs + tables)."""
    if DocxDocument is None:
        raise RuntimeError("python-docx is not installed. Please: pip install python-docx")
    doc = DocxDocument(path)
    blocks: List[str] = []
    for item in iter_block_items(doc):
        if isinstance(item, Paragraph):
            txt = para_text(item)
            if txt:
                blocks.append(txt)
        else:
            txt = table_text(item)
            if txt:
                blocks.append(txt)

    text = "\n".join(blocks)
    # Mild cleanup: normalize bullets (do NOT collapse tabs/newlines aggressively)
    text = re.sub(r"\u2022|\u25CF|\u25A0|\u2219", "-", text)  # bullets → dashes
    text = re.sub(r"(\s*\-\s*){3,}", " - ", text)  # too many dashes → one dash
    return text.strip()


# =========================
# Sentence split & TextRank
# =========================

def split_sentences(text: str) -> List[str]:
    # Keep tabs to preserve table lines as single units (later keep-rule will catch them)
    t = re.sub(r"[\r]", " ", text)
    # treat bullets like sentence boundaries
    t = re.sub(r"\s*-\s+", ". ", t)
    parts = re.split(r"(?<=[\.\!\?\u3002\uFF61])\s+", t)
    if len(parts) <= 1:
        parts = re.split(r"\s{2,}|;\s+", t)
    sents, seen = [], set()
    for s in parts:
        s = s.strip()
        if len(s) < 5:
            continue
        if s in seen:
            continue
        seen.add(s)
        sents.append(s)
    return sents

# def read_txt_text(path: str) -> str:
#     with open(path, "r", encoding="utf-8", errors="ignore") as f:
#         return f.read().strip()


# 전역(파일 상단 아무 곳) 추가: 강제 인코딩 옵션 저장용
TXT_FORCE_ENCODING = ""

def read_txt_text(path: str) -> str:
    # 1) 바이너리로 읽기
    with open(path, "rb") as fb:
        raw = fb.read()

    # 2) 강제 인코딩이 지정된 경우 최우선
    if TXT_FORCE_ENCODING:
        enc = TXT_FORCE_ENCODING.lower()
        try:
            text = raw.decode(enc)
            # 널/개행 정리
            text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
            print(f"[TXT] {path} -> forced encoding='{enc}', len={len(text)}")
            return text.strip()
        except Exception as e:
            print(f"[WARN] Forced encoding failed for {path} with '{enc}': {e}")

    # 3) BOM/UTF-16 힌트: 널바이트가 많으면 UTF-16 계열 먼저 시도
    has_many_nuls = raw.count(b"\x00") > max(1, len(raw) // 100)

    # 4) charset-normalizer로 감지
    detected = None
    try:
        res = from_bytes(raw)
        if res and res.best():
            detected = (res.best().encoding or "").lower()
    except Exception:
        pass

    # 5) 후보 인코딩 우선순위 구성
    candidates = []
    if has_many_nuls:
        candidates += ["utf-16", "utf-16-le", "utf-16-be"]
    if detected:
        candidates.insert(0, detected)
    candidates += ["utf-8-sig", "utf-8", "cp949", "euc-kr"]

    tried = set()
    chosen = None
    text = None
    for enc in candidates:
        if enc in tried: 
            continue
        tried.add(enc)
        try:
            t = raw.decode(enc)
            chosen = enc
            text = t
            break
        except Exception:
            continue

    # 6) 그래도 실패하면 손실 허용 UTF-8
    if text is None:
        chosen = "utf-8(errors=ignore)"
        text = raw.decode("utf-8", errors="ignore")

    # 7) 널/개행/공백 정리
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 8) 디버그 로그 (필요 없으면 주석)
    print(f"[TXT] {path} -> encoding='{chosen}', detected='{detected}', nuls={has_many_nuls}, len={len(text)}")
    return text.strip()

# ---- Format sniffers & fallbacks ----

def sniff_format(path: str) -> str:
    with open(path, 'rb') as f:
        head = f.read(8)
    if head.startswith(b'{\\rtf'):
        return 'rtf'
    if head.startswith(b'%PDF'):
        return 'pdf'
    if head[:8] == b'\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1':
        return 'ole'  # legacy .doc (OLE compound)
    if head[:2] == b'PK':
        return 'zip'  # docx/odt zip container
    return 'text'

# ---- Word/DRM sniffers ----
def is_drm_container(path: str) -> bool:
    try:
        with open(path, 'rb') as f:
            head = f.read(2)
        return head == b'SC'  # 보안 컨테이너 흔한 시그니처
    except Exception:
        return False

def sniff_word_like(path: str) -> str:
    with open(path, 'rb') as f:
        head = f.read(8)
    if head[:2] == b'SC':
        return 'drm'
    if head[:4] in (b'PK\x03\x04', b'PK\x05\x06', b'PK\x07\x08'):
        return 'docx_zip'   # 정상 docx(zip)
    if head[:8] == b'\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1':
        return 'ole_doc'    # 구형 .doc
    if head.startswith(b'{\\rtf'):
        return 'rtf'
    if head.startswith(b'%PDF'):
        return 'pdf'
    return 'unknown'


def extract_from_rtf(path: str) -> str:
    try:
        from striprtf.striprtf import rtf_to_text  # pip install striprtf
    except Exception:
        raise RuntimeError("RTF 폴백을 사용하려면 'pip install striprtf'가 필요합니다.")
    raw = open(path, 'rb').read()
    # RTF는 컨트롤워드 기반 → latin-1로 열고 파서에 맡기는 게 안전
    return rtf_to_text(raw.decode('latin-1', errors='ignore')).strip()

def extract_from_pdf(path: str) -> str:
    try:
        from pdfminer.high_level import extract_text  # pip install pdfminer.six
    except Exception:
        raise RuntimeError("PDF 폴백을 사용하려면 'pip install pdfminer.six'가 필요합니다.")
    return extract_text(path).strip()

def read_md_text(path: str) -> str:
    # 마크다운은 사실상 텍스트이므로, 강력한 인코딩 처리 로직을 그대로 사용
    return read_txt_text(path)

    
# ==== NEW: 입력 파일 수집(.docx, .txt) ====
# ==== 입력 파일 수집(.docx/.rtf/.pdf도 포함) ====
def iter_input_files(input_dir: str, exts=(".md",)) -> List[str]:
    paths = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if Path(f).suffix.lower() in exts:
                paths.append(os.path.join(root, f))
    paths.sort()
    return paths



def _build_tfidf(sentences: List[str]):
    n = len(sentences)
    if n < 5:
        return None, None
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except Exception:
        return None, None
    max_df = 0.9 if n >= 20 else 1.0
    min_df = 1
    vec = TfidfVectorizer(max_df=max_df, min_df=min_df, ngram_range=(1, 2))
    try:
        X = vec.fit_transform(sentences)
    except ValueError:
        return None, None
    return vec, X


def _cosine_sim_matrix(X):
    try:
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception:
        return None
    return cosine_similarity(X, dense_output=False)


def _textrank_scores(sim_mat, d=0.85, max_iter=50, tol=1e-6):
    import numpy as np
    n = sim_mat.shape[0]
    if n == 0:
        return []
    row_sums = np.asarray(sim_mat.sum(axis=1)).ravel()
    P = sim_mat.copy()
    for i in range(n):
        if row_sums[i] > 0:
            P[i] = P[i] / row_sums[i]
    r = np.ones(n) / n
    for _ in range(max_iter):
        r_new = (1-d)/n + d * P.T.dot(r)
        if np.linalg.norm(r_new - r, 1) < tol:
            r = r_new
            break
        r = r_new
    return r.tolist()


def _mmr_select(sentences: List[str], scores: List[float], k: int, sim_mat=None, lambda_=0.7):
    n = len(sentences)
    if k >= n:
        return list(range(n))
    selected, candidate = [], set(range(n))
    while len(selected) < k and candidate:
        best_i, best_val = None, -1e9
        for i in list(candidate):
            rel = scores[i]
            div = 0.0
            if selected and sim_mat is not None:
                div = max(sim_mat[i, j] if i != j else 0.0 for j in selected)
            val = lambda_ * rel - (1 - lambda_) * div
            if val > best_val:
                best_val, best_i = val, i
        selected.append(best_i)
        candidate.remove(best_i)
    return selected


def summarize_text(text: str, target_ratio: float, max_sentences: int) -> str:
    sents = split_sentences(text)
    if not sents:
        return text
    k = max(1, min(max_sentences, int(len(sents) * target_ratio)))

    vec, X = _build_tfidf(sents)
    if vec is None or X is None:
        return " ".join(sents[:k]).strip()

    sim = _cosine_sim_matrix(X)
    if sim is None:
        return " ".join(sents[:k]).strip()

    scores = _textrank_scores(sim)
    idx = _mmr_select(sents, scores, k=k, sim_mat=sim, lambda_=0.7)
    idx = sorted(idx)
    return " ".join(sents[i] for i in idx).strip()


# =========================
# Spec-preserving hybrid
# =========================

SPEC_RE = re.compile(
    r"(\b\d+(?:[.,]\d+)?\s?(mm|cm|m|km|kg|g|kW|kWh|W|Nm|N·m|N-m|rpm|V|A|°C|℃|L|ml)\b"
    r"|\b\d{3}/\d{2}R\d{2}\b|ISO\s?\d+|DIN\s?\w+|SAE\s?\w+)",
    re.I,
)
KEYWORDS = (
    "전장","전폭","전고","축거","윤거","휠베이스","출력","토크","연비","배기량","기어비",
    "허용오차","최소","최대","범위","규격","표준","등급","클래스","유효","불허","하중",
)

def is_keep_line(s: str) -> bool:
    return ("\t" in s) or bool(SPEC_RE.search(s)) or any(k in s for k in KEYWORDS)


def summarize_text_preserve_specs(text: str, target_ratio: float, max_sentences: int) -> str:
    """
    KEEP = spec/table lines.
    SUMMARY = TextRank+MMR over the remaining descriptive sentences.
    Then merge and restore original order.
    """
    sents = split_sentences(text)
    if not sents:
        return text

    order = {s: i for i, s in enumerate(sents)}
    keep = [s for s in sents if is_keep_line(s)]
    rest = [s for s in sents if s not in keep]

    k_total = max(1, min(max_sentences, int(len(sents) * target_ratio)))
    k_rest = max(0, k_total - len(keep))
    summary = ""
    if rest and k_rest > 0:
        summary = summarize_text(" ".join(rest), target_ratio=float(k_rest / max(len(rest), 1)), max_sentences=k_rest)
    summary_sents = split_sentences(summary)[:k_rest] if summary else []

    merged = list({*keep, *summary_sents})  # dedup
    merged_sorted = sorted(merged, key=lambda x: order.get(x, 10**9))
    out = " ".join(merged_sorted).strip()

    # Safety guard: if result too short, fallback to original
    if len(out) < max(200, int(len(text) * 0.05)):
        return text
    return out


# =========================
# Large DOCX chunking
# =========================

def iter_docx_chunks(path: str, chunk_chars: int = 9000, section_boundary: bool = True) -> Iterable[str]:
    """
    Stream a big .docx into chunks (~chunk_chars), respecting tables and (optionally) headings as hard boundaries.
    """
    if DocxDocument is None:
        raise RuntimeError("python-docx is not installed. Please: pip install python-docx")
    doc = DocxDocument(path)

    buf, buf_len = [], 0

    def flush():
        nonlocal buf, buf_len
        if buf:
            text = "\n".join(buf).strip()
            if text:
                yield text
        buf, buf_len = [], 0

    for item in iter_block_items(doc):
        if isinstance(item, Paragraph):
            txt = para_text(item)
            if not txt:
                continue
            is_heading = False
            try:
                style_name = (item.style.name or "").lower()
                is_heading = style_name.startswith("heading")
            except Exception:
                is_heading = False

            # hard boundary on heading
            if section_boundary and is_heading and buf:
                yield from flush()

            # chunk size boundary
            if buf_len + len(txt) > chunk_chars:
                yield from flush()

            buf.append(txt)
            buf_len += len(txt)

        else:
            # table as atomic block
            txt = table_text(item)
            if not txt:
                continue
            # force flush before and after a big table
            if buf:
                yield from flush()
            # if a single table exceeds chunk_chars, split by lines conservatively
            if len(txt) > chunk_chars:
                lines = txt.split("\n")
                tb = []
                tb_len = 0
                for line in lines:
                    if tb_len + len(line) + 1 > chunk_chars:
                        if tb:
                            yield "\n".join(tb)
                        tb = [line]
                        tb_len = len(line)
                    else:
                        tb.append(line)
                        tb_len += len(line) + 1
                if tb:
                    yield "\n".join(tb)
            else:
                yield txt

    # flush remaining
    yield from flush()


# =========================
# JSONL writers
# =========================

def write_jsonl(records: List[Dict], out_path: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(record: Dict, out_path: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# =========================
# Summarize commands
# =========================

def cmd_summarize(args):
    files = iter_docx_files(args.input_dir)
    if not files:
        raise FileNotFoundError(f"No .docx files found under: {args.input_dir}")

    # If large_mode: stream per-chunk to disk as we go (no big in-memory list)
    if args.large_mode:
        # truncate/initialize output
        open(args.output_jsonl, "w", encoding="utf-8").close()

        for p in tqdm(files, desc="Summarizing (large, streaming)"):
            try:
                for chunk in iter_docx_chunks(
                    p,
                    chunk_chars=args.chunk_chars,
                    section_boundary=not args.no_section_boundary
                ):
                    text = chunk
                    if len(text) < args.min_chars:
                        continue
                    if args.preserve_specs:
                        summ = summarize_text_preserve_specs(
                            text,
                            target_ratio=args.target_ratio,
                            max_sentences=args.max_sentences,
                        )
                    else:
                        summ = summarize_text(
                            text,
                            target_ratio=args.target_ratio,
                            max_sentences=args.max_sentences,
                        )

                    if not summ or len(summ) < max(200, int(len(text) * 0.05)):
                        summ = text

                    append_jsonl({
                        "text": summ,
                        "source_path": os.path.relpath(p, args.input_dir),
                        "summary": True,
                    }, args.output_jsonl)

            except Exception as e:
                print(f"[WARN] Failed to process {p}: {e}")

        print(f"Wrote streaming summaries to {args.output_jsonl}")
        return

    # Small/normal mode (original behavior, but with optional spec-preserve)
    recs = []
    for p in tqdm(files, desc="Summarizing .docx"):
        try:
            text = read_docx_text(p)
        except Exception as e:
            print(f"[WARN] Failed to parse {p}: {e}")
            continue
        if len(text) < args.min_chars:
            continue

        if args.preserve_specs:
            summ = summarize_text_preserve_specs(
                text, target_ratio=args.target_ratio, max_sentences=args.max_sentences
            )
        else:
            summ = summarize_text(
                text, target_ratio=args.target_ratio, max_sentences=args.max_sentences
            )

        if not summ or len(summ) < max(200, int(len(text) * 0.05)):
            summ = text

        recs.append({
            "text": summ,
            "source_path": os.path.relpath(p, args.input_dir),
            "summary": True,
        })
    write_jsonl(recs, args.output_jsonl)
    print(f"Wrote {len(recs)} summarized records to {args.output_jsonl}")


# =========================
# Preprocess (unchanged)
# =========================

def iter_docx_files(input_dir: str) -> List[str]:
    paths = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(".docx"):
                paths.append(os.path.join(root, f))
    paths.sort()
    return paths


# def cmd_preprocess(args):
#     random.seed(args.seed)
#     files = iter_docx_files(args.input_dir)
#     if not files:
#         raise FileNotFoundError(f"No .docx files found under: {args.input_dir}")

#     records: List[Dict[str, str]] = []
#     for p in tqdm(files, desc="Extracting .docx"):
#         try:
#             text = read_docx_text(p)
#         except Exception as e:
#             print(f"[WARN] Failed to parse {p}: {e}")
#             continue
#         if len(text) < args.min_chars:
#             continue
#         records.append({"text": text, "source_path": os.path.relpath(p, args.input_dir)})

#     if not records:
#         raise RuntimeError("No documents survived preprocessing; try lowering --min_chars.")

#     write_jsonl(records, args.output_jsonl)
#     print(f"Wrote {len(records)} records to {args.output_jsonl}")

def cmd_preprocess(args):
    random.seed(args.seed)
    files = iter_input_files(args.input_dir, exts=(".md",))
    if not files:
        raise FileNotFoundError(f"No .md files found under: {args.input_dir}")

    records: List[Dict[str, str]] = []

    for p in tqdm(files, desc="Extracting .md"):
        ext = Path(p).suffix.lower()
        text = ""
        try:
            if ext == ".md":
                text = read_md_text(p)
            else:
                continue
        except Exception as e:
            print(f"[WARN] Failed to parse {p}: {e}")
            continue

        if len(text) < args.min_chars:
            continue

        records.append({
            "text": text,
            "source_path": os.path.relpath(p, args.input_dir)
        })

    if not records:
        raise RuntimeError("No documents survived preprocessing; try lowering --min_chars.")

    write_jsonl(records, args.output_jsonl)
    print(f"Wrote {len(records)} records to {args.output_jsonl}")







# =========================
# Dataset tokenization/packing (unchanged)
# =========================

@dataclass
class PackingConfig:
    block_size: Optional[int] = None
    add_eos_between_docs: bool = True


def load_jsonl_as_dataset(train_jsonl: str, eval_ratio: float = 0.05, seed: int = 42):
    ds = load_dataset("json", data_files=train_jsonl, split="train")
    ds = ds.shuffle(seed=seed)
    split = ds.train_test_split(test_size=max(1, int(len(ds) * eval_ratio)), seed=seed)
    return datasets.DatasetDict({
        "train": split["train"],
        "validation": split["test"],
    })


def build_tokenize_and_group_fn(tokenizer: AutoTokenizer, packing: PackingConfig):
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        eos_id = tokenizer.eos_token_id

    def tokenize(example):
        text = example["text"]
        if packing.add_eos_between_docs and tokenizer.eos_token:
            text = text + tokenizer.eos_token
        return tokenizer(text, add_special_tokens=False)

    if packing.block_size is None:
        try:
            max_len = tokenizer.model_max_length
            if isinstance(max_len, int) and max_len < 100000:
                block_size = min(4096, max_len)
            else:
                block_size = 4096
        except Exception:
            block_size = 4096
    else:
        block_size = packing.block_size

    def group_texts(examples):
        concatenated = []
        for ids in examples["input_ids"]:
            concatenated.extend(ids)
        total_length = (len(concatenated) // block_size) * block_size
        concatenated = concatenated[:total_length]
        result = {
            "input_ids": [concatenated[i: i + block_size] for i in range(0, total_length, block_size)]
        }
        result["attention_mask"] = [[1] * block_size for _ in range(len(result["input_ids"]))]
        result["labels"] = [x.copy() for x in result["input_ids"]]
        return result

    return tokenize, group_texts, block_size


# =========================
# Training (unchanged)
# =========================

def maybe_prepare_lora(model, use_lora: bool, use_qlora: bool, lora_r: int, lora_alpha: int, lora_dropout: float):
    if not use_lora:
        return model
    if LoraConfig is None:
        raise RuntimeError("peft is not installed. Please: pip install peft")

    if use_qlora and prepare_model_for_kbit_training is not None:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_cfg)
    try:
        model.print_trainable_parameters()
    except Exception:
        pass
    return model


def cmd_train(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(
        device_map="auto",
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        attn_implementation=getattr(args, "attn_impl", "eager"),
    )

    qconf = None
    if args.use_qlora:
        qconf = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )
    elif getattr(args, "use_int8", False):
        qconf = BitsAndBytesConfig(load_in_8bit=True)

    if qconf is not None:
        load_kwargs["quantization_config"] = qconf

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **load_kwargs)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    ds = load_jsonl_as_dataset(args.train_jsonl, eval_ratio=args.eval_ratio, seed=args.seed)

    packing_cfg = PackingConfig(block_size=args.block_size if args.block_size > 0 else None,
                                add_eos_between_docs=not args.no_add_eos)
    tokenize_fn, group_fn, used_block = build_tokenize_and_group_fn(tokenizer, packing_cfg)

    tokenized = ds.map(tokenize_fn, batched=False, remove_columns=ds["train"].column_names)

    if args.packing:
        tokenized = tokenized.map(
            group_fn,
            batched=True,
            batch_size=args.group_batch_size,
        )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    os.makedirs(args.output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=not args.bf16,
        dataloader_num_workers=args.num_workers,
        gradient_checkpointing=args.gradient_checkpointing,
        optim="adamw_torch",
        report_to=["none"],
        prediction_loss_only=True,
    )

    model = maybe_prepare_lora(
        model,
        use_lora=args.use_lora,
        use_qlora=args.use_qlora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
    )

    trainer.train()
    eval_out = trainer.evaluate()
    try:
        ppl = float(torch.exp(torch.tensor(eval_out["eval_loss"])))
        print(f"Perplexity: {ppl:.2f}")
    except Exception:
        pass

    if args.use_lora:
        model.save_pretrained(os.path.join(args.output_dir, "adapter"))
        tokenizer.save_pretrained(args.output_dir)
        print(f"Saved LoRA adapter to {os.path.join(args.output_dir, 'adapter')}\n"
              f"To merge adapters into the base model later, use PEFT utilities or inference with model+adapter.")
    else:
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Saved full model to {args.output_dir}")


# =========================
# CLI
# =========================

def build_arg_parser():
    p = argparse.ArgumentParser(description="DOCX → JSONL → DAPT for Llama-3 Instruct")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("preprocess", help="Extract text from .docx to JSONL")
    sp.add_argument("--input_dir", type=str, required=True, help="Folder containing .docx files")
    sp.add_argument("--output_jsonl", type=str, required=True, help="Path to write JSONL")
    sp.add_argument("--min_chars", type=int, default=800, help="Drop docs shorter than this")
    sp.add_argument("--seed", type=int, default=42)
    sp.set_defaults(func=cmd_preprocess)
    
    sp.add_argument("--txt_encoding", type=str, default="",
                help="Force encoding for .txt (e.g., 'utf-16', 'cp949', 'utf-8'). If empty, auto-detect.")


    ss = sub.add_parser("summarize", help="Summarize .docx to compressed JSONL (hybrid, large-safe)")
    ss.add_argument("--input_dir", type=str, required=True)
    ss.add_argument("--output_jsonl", type=str, required=True)
    ss.add_argument("--target_ratio", type=float, default=0.25, help="Fraction of sentences to keep (per chunk)")
    ss.add_argument("--max_sentences", type=int, default=40)
    ss.add_argument("--min_chars", type=int, default=500)
    # large-mode options
    ss.add_argument("--large_mode", action="store_true", help="Stream huge .docx as chunks (memory safe)")
    ss.add_argument("--chunk_chars", type=int, default=9000, help="Max characters per chunk in large_mode")
    ss.add_argument("--no_section_boundary", action="store_true", help="Do not force chunk boundary at headings")
    # spec-preserve
    ss.add_argument("--preserve_specs", action="store_true", help="Always keep spec/table lines, summarize only prose")
    ss.set_defaults(func=cmd_summarize)

    st = sub.add_parser("train", help="Run continued pre-training (causal LM)")
    st.add_argument("--use_int8", action="store_true")
    st.add_argument("--attn_impl", type=str, default="eager", choices=["eager","sdpa","flash_attention_2"],
                    help="Attention backend. 'eager'는 가장 호환성이 높음")
    st.add_argument("--model_id", type=str, required=True,
                    help="HF model id or local path (e.g., meta-llama/Meta-Llama-3-8B-Instruct)")
    st.add_argument("--train_jsonl", type=str, required=True, help="JSONL from preprocess/summarize (has 'text')")
    st.add_argument("--output_dir", type=str, required=True)

    # Data & packing
    st.add_argument("--eval_ratio", type=float, default=0.05)
    st.add_argument("--block_size", type=int, default=-1, help="Token block size; -1 → auto (≈4096)")
    st.add_argument("--group_batch_size", type=int, default=1000,
                    help="How many examples to group per map() call when packing")
    st.add_argument("--no_add_eos", action="store_true", help="Do not append EOS between docs")
    st.add_argument("--packing", action="store_true", help="Pack sequences up to block_size")

    # Training hyperparams
    st.add_argument("--num_train_epochs", type=int, default=10)
    st.add_argument("--per_device_train_batch_size", type=int, default=2)
    st.add_argument("--per_device_eval_batch_size", type=int, default=2)
    st.add_argument("--gradient_accumulation_steps", type=int, default=8)
    st.add_argument("--learning_rate", type=float, default=2e-5)
    st.add_argument("--weight_decay", type=float, default=0.05)
    st.add_argument("--warmup_ratio", type=float, default=0.1)
    st.add_argument("--lr_scheduler_type", type=str, default="cosine")
    st.add_argument("--logging_steps", type=int, default=2)
    st.add_argument("--save_steps", type=int, default=200)
    st.add_argument("--eval_steps", type=int, default=2)
    st.add_argument("--save_total_limit", type=int, default=2)
    st.add_argument("--num_workers", type=int, default=2)

    # Mixed precision & memory
    st.add_argument("--bf16", action="store_true")
    st.add_argument("--gradient_checkpointing", action="store_true")

    # (Q)LoRA options
    st.add_argument("--use_lora", action="store_true")
    st.add_argument("--use_qlora", action="store_true")
    st.add_argument("--lora_r", type=int, default=16)
    st.add_argument("--lora_alpha", type=int, default=32)
    st.add_argument("--lora_dropout", type=float, default=0.05)

    st.add_argument("--seed", type=int, default=42)

    st.set_defaults(func=cmd_train)
    return p


# def main():
#     parser = build_arg_parser()
#     args = parser.parse_args()
#     args.func(args)
    
def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # 전역 강제 인코딩 설정
    global TXT_FORCE_ENCODING
    TXT_FORCE_ENCODING = (getattr(args, "txt_encoding", "") or "").strip()

    args.func(args)


if __name__ == "__main__":
    main()
