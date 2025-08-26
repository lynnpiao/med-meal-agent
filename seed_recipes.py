# seed_recipes.py
# -*- coding: utf-8 -*-
"""
Build a hybrid search stack for recipes:
- Embed with sentence_transformers (BAAI/bge-m3) into FAISS
- Index search_text into Elasticsearch (BM25)
- Optional Chinese query expansion + hybrid retrieve + multilingual rerank

Layout expected:
data/
  recipes/
    _index.json               # [{"id":..., "title":..., "cuisine":"Chinese","file":"184....json"}, ...]
    Chinese/
      1841189152265065138.json
    Mediterranean/
      ...
  embeddings/
    recipes.faiss
    recipes_meta.jsonl
    recipes_config.json
    # (ES 索引是独立服务里)

Run (build only):
  python seed_recipes.py --root data/recipes --out data/embeddings --limit 0

Run (build + test a Chinese query):
  python seed_recipes.py --root data/recipes --out data/embeddings --query "简易蔬菜炒饭" --topk 10

Env:
  ES_URL=http://localhost:9200
  ES_USER=elastic        # 可选
  ES_PASS=xxxx           # 可选
  COHERE_API_KEY=xxxxx   # 可选（启用 rerank）
"""

from __future__ import annotations
import os
import re
import json
import time
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Iterable, Optional

from dotenv import load_dotenv, find_dotenv

import numpy as np

def load_env():
    """不覆盖已存在的进程环境；依次加载：脚本同目录 .env、CWD/.env、向上搜索 .env。"""
    loaded_from = []
    for p in [
        Path(__file__).with_name(".env"),        # seed_recipes.py 同目录
        Path.cwd() / ".env",                     # 当前工作目录
        Path(find_dotenv() or "")               # 向上搜索到的最近 .env
    ]:
        if p and p.exists():
            ok = load_dotenv(p, override=False)
            if ok:
                loaded_from.append(str(p))

    if os.getenv("DEBUG_ENV") == "1":
        print("[ENV] loaded from:", loaded_from or "none")
        for k in ("ES_URL"): # "ES_USER","ES_PASS","ES_CA_CERT","ES_INSECURE"
            print(f"[ENV] {k} =", os.getenv(k))

# ---- FAISS ----
try:
    import faiss
except Exception as e:
    raise SystemExit("Missing dependency: faiss-cpu  →  pip install faiss-cpu\n" + str(e))

# ---- sentence_transformers (BGE-M3) ----
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise SystemExit("Missing dependency: sentence-transformers  →  pip install sentence-transformers\n" + str(e))

# ---- Elasticsearch (optional but recommended) ----
try:
    from elasticsearch import Elasticsearch, helpers
    ES_AVAILABLE = True
except Exception:
    ES_AVAILABLE = False

# ---- Cohere (optional reranker) ----
try:
    import cohere
    COHERE_AVAILABLE = True
except Exception:
    COHERE_AVAILABLE = False


# =========================
# IO helpers
# =========================

def _load_index(index_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(index_path.read_text("utf-8"))
    items = data["recipes"] if isinstance(data, dict) and "recipes" in data else data
    out = []
    for it in items:
        if all(k in it for k in ("id", "title", "cuisine", "file")):
            out.append(it)
    return out


def _read_recipe_payload(root: Path, entry: Dict[str, Any]) -> Dict[str, Any]:
    cuisine = entry.get("cuisine") or ""
    file_ = entry.get("file") or ""
    p = (root / file_) if any(sep in file_ for sep in ("/", "\\")) else (root / cuisine / file_)
    if not p.exists():
        raise FileNotFoundError(f"Recipe file not found: {p}")
    text = p.read_text("utf-8", errors="ignore")
    if p.suffix.lower() == ".json":
        try:
            return json.loads(text)
        except Exception:
            return {"title": entry.get("title"), "cuisine": cuisine, "raw": text}
    else:
        return {"title": entry.get("title"), "cuisine": cuisine, "raw": text}


def _fmt_ingredients(ings: Iterable[Dict[str, Any]]) -> str:
    """兼容你当前 schema: {name, amount}；若将来有 {quantity, unit} 也能印出来。"""
    lines = []
    for ing in ings or []:
        name = str(ing.get("name", "")).strip()
        amount = str(ing.get("amount", "")).strip()
        qty = str(ing.get("quantity", "")).strip()
        unit = str(ing.get("unit", "")).strip()
        if not name:
            continue
        if amount:
            lines.append(f"- {name} {amount}")
        elif qty and unit:
            lines.append(f"- {name} {qty} {unit}")
        elif qty:
            lines.append(f"- {name} {qty}")
        else:
            lines.append(f"- {name}")
    return "\n".join(lines)


def _fmt_instructions(steps: Iterable[str]) -> str:
    lines = []
    for i, s in enumerate(steps or [], start=1):
        s = str(s).strip()
        if s:
            lines.append(f"{i}. {s}")
    return "\n".join(lines)


def _fmt_nutrition(nut: Dict[str, Any]) -> str:
    if not isinstance(nut, dict) or not nut:
        return ""
    parts = []
    for k, v in nut.items():
        try:
            parts.append(f"{k}: {float(v)}")
        except Exception:
            parts.append(f"{k}: {v}")
    return ", ".join(parts)


def build_doc(recipe: Dict[str, Any], fallback_title: str, cuisine: str) -> str:
    """把一条 recipe 打平成一个可检索文本（英文为主，但不妨碍中文检索）。"""
    title = (recipe.get("title") or fallback_title or "").strip()
    ings = recipe.get("ingredients") or []
    steps = recipe.get("instructions") or recipe.get("steps") or []
    nut = recipe.get("per_serving_nutrition") or {}

    if not title and "raw" in recipe:
        return f"Title: {fallback_title}\nCuisine: {cuisine}\n\n{recipe['raw']}"

    doc = [
        f"Title: {title}",
        f"Cuisine: {cuisine}",
    ]
    if ings:
        doc.append("Ingredients:")
        doc.append(_fmt_ingredients(ings))
    if steps:
        doc.append("Instructions:")
        doc.append(_fmt_instructions(steps))
    if nut:
        doc.append("Per-serving nutrition:")
        doc.append(_fmt_nutrition(nut))
    if "raw" in recipe and not (ings or steps):
        doc.append(recipe["raw"])
    return "\n".join(doc).strip()


# =========================
# Chunking
# =========================

def chunk_text(doc: str, max_chars: int = 1200, overlap: int = 120) -> List[str]:
    doc = doc.strip()
    if len(doc) <= max_chars:
        return [doc]
    chunks, start = [], 0
    while start < len(doc):
        end = min(len(doc), start + max_chars)
        chunks.append(doc[start:end])
        if end == len(doc):
            break
        start = max(0, end - overlap)
    return chunks


# =========================
# Embeddings (BGE-M3)
# =========================

def load_embed_model(name: str = "BAAI/bge-m3") -> SentenceTransformer:
    model = SentenceTransformer(name)
    return model

# BGE 提示词建议（可关闭）
DOC_PREFIX = "Represent the document for retrieval: "
QRY_PREFIX = "Represent the question for retrieving relevant documents: "

def embed_docs(model: SentenceTransformer, texts: List[str], batch_size: int = 64, add_prefix: bool = True) -> np.ndarray:
    if add_prefix:
        texts = [DOC_PREFIX + t for t in texts]
    vecs = model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(vecs, dtype="float32")

def embed_query(model: SentenceTransformer, query: str, add_prefix: bool = True) -> np.ndarray:
    q = QRY_PREFIX + query if add_prefix else query
    v = model.encode([q], show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(v, dtype="float32")


# =========================
# Elasticsearch (optional)
# =========================

def get_es_client() -> Optional[Elasticsearch]:
    """Connect to Elasticsearch (no security). Defaults to http://localhost:9200.
    Adds 'compatible-with=8' headers so ES 8.x works even if python client is 9.x."""
    if not ES_AVAILABLE:
        print("[ES] elasticsearch client not installed (pip install 'elasticsearch>=8,<9').")
        return None

    url = os.getenv("ES_URL") or "http://localhost:9200"

    # 兼容 ES 8.x + es-py 9.x 的媒体类型头；8.x 客户端也能正常工作（无副作用）
    compat_headers = {
        "accept": "application/vnd.elasticsearch+json; compatible-with=8",
        "content-type": "application/vnd.elasticsearch+json; compatible-with=8",
    }

    try:
        es = Elasticsearch(
            url,
            request_timeout=30,
            headers=compat_headers,
        )
        es.info()  # ping
        return es
    except Exception as e:
        print(f"[ES] connect failed: {e}")
        return None
    
def ensure_es_index(es: Elasticsearch, index_name: str):
    if es.indices.exists(index=index_name):
        return
    body = {
        "settings": {
            "index": {"number_of_shards": 1, "number_of_replicas": 0},
            "analysis": {
                "filter": {
                    "recipe_syns": {
                        "type": "synonym",
                        "synonyms": [
                            "炒饭, fried rice",
                            "蔬菜, vegetable, veggie, veggies",
                            "低脂, low fat, low-fat, skinny, light",
                            "低卡, low calorie, low-calorie",
                            "高蛋白, high protein, high-protein, protein rich",
                            "无麸质, gluten free, gluten-free",
                            "香油, 芝麻油, sesame oil",
                            "锅贴, potsticker, pot stickers, dumpling, gyoza",
                            "馄饨, wonton, wontons",
                            "中餐, chinese",
                            "地中海, mediterranean"
                        ]
                    }
                },
                "analyzer": {
                    # 若已安装 IK，可把 tokenizer 换成 ik_max_word
                    "recipe_ana": {
                        "tokenizer": "standard",
                        "filter": ["lowercase", "recipe_syns"]
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "title": {"type": "text", "analyzer": "recipe_ana"},
                "cuisine": {"type": "keyword"},
                "file": {"type": "keyword"},
                "chunk": {"type": "integer"},
                "text": {"type": "text", "analyzer": "recipe_ana"}
            }
        }
    }
    es.indices.create(index=index_name, body=body)


# def ensure_es_index(es: Elasticsearch, index_name: str):
#     if es.indices.exists(index=index_name):
#         return
#     body = {
#         "settings": {
#             "index": {
#                 "number_of_shards": 1,
#                 "number_of_replicas": 0
#             },
#             "analysis": {
#                 # 先用标准分词；如需中文细分可换 ik_smart/ik_max_word（需安装插件）
#                 "analyzer": {
#                     "default": {"type": "standard"}
#                 }
#             }
#         },
#         "mappings": {
#             "properties": {
#                "id": {"type": "keyword"},      # 精确匹配，不分词
#                "title": {"type": "text"},      # 分词→BM25
#                "cuisine": {"type": "keyword"}, # 精确过滤/聚合
#                "file": {"type": "keyword"},    # 精确存储文件名
#                "chunk": {"type": "integer"},   # 第几个切片
#                "text": {"type": "text"}        # 主要检索正文（BM25）
#             }
#         }
#     }
#     es.indices.create(index=index_name, body=body)

def es_bulk_index(es: Elasticsearch, index_name: str, docs: List[Dict[str, Any]]):
    actions = ({
        "_op_type": "index",
        "_index": index_name,
        "_id": f"{d['id']}::{d['chunk']}",
        "_source": d
    } for d in docs)
    helpers.bulk(es, actions, request_timeout=120)


# =========================
# Build indices
# =========================

def build_indices(
    root: Path,
    outdir: Path,
    limit: int = 0,
    max_chars: int = 1200,
    overlap: int = 120,
    model_name: str = "BAAI/bge-m3",
    es_index: str = "recipes"
) -> Tuple[faiss.Index, List[Dict[str, Any]], Optional[Elasticsearch]]:
    idx_path = root / "_index.json"
    entries = _load_index(idx_path)
    if limit and limit > 0:
        entries = entries[:limit]

    # Build docs & metas
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []
    es_docs: List[Dict[str, Any]] = []

    for e in entries:
        rid = str(e["id"])
        title = e["title"]
        cuisine = e["cuisine"]
        try:
            payload = _read_recipe_payload(root, e)
        except Exception as ex:
            print(f"[WARN] skip {rid}: {ex}")
            continue
        full_text = build_doc(payload, fallback_title=title, cuisine=cuisine)
        chunks = chunk_text(full_text, max_chars=max_chars, overlap=overlap)
        for ci, ch in enumerate(chunks):
            docs.append(ch)
            meta = {"id": rid, "title": title, "cuisine": cuisine, "file": e["file"], "chunk": ci}
            metas.append(meta)
            es_docs.append({**meta, "text": ch})

    if not docs:
        raise RuntimeError("No recipe documents built; nothing to index.")

    # Embedding → FAISS
    model = load_embed_model(model_name)
    vecs = embed_docs(model, docs)
    dim = vecs.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)   # cosine with normalized vectors
    faiss_index.add(vecs)

    # Save FAISS artifacts
    outdir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(faiss_index, str(outdir / "recipes.faiss"))
    with (outdir / "recipes_meta.jsonl").open("w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    (outdir / "recipes_config.json").write_text(
        json.dumps({"model": model_name, "dim": dim, "es_index": es_index}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"[FAISS] {len(docs)} chunks / dim={dim} → {outdir/'recipes.faiss'}")

    # Elasticsearch
    es = get_es_client()
    if es is None:
        print("[ES] Elasticsearch not available → skip BM25 index.")
    else:
        ensure_es_index(es, es_index)
        es_bulk_index(es, es_index, es_docs)
        es.indices.refresh(index=es_index)
        print(f"[ES] Indexed {len(es_docs)} docs into '{es_index}'")

    return faiss_index, metas, es


# =========================
# Chinese query expansion
# =========================
ZH_EN_PATH = os.getenv("ZH_EN_PATH", "data/zh_en_synonyms.yaml")

def load_zh_en_dict(path: str) -> Dict[str, List[str]]:
    if not os.path.exists(path):
        return {}
    if path.endswith(".json"):
        return json.loads(Path(path).read_text("utf-8"))
    # 默认 YAML
    return yaml.safe_load(Path(path).read_text("utf-8")) or {}

_ZH_EN = load_zh_en_dict(ZH_EN_PATH)

def expand_query(query: str) -> List[str]:
    """中文→英文扩展；保留原始中文，并加入在 query 中出现的词条对应英文同义词。"""
    toks = set([query])

    # 命中子串就扩展（无需中文分词也能工作）
    for zh, ens in _ZH_EN.items():
        if zh and zh in query:
            for en in ens or []:
                if en:
                    toks.add(en)

    # 额外抓取 query 中本就带的英文片段（大小写/空白清理）
    extra = re.findall(r"[A-Za-z][A-Za-z\s\-&]+", query)
    toks.update(s.strip().lower() for s in extra if s.strip())

    return list(toks)

# _ZH_EN = {
#     "炒饭": ["fried rice"],
#     "素": ["vegetarian", "veggie"],
#     "低脂": ["low fat", "skinny", "light"],
#     "糙米": ["brown rice"],
#     "菜花": ["cauliflower"], 
#     "花椰菜": ["cauliflower"],
#     "西兰花": ["broccoli"],
#     "豌豆": ["peas"],
#     "葱": ["scallion", "green onion"],
#     "蒜": ["garlic"],
#     "姜": ["ginger"],
#     "香油": ["sesame oil"], 
#     "芝麻油": ["sesame oil"],
#     "酱油": ["soy sauce"],
#     "锅贴": ["potsticker", "pot stickers", "dumpling", "gyoza"],
#     "馄饨": ["wonton", "wontons"],
#     "蔬菜": ["vegetable", "veggie"],
#     "瘦身": ["skinny", "low-calorie", "light"],
#     "清爽": ["light", "refreshing"],
# }

# def expand_query(query: str) -> List[str]:
#     """简单中文→英文扩展；保留原始中文，用 OR 联合。"""
#     toks = set([query])
#     for zh, ens in _ZH_EN.items():
#         if zh in query:
#             for en in ens:
#                 toks.add(en)
#     # 简单中英数字/空白标准化
#     extra = re.findall(r"[A-Za-z][A-Za-z\s-]+", query)
#     toks.update(s.strip().lower() for s in extra)
#     return list(toks)


# =========================
# Hybrid search + Rerank
# =========================

def load_for_search(outdir: Path) -> Tuple[faiss.Index, List[Dict[str, Any]], Dict[str, Any]]:
    faiss_idx = faiss.read_index(str(outdir / "recipes.faiss"))
    metas = [json.loads(line) for line in (outdir / "recipes_meta.jsonl").read_text("utf-8").splitlines()]
    cfg = json.loads((outdir / "recipes_config.json").read_text("utf-8"))
    return faiss_idx, metas, cfg

def faiss_topk(faiss_idx: faiss.Index, metas: List[Dict[str, Any]], model_name: str, query: str, k: int) -> List[Dict[str, Any]]:
    model = load_embed_model(model_name)
    qv = embed_query(model, query)
    D, I = faiss_idx.search(qv, k)
    results = []
    for score, i in zip(D[0], I[0]):
        if i == -1:
            continue
        m = metas[int(i)]
        results.append({"score_vec": float(score), **m})
    return results

def es_topk(es: Elasticsearch, index_name: str, query: str, k: int) -> List[Dict[str, Any]]:
    exp = expand_query(query)
    # 原始短语强力加权（title^3、text^1.5）
    phrase_boost = {
        "multi_match": {
            "query": query,
            "type": "phrase",
            "fields": ["title^3", "text^1.5"],
            "boost": 3.0
        }
    }
    # 扩展词 OR，仍对 title 提高权重
    keyword_boost = {
        "query_string": {
            "query": " OR ".join([f"({es_escape(t)})" for t in exp]),
            "fields": ["title^2", "text"]
        }
    }
    body = {
        "size": k,
        "query": {
            "bool": {
                "should": [phrase_boost, {"bool": {"should": [keyword_boost]}}],
                "minimum_should_match": 1
            }
        }
    }
    hits = es.search(index=index_name, body=body)["hits"]["hits"]
    out = []
    for h in hits:
        src = h["_source"]
        out.append({"score_bm25": float(h["_score"]), **{k: src[k] for k in ("id","title","cuisine","file","chunk")}})
    return out

# def es_topk(es: Elasticsearch, index_name: str, query: str, k: int) -> List[Dict[str, Any]]:
#     exp = expand_query(query)
#     # 用 query_string 做 OR 扩展；必要时可以对 title/text 分字段加权
#     q = " OR ".join([f"({es_escape(x)})" for x in exp])
#     body = {
#         "size": k,
#         "query": {
#             "query_string": {
#                 "query": q,
#                 "fields": ["title^2", "text"]
#             }
#         }
#     }
#     hits = es.search(index=index_name, body=body)["hits"]["hits"]
#     out = []
#     for h in hits:
#         src = h["_source"]
#         out.append({"score_bm25": float(h["_score"]), **{k: src[k] for k in ("id","title","cuisine","file","chunk")}})
#     return out

def es_escape(s: str) -> str:
    # 简单转义 ES query_string 特殊字符
    return re.sub(r'([+\-!(){}\[\]^"~*?:\\/]|&&|\|\|)', r'\\\1', s)

def merge_candidates(vec_res: List[Dict[str, Any]], bm_res: List[Dict[str, Any]], topk: int = 20) -> List[Dict[str, Any]]:
    # 用 (id,chunk) 去重合并；保留双方分数
    key = lambda r: f"{r['id']}::{r['chunk']}"
    merged: Dict[str, Dict[str, Any]] = {}
    for r in vec_res + bm_res:
        k = key(r)
        if k not in merged:
            merged[k] = r
        else:
            merged[k].update(r)
    # 简单加权：vec 优先，其次 bm25（可改为学习到的融合策略）
    def score(r):
        return (r.get("score_vec", 0) * 1.0) + (r.get("score_bm25", 0) * 0.2)
    res = sorted(merged.values(), key=score, reverse=True)[:topk]
    return res

def cohere_rerank_if_available(query: str, candidates: List[Dict[str, Any]], texts: List[str]) -> List[int]:
    if not (COHERE_AVAILABLE and os.getenv("COHERE_API_KEY")):
        return list(range(len(candidates)))  # 原顺序
    client = cohere.Client(os.getenv("COHERE_API_KEY"))
    # Cohere 支持传 {"text": "..."} 数组
    docs = [{"text": t} for t in texts]
    try:
        resp = client.rerank(model="rerank-multilingual-v3.0", query=query, documents=docs, top_n=len(docs))
        order = [r.index for r in resp.results]  # 索引对应传入 docs 的顺序
        return order
    except Exception:
        return list(range(len(candidates)))


def pretty_print_results(cands: List[Dict[str, Any]]):
    print("\n=== Search results ===")
    for r in cands:
        sv = r.get("score_vec", 0.0)
        sb = r.get("score_bm25", 0.0)
        print(f"[vec={sv:>6.4f} bm25={sb:>6.2f}] ({r['cuisine']}) {r['title']}  "
              f"#{r['id']}  [chunk {r['chunk']}]  file={r['file']}")


def hybrid_search(outdir: Path, query: str, topk: int = 10, vec_k: int = 50, bm25_k: int = 50):
    faiss_idx, metas, cfg = load_for_search(outdir)
    model_name = cfg["model"]
    es_index = cfg.get("es_index", "recipes")

    # 向量召回
    vec_res = faiss_topk(faiss_idx, metas, model_name, query, vec_k)

    # BM25（若 ES 不可用则为空）
    es = get_es_client()
    bm_res = es_topk(es, es_index, query, bm25_k) if es else []

    # 合并去重
    merged = merge_candidates(vec_res, bm_res, topk=topk)

    # 取出 candidate 文本（用于 rerank）；这里简单用 title+file 标注
    texts = []
    for r in merged:
        # 为了 rerank 质量更好，可加载 chunk 原文；这里用最简实现：根据 metas 反查文本需要额外存储或读取
        # 轻量做法：用 title + cuisine + file 标注（已足够把“炒饭/锅贴”区分开）；要更准可以把 chunk 文本另存 jsonl。
        texts.append(f"{r['title']} | cuisine={r['cuisine']} | file={r['file']} | chunk={r['chunk']}")

    # 多语种 rerank（可选）
    order = cohere_rerank_if_available(query, merged, texts)
    reranked = [merged[i] for i in order[:topk]]

    pretty_print_results(reranked)


# =========================
# CLI
# =========================

def main():
    load_env()
    ap = argparse.ArgumentParser(description="Seed FAISS + Elasticsearch hybrid index from recipes (BGE-M3).")
    ap.add_argument("--root", default="data/recipes", type=str, help="Recipe root directory")
    ap.add_argument("--out", default="data/embeddings", type=str, help="Output directory for FAISS")
    ap.add_argument("--limit", default=0, type=int, help="Limit number of recipes (0 = no limit)")
    ap.add_argument("--max-chars", default=1200, type=int, help="Max chars per chunk")
    ap.add_argument("--overlap", default=120, type=int, help="Overlap chars between chunks")
    ap.add_argument("--model", default="BAAI/bge-m3", type=str, help="sentence-transformers model name")
    ap.add_argument("--es-index", default="recipes", type=str, help="Elasticsearch index name")
    ap.add_argument("--query", default=None, type=str, help="Optional test query after building (supports Chinese)")
    ap.add_argument("--topk", default=10, type=int, help="Final topK to print")
    ap.add_argument("--vec-k", default=50, type=int, help="Vector candidates K")
    ap.add_argument("--bm25-k", default=50, type=int, help="BM25 candidates K")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out = Path(args.out).resolve()

    build_indices(
        root=root,
        outdir=out,
        limit=args.limit,
        max_chars=args.max_chars,
        overlap=args.overlap,
        model_name=args.model,
        es_index=args.es_index,
    )

    if args.query:
        hybrid_search(outdir=out, query=args.query, topk=args.topk, vec_k=args.vec_k, bm25_k=args.bm25_k)


if __name__ == "__main__":
    main()
