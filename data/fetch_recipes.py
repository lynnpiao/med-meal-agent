#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch recipes from Spoonacular API only, normalize to your Recipe schema,
and save as data/recipes/{cuisine}.json.

Recipe schema fields produced:
- id: str
- title: str
- ingredients: List[{"name": str, "amount": str}]
- instructions: List[str]
- per_serving_nutrition: Dict[str, float]  # keys: kcal, protein_g, carb_g, fat_g, sodium_mg
- tags: List[str]  # lowercase & unique
"""

import os
import re
import json
import argparse
import hashlib
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# 优先同目录 .env，其次向上查找
env_path = Path(__file__).with_name(".env")
if env_path.exists():
    load_dotenv(env_path)          # 明确同目录 .env
else:
    load_dotenv(find_dotenv())     # 或者向上层查找


from typing import List, Dict, Any, Optional
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

SPOONACULAR_ENDPOINT = "https://api.spoonacular.com/recipes/complexSearch"
INFO_BULK_ENDPOINT = "https://api.spoonacular.com/recipes/informationBulk"
NUTRITION_WIDGET_TMPL = "https://api.spoonacular.com/recipes/{id}/nutritionWidget.json"

# -----------------------------
# Utils
# -----------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def norm_space(x: str) -> str:
    return re.sub(r"\s+", " ", x or "").strip()

def make_id(title: str, cuisine: str) -> int:
    """
    由 (title, cuisine) 生成稳定的 64 位无符号整数ID，并以数字字符串返回。
    同一 (title, cuisine) 在任何机器/任何时间都会得到相同ID。
    """
    key = f"{title.strip().lower()}|{cuisine.strip().lower()}".encode("utf-8")
    # 8字节 = 64位；BLAKE2b 稳定、速度快
    digest8 = hashlib.blake2b(key, digest_size=8).digest()
    n = int.from_bytes(digest8, byteorder="big", signed=False)  # 0..2^64-1
    return n

def norm_tags(tags: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for t in tags or []:
        k = (t or "").strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out

def dedup_by_title(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for r in items:
        k = (r.get("title") or "").strip().lower()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out

# ---- Ingredient shaping ------------------------------------------------------
def make_ingredient(name: str, amount_text: str) -> Dict[str, Any]:
    """
    默认输出：{"name": str, "amount": str}
    若你的 InventoryItem 是 {name, quantity, unit}，可在此改造解析。
    """
    return {"name": norm_space(name), "amount": norm_space(amount_text)}

# ---- Nutrition mapping -------------------------------------------------------
def map_spoonacular_nutrition(nutrients: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Map Spoonacular 'nutrition.nutrients' to:
      kcal, protein_g, carb_g, fat_g, sodium_mg
    所有数值 round(..., 3)
    """
    out: Dict[str, float] = {}

    def pick(name_lc: str) -> Optional[Dict[str, Any]]:
        for n in nutrients or []:
            if (n.get("name") or "").strip().lower() == name_lc:
                return n
        return None

    # kcal
    cal = pick("calories")
    if cal and cal.get("amount") is not None:
        amt = float(cal["amount"])
        unit = (cal.get("unit") or "").lower()
        if unit == "cal":  # 从小卡转千卡
            amt = amt / 1000.0
        out["kcal"] = round(amt, 3)

    # protein_g
    pro = pick("protein")
    if pro and pro.get("amount") is not None:
        out["protein_g"] = round(float(pro["amount"]), 3)

    # carb_g
    carb = pick("carbohydrates")
    if carb and carb.get("amount") is not None:
        out["carb_g"] = round(float(carb["amount"]), 3)

    # fat_g
    fat = pick("fat")
    if fat and fat.get("amount") is not None:
        out["fat_g"] = round(float(fat["amount"]), 3)

    # sodium_mg
    sod = pick("sodium")
    if sod and sod.get("amount") is not None:
        amt = float(sod["amount"])
        unit = (sod.get("unit") or "").lower()
        if unit in ("g", "gram", "grams"):
            amt *= 1000.0
        out["sodium_mg"] = round(amt, 3)

    return out

def _chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i+size]

def _num_from_str(s: str) -> Optional[float]:
    if not s:
        return None
    m = re.search(r"([\d.]+)", s)
    return float(m.group(1)) if m else None

def _parse_widget_nutrition(widget_json: Dict[str, Any]) -> Dict[str, float]:
    """
    解析 /recipes/{id}/nutritionWidget.json 返回的简要营养信息：
      {"calories":"405kcal","carbs":"64g","fat":"15g","protein":"19g", "bad":[...], "good":[...]}
    从中提取 kcal / carb_g / fat_g / protein_g；
    sodium 尝试从 bad/good 列表中抓 "Sodium" 并统一为 mg。
    """
    out: Dict[str, float] = {}
    kcal = _num_from_str(widget_json.get("calories"))
    carb = _num_from_str(widget_json.get("carbs"))
    fat  = _num_from_str(widget_json.get("fat"))
    prot = _num_from_str(widget_json.get("protein"))
    if kcal is not None: out["kcal"] = round(kcal, 3)
    if prot is not None: out["protein_g"] = round(prot, 3)
    if carb is not None: out["carb_g"] = round(carb, 3)
    if fat  is not None: out["fat_g"] = round(fat, 3)

    sodium_mg = None
    for arr_key in ("bad", "good"):
        for n in widget_json.get(arr_key, []) or []:
            name = (n.get("title") or n.get("name") or "").strip().lower()
            if name == "sodium":
                amt = _num_from_str(n.get("amount"))
                unit = (n.get("unit") or "").lower()
                if amt is not None:
                    if unit in ("g", "gram", "grams"):
                        amt *= 1000.0
                    sodium_mg = amt
                    break
        if sodium_mg is not None:
            break
    if sodium_mg is not None:
        out["sodium_mg"] = round(sodium_mg, 3)
    return out


# -----------------------------
# HTTP helper with retries
# -----------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=6))
def _get(url, params=None, timeout=20):
    r = requests.get(url, params=params, timeout=timeout)
    if r.status_code == 402:
        raise RuntimeError("Spoonacular quota exceeded or plan doesn’t allow this request.")
    r.raise_for_status()
    return r

# -----------------------------
# Spoonacular fetcher
# -----------------------------

def fetch_spoonacular(cuisine: str, limit: int) -> List[Dict[str, Any]]:
    api_key = os.getenv("SPOONACULAR_API_KEY")
    if not api_key:
        print("[Spoonacular] WARNING: Missing SPOONACULAR_API_KEY. Skipping Spoonacular source.")
        return []

    # 1) complexSearch：直接开 addRecipeNutrition / addRecipeInformation / addRecipeInstructions / fillIngredients
    results = []
    fetched = 0
    offset = 0
    limit = max(0, limit)
    while fetched < limit:
        page_size = min(100, limit - fetched)
        params = {
            "apiKey": api_key,
            "cuisine": cuisine,
            "instructionsRequired": True,
            "fillIngredients": True,
            "addRecipeInformation": True,
            "addRecipeInstructions": True,
            "addRecipeNutrition": True,   
            "number": page_size,
            "offset": offset,
            "sort": "popularity",
        }
        resp = _get(SPOONACULAR_ENDPOINT, params=params)
        batch = resp.json().get("results", []) or []
        if not batch:
            break
        results.extend(batch)
        fetched += len(batch)
        offset  += len(batch)

    out: List[Dict[str, Any]] = []
    missing_ids: List[str] = []           # 仍然缺营养的条目（备用补齐）
    srcid_to_idx: Dict[str, int] = {}     # 外部ID -> out 中的索引

    # 2) 直接用 complexSearch 的结果构建
    for item in results[:limit]:
        title = (item.get("title") or "Untitled").strip()
        spoonacular_recipe_id = str(item.get("id")) if item.get("id") is not None else None

        # ---- ingredients----
        ings: List[Dict[str, Any]] = []
        for ing in item.get("extendedIngredients") or []:
            ing_name = ing.get("name") or ""
            qty = ing.get("amount")
            unit = ing.get("unit") or ""
            amount_text = f"{qty} {unit}".strip() if qty else unit
            if not amount_text and ing.get("original"):
                amount_text = ing["original"]
            ings.append(make_ingredient(ing_name, amount_text))

        # ---- instructions----
        instructions: List[str] = []
        analyzed = item.get("analyzedInstructions") or []
        if analyzed:
            for instr in analyzed:
                section = norm_space(instr.get("name") or "")
                # 显式按 number 排序；没有 number 的排后面
                steps = sorted(
                    instr.get("steps") or [],
                    key=lambda s: (s.get("number") is None, s.get("number", 0)))
                for s in steps:
                    txt = norm_space(s.get("step") or "")
                    if not txt:
                        continue
                    # 规范化 number，尽量用整数显示；缺失时用 '?'
                    n = s.get("number")
                    if isinstance(n, (int, float)):
                        n_str = str(int(n)) if float(n).is_integer() else str(n)
                    else:
                        n_str = "?"

                    # 组名存在则前缀；否则只打印编号+文本
                    line = f"{section} - 「{n_str}」 - {txt}" if section else f"「{n_str}」 - {txt}"
                    instructions.append(line)
        elif item.get("instructions"):
            instructions = [norm_space(s)
                            for s in re.split(r"[。.\n]", item["instructions"])
                            if norm_space(s)]

        # ---- tags----
        tags = norm_tags((item.get("dishTypes") or []) + (item.get("diets") or []))

        # ---- nutrition：优先用 complexSearch(addRecipeNutrition) 返回 ----
        nutri_list = (item.get("nutrition") or {}).get("nutrients") or []
        per_serving_nutrition = map_spoonacular_nutrition(nutri_list)

        rec = {
            "id": make_id(title, cuisine),
            "title": title,
            "ingredients": ings,
            "instructions": instructions,
            "per_serving_nutrition": per_serving_nutrition,
            "tags": tags,
        }
        out.append(rec)

        # 记录缺营养的，后续补齐
        if not per_serving_nutrition and spoonacular_recipe_id:
            missing_ids.append(spoonacular_recipe_id)
            srcid_to_idx[spoonacular_recipe_id] = len(out) - 1

    # 3) 批量补齐：informationBulk?includeNutrition=true
    if missing_ids:
        for chunk in _chunks(missing_ids, 100):
            info_params = {
                "apiKey": api_key,
                "ids": ",".join(chunk),
                "includeNutrition": True,
            }
            info_data = _get(INFO_BULK_ENDPOINT, params=info_params).json() or []
            for it in info_data:
                sid = str(it.get("id"))
                idx = srcid_to_idx.get(sid)
                if idx is None:
                    continue
                nutri_list2 = (it.get("nutrition") or {}).get("nutrients") or []
                patched = map_spoonacular_nutrition(nutri_list2)
                if patched:
                    out[idx]["per_serving_nutrition"] = patched

    # 4) 仍缺则逐条兜底：nutritionWidget.json
    for sid, idx in list(srcid_to_idx.items()):
        if out[idx]["per_serving_nutrition"]:
            continue
        try:
            wid_url = NUTRITION_WIDGET_TMPL.format(id=sid)
            wid = _get(wid_url, params={"apiKey": api_key}).json()
            patched = _parse_widget_nutrition(wid) or {}
            if patched:
                out[idx]["per_serving_nutrition"] = patched
        except Exception:
            pass

    return out

# -----------------------------
# IO
# -----------------------------
def _safe_stem(s: Any) -> str:
    s = str(s) 
    return re.sub(r"[^a-zA-Z0-9._-]", "_", s.strip()) or "recipe"

def _unique_path(dirpath: Path, stem: str) -> Path:
    p = dirpath / f"{stem}.json"
    if not p.exists():
        return p
    i = 2
    while True:
        p2 = dirpath / f"{stem}-{i}.json"
        if not p2.exists():
            return p2
        i += 1

def save_recipes(recipes: List[Dict[str, Any]], cuisine: str, outdir: Path):
    # 保存为 data/recipes/{cuisine}/{recipe_id}.json
    cuisine_dir = outdir / cuisine
    cuisine_dir.mkdir(parents=True, exist_ok=True)

    index = []  # 可选：生成一个目录索引
    saved = 0
    for r in recipes:
        rid = r.get("id") or "recipe"
        stem = _safe_stem(rid)
        path = _unique_path(cuisine_dir, stem)
        with path.open("w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)
        saved += 1
        index.append({
            "id": r.get("id"),
            "title": r.get("title"),
            "cuisine": cuisine,
            "file": path.name
        })

    ## 去重，重新写recipe的索引
    # 可选：写一个索引文件，便于快速浏览
    index_path = outdir / "_index.json"
    # 读取旧索引
    old = []
    if index_path.exists():
        try:
            with index_path.open("r", encoding="utf-8") as f:
                old = json.load(f)
        except json.JSONDecodeError:
            old = []

    # 合并 & 按 id 去重（后写覆盖前写）
    merged = {str(item["id"]): item for item in old}
    for item in index:  # index 是本次新增的列表
        merged[str(item["id"])] = item
    
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(list(merged.values()), f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {saved} recipes to {cuisine_dir} (one file per recipe).")

# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Fetch recipes to Recipe schema JSON from Spoonacular only.")
    parser.add_argument("--cuisines", nargs="+", required=True, help="Cuisine names, e.g. Chinese Mediterranean")
    parser.add_argument("--per-source", type=int, default=20, help="Max recipes per cuisine from Spoonacular")
    parser.add_argument("--outdir", default="recipes", help="Output directory")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    for cuisine in args.cuisines:
        print(f"\n=== Fetching {cuisine} (Spoonacular) ===")
        spoon = fetch_spoonacular(cuisine, args.per_source)
        merged = dedup_by_title(spoon)
        save_recipes(merged, cuisine, outdir)

if __name__ == "__main__":
    main()
