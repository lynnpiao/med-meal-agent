# models/schemas.py
"""
Core data schemas for the Med-Meal Agent (Pydantic v2).

- Enumerations for sex, activity level, and units
- User profile and optional health report
- Inventory / recipe / meal plan / shopping list models
- Basic validations and helper utilities

NOTE:
- Keep quantities in SI units when possible (g / ml). `pcs` is allowed
  for count-based items but downstream should convert to grams/milliliters
  via a mapping (see `UnitConverter`).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Literal
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator


# -------------------------
# Enums & Constants
# -------------------------

class Sex(str, Enum):
    male = "male"
    female = "female"


class ActivityLevel(str, Enum):
    sedentary = "sedentary"
    light = "light"
    moderate = "moderate"
    active = "active"
    very_active = "very_active"


class Unit(str, Enum):
    g = "g"        # grams
    ml = "ml"      # milliliters
    pcs = "pcs"    # pieces (count-based)


# 按需扩展的近似换算（示例值，可在运行期由配置覆盖）
DEFAULT_PCS_TO_GRAMS: Dict[str, float] = {
    "egg": 55.0,          # 1 个鸡蛋 ≈ 55g
    "apple": 180.0,       # 1 个苹果 ≈ 180g
    "onion": 120.0,       # 1 个洋葱 ≈ 120g
    "tomato": 120.0,      # 1 个番茄 ≈ 120g
    "garlic clove": 5.0,  # 1 瓣蒜 ≈ 5g
}


class UnitConverter:
    """
    Helper for converting pcs -> grams using a lookup.
    For robust behavior, maintain a normalized key (lowercased).
    """
    def __init__(self, mapping: Optional[Dict[str, float]] = None):
        self.mapping = {k.lower(): v for k, v in (mapping or DEFAULT_PCS_TO_GRAMS).items()}

    def pcs_to_grams(self, name: str, pcs: float) -> Optional[float]:
        w = self.mapping.get(name.lower())
        if w is None:
            return None
        return pcs * w


# -------------------------
# Health-related Schemas
# -------------------------

class HealthReport(BaseModel):
    """
    Optional structured health report snapshot. Values are examples and optional.
    Extend as needed (HbA1c, BP, lipids, etc.).
    """
    # 用途：决定是否需要限钠（如每日 ≤1500–2000 mg）及少加工食品 / 与收缩压一起判定血压控制策略与盐摄入上限
    systolic_bp: Optional[float] = Field(default=None, ge=60, le=250, description="mmHg") # 收缩压，mmHg 参考：<120 正常，130–139 为 1 级高血压边缘，≥140 常视为高血压
    diastolic_bp: Optional[float] = Field(default=None, ge=30, le=200, description="mmHg") # 舒张压，mmHg 参考：<80 正常，80–89 边缘，≥90 偏高
    # 用途：控制碳水比例、添加糖上限，优先全谷物与高纤维配比
    hba1c_percent: Optional[float] = Field(default=None, ge=3.0, le=20.0) # 糖化血红蛋白，% 参考：<5.7 正常，5.7–6.4 糖耐量受损，≥6.5 可提示糖尿病
    # 用途：限制饱和脂肪、反式脂肪，增补不饱和脂肪与膳食纤维
    ldl_mg_dl: Optional[float] = Field(default=None, ge=0, le=400) # 低密度脂蛋白胆固醇（"坏胆固醇"） 参考：<100 理想，100–129 较好，≥160 偏高 
    # 用途：鼓励橄榄油、坚果、鱼类等优质脂肪来源
    hdl_mg_dl: Optional[float] = Field(default=None, ge=0, le=150) # 高密度脂蛋白胆固醇 ("好胆固醇") 参考：男性<40、女性<50 偏低；越高越好
    # 用途：减少精制碳水、酒精与添加糖；提高膳食纤维与 ω-3 脂肪酸摄入
    triglycerides_mg_dl: Optional[float] = Field(default=None, ge=0, le=2000) # 三酰甘油(血脂成分之一) 参考：<150 正常，200–499 偏高，≥500 危险区
    
    # --- optional extensions ---
    # 用途：控制添加糖；优先高纤维、低/中GI主食（全谷物、豆类）；碳水分配更均衡（小份、多次）
    fasting_glucose_mg_dl: Optional[float] = Field(default=None, ge=20, le=600) # 空腹血糖 参考：<100 正常；100–125 糖耐量受损（前期）；≥126 糖尿病（需重复确认）
    ogtt_2h_glucose_mg_dl: Optional[float] = Field(default=None, ge=20, le=800) # 口服葡萄糖耐量试验2小时血糖 参考：<140 正常；140–199 前期；≥200 糖尿病
    # 用途： 减少饱和脂肪/反式脂肪（少油炸、奶油类），多不饱和脂肪（橄榄油、坚果、海鱼），增加可溶性纤维；优先地中海饮食模式
    total_cholesterol_mg_dl: Optional[float] = Field(default=None, ge=50, le=500) # 总胆固醇 参考：<200 mg/dL
    non_hdl_mg_dl: Optional[float] = Field(default=None, ge=0, le=500) # 非HDL胆固醇  参考：<130 mg/dL
    # 用途： 若升高提示肾功能可能受限时，考虑限钠、适度蛋白，并根据医嘱评估钾/磷摄入（菜谱中少用高钾食材的大份量（如大量香蕉、土豆））
    creatinine_mg_dl: Optional[float] = Field(default=None, ge=0.1, le=10) # 肾功能 （肌酐）参考：男约 0.74–1.35，女约 0.59–1.04 mg/dL
    egfr_ml_min_1_73m2: Optional[float] = Field(default=None, ge=1, le=200) # 肾功能 （估算肾小球滤过率） 参考：eGFR <60 ml/min/1.73m² 持续 ≥3 个月可提示慢性肾病
    # 用途： 异常升高时减少酒精、高糖、高脂与重油炸；偏向高纤维、适量优质脂肪与充足蔬果的清淡烹调
    alt_u_l: Optional[float] = Field(default=None, ge=0, le=1000) # 肝功能（丙氨酸转氨酶）参考：ALT 约 7–55 U/L 
    ast_u_l: Optional[float] = Field(default=None, ge=0, le=1000) # 肝功能（天冬氨酸转氨酶）参考：AST 约 8–48 U/L
    # 用途： 高尿酸/痛风倾向时，减少高嘌呤（动物内脏、部分海鲜）、限制酒精尤其啤酒与果糖甜饮；足量饮水
    uric_acid_mg_dl: Optional[float] = Field(default=None, ge=0.5, le=20) # 尿酸 参考：成年男约 3.5–7.2 mg/dL，女约 2.7–7.3 mg/dL；>6.8 mg/dL 易过饱和并与痛风相关
    # 饮食对TSH 影响有限
    tsh_u_iu_ml: Optional[float] = Field(default=None, ge=0.01, le=100)# 促甲状腺激素  参考：成人常用范围约 0.4–4.0 mIU/L
    # 用途： 偏低时，增加富含维D食物（多脂鱼类、强化奶/植物奶、蛋黄）
    vitamin_d_25oh_ng_ml: Optional[float] = Field(default=None, ge=2, le=150) # 25-羟维生素D 参考（NIH/ODS）：<12 ng/mL 缺乏；12–19.6 ng/mL 可能不足；≥20 ng/mL 多数人群被视为足够
    # 用途： 偏低时（贫血风险）增加富铁（红肉、动物肝、豆类、深绿叶菜），与维C同餐促进吸收；茶/咖啡与富铁餐错开
    hemoglobin_g_dl: Optional[float] = Field(default=None, ge=3, le=25) # 血红蛋白 参考：男约 13.2–16.6，女约 11.6–15.0 g/dL
    ferritin_ng_ml: Optional[float] = Field(default=None, ge=1, le=2000) # 铁蛋白 参考：女约 15–205 ng/mL，男约 30–566 ng/mL

    gestational_weeks: Optional[float] = Field(default=None, ge=0, le=42) # 孕周
    gdm: Optional[bool] = None # 妊娠期糖尿病
    medications: List[str] = [] # 用药清单
    notes: Optional[str] = None #备注


class UserProfile(BaseModel):
    name: Optional[str] = None
    age: int = Field(..., ge=1, le=120)
    sex: Sex
    height_cm: float = Field(..., gt=30, lt=260)
    weight_kg: float = Field(..., gt=2, lt=400)
    activity_level: ActivityLevel = ActivityLevel.light
    # e.g. ["hypertension", "t2dm", "hyperlipidemia"]
    conditions: List[str] = Field(default_factory=list)
    # e.g. ["peanut", "shellfish", "lactose"]
    allergies: List[str] = Field(default_factory=list)
    # Whether to store/report sensitive health data
    share_health_data: bool = Field(default=False)
    health_report: Optional[HealthReport] = None

    @field_validator("conditions", "allergies")
    @classmethod
    def dedup_and_norm(cls, v: List[str]) -> List[str]:
        # 去重 + 去首尾空格 + 小写
        seen = set()
        out: List[str] = []
        for x in v:
            k = x.strip().lower()
            if k and k not in seen:
                seen.add(k)
                out.append(k)
        return out


# -------------------------
# Inventory & Recipe Schemas
# -------------------------

class InventoryItem(BaseModel):
    """
    Represents a single ingredient with quantity.
    Prefer g/ml for consistent downstream nutrition math.
    """
    name: str = Field(..., min_length=1)
    quantity: float = Field(..., ge=0)
    unit: Unit = Unit.g

    @field_validator("name")
    @classmethod
    def normalize_name(cls, v: str) -> str:
        return v.strip()

    def approx_grams(self, converter: Optional[UnitConverter] = None) -> Optional[float]:
        """
        Returns grams if unit is already grams.
        If unit is pcs, try to approx grams using converter (or default).
        If unit is ml, returns None (conversion is density-dependent).
        """
        if self.unit == Unit.g:
            return self.quantity
        if self.unit == Unit.pcs:
            grams = (converter or UnitConverter()).pcs_to_grams(self.name, self.quantity)
            return grams
        return None  # ml -> g 需要密度，放到上层处理


class Recipe(BaseModel):
    """
    A single recipe with per-serving nutrition.
    """
    id: str
    title: str
    ingredients: List[InventoryItem]
    instructions: List[str] = Field(default_factory=list)
    # e.g. {"kcal": 520, "protein_g": 30, "carb_g": 60, "fat_g": 18, "sodium_mg": 800}
    per_serving_nutrition: Dict[str, float] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)  # e.g. ["low-sodium","high-protein"]

    @field_validator("title")
    @classmethod
    def title_non_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Recipe.title cannot be empty.")
        return v

    @field_validator("tags")
    @classmethod
    def norm_tags(cls, v: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for x in v:
            k = x.strip().lower()
            if k and k not in seen:
                seen.add(k)
                out.append(k)
        return out


# -------------------------
# Planning & Shopping Schemas
# -------------------------

class MealPlanDay(BaseModel):
    """
    A day's meal plan, storing recipe IDs (referencing Recipe.id).
    """
    date: Optional[str] = None  # ISO date string if needed
    meals: List[str] = Field(default_factory=list)  # list of recipe ids

    @field_validator("meals")
    @classmethod
    def dedup_meals(cls, v: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for rid in v:
            r = rid.strip()
            if r and r not in seen:
                seen.add(r)
                out.append(r)
        return out


class MealPlan(BaseModel):
    """
    7-day (or N-day) meal plan and daily target macros.
    daily_targets example: {"kcal": 2000, "protein_g": 100, "fat_g": 65, "carb_g": 250, "sodium_mg": 1500}
    """
    days: List[MealPlanDay]
    daily_targets: Dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def ensure_non_empty(self) -> "MealPlan":
        if not self.days:
            raise ValueError("MealPlan.days cannot be empty.")
        return self


class ShoppingItem(BaseModel):
    name: str
    quantity: float = Field(..., ge=0)
    unit: Unit = Unit.g

    @field_validator("name")
    @classmethod
    def clean_name(cls, v: str) -> str:
        return v.strip()


class ShoppingList(BaseModel):
    items: List[ShoppingItem] = Field(default_factory=list)

    @field_validator("items")
    @classmethod
    def merge_duplicates(cls, v: List[ShoppingItem]) -> List[ShoppingItem]:
        """
        Merge duplicate items with same (name, unit) by summing quantities.
        """
        agg: Dict[tuple, float] = {}
        for it in v:
            key = (it.name.strip().lower(), it.unit.value)
            agg[key] = agg.get(key, 0.0) + it.quantity
        out: List[ShoppingItem] = []
        for (name, unit), qty in agg.items():
            out.append(ShoppingItem(name=name, quantity=round(qty, 3), unit=Unit(unit)))
        return out


# -------------------------
# Convenience: TDEE & Macro Targets (optional helpers)
# -------------------------

def estimate_tdee_kcal(profile: UserProfile) -> float:
    """
    Mifflin–St Jeor (simplified) + activity factor.
    """
    s = 5 if profile.sex == Sex.male else -161
    bmr = 10 * profile.weight_kg + 6.25 * profile.height_cm - 5 * profile.age + s
    factor = {
        ActivityLevel.sedentary: 1.2,
        ActivityLevel.light: 1.375,
        ActivityLevel.moderate: 1.55,
        ActivityLevel.active: 1.725,
        ActivityLevel.very_active: 1.9,
    }[profile.activity_level]
    return float(bmr * factor)


def macro_targets(profile: UserProfile, tdee: Optional[float] = None) -> Dict[str, float]:
    """
    Very simple macro heuristic:
    - Protein: 1.6 g/kg
    - Fat: 30% kcals
    - Carbs: remainder
    """
    tdee = tdee if tdee is not None else estimate_tdee_kcal(profile)
    protein_g = 1.6 * profile.weight_kg
    fat_kcal = 0.30 * tdee
    fat_g = fat_kcal / 9.0
    carb_kcal = max(0.0, tdee - (protein_g * 4.0 + fat_g * 9.0))
    carb_g = carb_kcal / 4.0
    return {
        "kcal": float(round(tdee, 3)),
        "protein_g": float(round(protein_g, 3)),
        "fat_g": float(round(fat_g, 3)),
        "carb_g": float(round(carb_g, 3)),
    }
    

def disease_rules() -> Dict[str, Dict]:
    # 极简示例：实际可更细化（与营养师合作）
    return {
        "hypertension": {"sodium_mg_max_per_day": 1500},
        "t2dm": {"added_sugar_g_max": 25, "carb_ratio_max": 0.50},
        "hyperlipidemia": {"sat_fat_g_max": 20},
    }


def score_recipe_against_profile(recipe: Recipe, profile: UserProfile) -> float:
    # 极简：若含过敏原→强惩罚；若钠/糖/饱和脂肪超限→扣分；宏量营养接近目标→加分
    score = 0.0
    # 过敏检查（极简：名称包含判断）
    for a in profile.allergies:
        if any(a.lower() in ing.name.lower() for ing in recipe.ingredients):
            return -1e6
    # 可加更多细则...
    return score
