# 🥗 Med-Meal Agent

一个基于 **LangChain** 构建的医疗食谱 Agent。  
功能包括：  
- 根据用户健康信息（身高、体重、年龄、疾病史/过敏史/体检报告）推荐个性化食谱  
- 支持冰箱扫描（或手动输入库存），匹配可用食材  
- 自动生成一周食谱计划  
- 根据库存差额生成**精确到重量**的购物清单  

⚠️ **免责声明**：本项目仅用于技术探索和学习，不构成任何医疗建议。饮食和健康方案请务必咨询医生或营养师。

---

## 🚀 功能概览
- **用户画像管理**：支持 TDEE/宏量营养计算，疾病约束（如高血压、糖尿病）  
- **菜谱检索 (RAG)**：基于向量数据库（FAISS/Chroma）检索菜谱  
- **智能 Agent**：调用工具进行：
  - 冰箱扫描解析 → 结构化库存
  - 菜谱检索 + 过滤（过敏原、疾病约束）
  - 一周排餐规划
  - 差额采购清单生成（含重量/数量）
- **API 服务**：通过 FastAPI 提供 REST 接口

---

## 📂 目录结构

med-meal-agent/
├─ i18n/
│ ├─ base_dir/allergens.json, conditions.json, ingredients.json        # 基础词库映射
│ ├─ zh_lexicon.py          # 中英映射词表与工具函数
│ └─ units.py               # 单位与数值换算（mmol→mg/dL、斤/两→g 等）
├─ parsers/
│ └─ health_report_zh.py    # 中文体检报告抽取 → HealthReport
├─ data/recipes/ # 种子菜谱 JSON/MD
├─ data/embeddings/ # 向量库持久化
├─ agents/
│ ├─ tools.py # 工具函数（库存解析、清单生成）
│ ├─ planner.py # Agent 调度与食谱规划
│ └─ constraints.py # 营养/疾病约束与打分规则
├─ models/
│ └─ schemas.py # Pydantic 数据模型
├─ services/
│ ├─ recipe_retriever.py # 向量库构建与检索
│ ├─ nutrition.py # 营养计算辅助函数
│ └─ inventory.py # 库存管理
├─ app.py # FastAPI 服务入口
├─ seed_recipes.py # 构建向量库脚本
├─ requirements.txt
└─ README.md