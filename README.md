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

<details>
<summary>📂 目录结构（点击展开）</summary>

<pre>
&lt;pre&gt; med-meal-agent/
├─ .env                                      # 本地测试环境变量（示例；勿提交）
├─ i18n/
│  ├─ base_dir/
│  │  ├─ allergens.json                      # 基础词库映射(过敏原映射)
│  │  ├─ conditions.json                     # 基础词库映射(疾病/状况映射)
│  │  └─ ingredients.json                    # 基础词库映射(食材映射)
│  ├─ profile_enum_map.py                    # Profile内中英映射词表与工具函数
│  ├─ zh_lexicon.py                          # 中英映射词表与工具函数
│  └─ units.py                               # 单位与数值换算（mmol→mg/dL、斤/两→g 等）
├─ parsers/
│  ├─ health_report_zh.py                    # 中文体检报告抽取 → HealthReport
│  ├─ user_profiles_zh.py                    # 中文userfile（sex/activity/conditions/allergens）抽取
│  └─ base_dir/                              # 测试/演示用样例素材
│     ├─ test.txt                            # OCR 文本样例
│     ├─ test11.jpg                          # 图片样例（血压页）
│     └─ test22.jpg                          # 图片样例（化验页）
├─ data/
│  ├─ recipes/                               # 种子菜谱 JSON
│  ├─ embeddings/                            # 向量库持久化
│  ├─ fetch_recipes.py                       # 从Spoonacular API 获取recipe，并生成种子菜谱JSON
│  └─ zh_en_synonyms.yaml                    # 中英文菜谱内部词汇互换，支持ElasticSearch查询
├─ agents/
│  ├─ tools.py                               # 工具函数（库存解析、清单生成）
│  ├─ planner.py                             # Agent 调度与食谱规划
│  └─ constraints.py                         # 营养/疾病约束与打分规则
├─ models/
│  └─ schemas.py                             # Pydantic 数据模型
├─ services/
│  ├─ recipe_retriever.py                    # 向量库构建与检索
│  ├─ nutrition.py                           # 营养计算辅助函数
│  ├─ vision_inventory.py                    # 库存管理（冰箱图片扫描OCR- LLM or OWL-ViT） 
│  └─ inventory.py                           # 库存管理（text/购物小票OCR） 
├─ tests/                                    # 单元/集成测试
│  ├─ conftest.py                            # pytest 配置（加载 .env、注册标记等）
│  ├─ test_models_userprofile_zh.py          # user profile离线/伪造测试
│  ├─ test_parsers_health_report_zh_full.py  # 全量逻辑的离线/伪造测试
│  ├─ test_parsers_health_report_zh_live.py  # 调真实 LLM 的 live 测试
│  ├─ test_i18n_lexicon.py                   # 词库加载/映射测试
│  └─ test_i18n_units.py                     # 单位/数值换算测试
├─ app.py                                    # FastAPI 服务入口
├─ seed_recipes.py                           # 构建向量库脚本
├─ requirements.txt
└─ README.md
&lt;/pre&gt;
</pre>
</details>