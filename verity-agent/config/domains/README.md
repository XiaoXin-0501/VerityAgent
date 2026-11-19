# 垂直领域画像目录

领域画像**只做辅助**：提供类型/关系白黑名单、实体别名与提示词，用来提高
抽取质量、收缩明显噪声；它**不是实体/关系全集，也不做硬穷举**。
实体与关系必须由 LLM 从正文抽取，画像不直接生成图节点，也不生成检索 seed。

> 白名单/黑名单只约束**类型和关系**；实体标准名/别名单独放在 `aliases`。
> 默认声明的 `entity_types` / `relations` 都可以抽取、入库和召回；白名单是
> 补充，黑名单是限制/过滤。

## 主画像与子画像

画像分两层，加载时自动合并：

- **主画像**（`master` 为空）：一个领域主题的公共底座，例如
  `pet_medical.json`（宠物医疗）。提供公共类型、公共关系、公共类型/关系
  白黑名单、公共别名和提示词。
- **子画像**（`master` 指向主画像）：绑定具体知识库，只声明该库特有的
  类型/关系、类型/关系白黑名单与别名；公共部分继承主画像。

合并规则：

- `entity_types` / `relations`：按 `name` 合并，子画像同名项覆盖主画像；
- `type_whitelist` / `relation_whitelist`：按 `name` 合并；
- `type_blacklist` / `relation_blacklist` / `noise_terms`：保序并集；
- `aliases`：按 `name` 合并；同名条目继承主画像别名并叠加子画像别名；
- `prompt`：字符串子画像非空覆盖，列表拼接去重。

## Prompt 携带什么

图抽取 system prompt 只携带：

- 有效 `entity_types`（默认类型 + 类型白名单 - 类型黑名单）；
- 有效 `relations`（默认关系 + 关系白名单 - 关系黑名单）；
- `aliases` 的**标准名**（不含别名值）；
- 角色、抽取规则、示例和数量上限。

不携带：

- `type_blacklist` / `relation_blacklist`：只做抽取后过滤；
- `noise_terms`：只做实体名过滤；
- `aliases` 的别名值：只做抽取后归一。

## JSON 结构

```jsonc
{
  "name": "pet_medical",
  "display_name": "宠物医疗",
  "version": "2",
  "language": "zh",
  "description": "……",
  "master": "",

  // 默认实体类型：全部可抽取；白名单可补充，黑名单可移除
  "entity_types": [
    {
      "name": "disease",
      "label": "疾病/综合征",
      "description": "具有明确诊断名称的疾病、综合征、感染或病理状态",
      "examples": ["犬瘟热", "犬细小病毒病", "猫传染性腹膜炎"]
    }
  ],

  // 类型白名单/黑名单：只影响类型集合
  "type_whitelist": [
    {"name": "lab_parameter", "label": "检验参数"}
  ],
  "type_blacklist": [],

  // 默认关系：全部可抽取；白名单可补充，黑名单可移除
  "relations": [
    {"name": "has_symptom", "label": "疾病→症状", "description": "疾病表现出某症状/体征"}
  ],
  "relation_whitelist": [
    {"name": "monitored_by", "label": "疾病→监测手段"}
  ],
  "relation_blacklist": ["mentions", "related_to"],

  // 实体标准名与别名；标准名进入 Prompt，别名值只用于抽取后归一
  "aliases": [
    {"name": "犬细小病毒病", "type": "disease", "aliases": ["犬细小", "CPV"]},
    {"name": "血常规", "type": "exam", "aliases": ["CBC"]}
  ],

  // 实体名噪声过滤（版式词/泛化词/叙述词）；不进入 Prompt
  "noise_terms": ["宠物", "主人", "诊断", "治疗", "检查结果", "表1", "图2"],

  // Prompt 片段；类型/关系/别名标准名清单由程序自动追加
  "prompt": {
    "role": "你是宠物医疗知识图谱的实体与关系抽取器。",
    "entity_guidance": "只抽取文本中明确出现的宠物医疗概念……",
    "relation_guidance": "只抽取正文明确表达的关系……",
    "extra_instructions": ["……"],
    "entity_examples": ["- 犬细小病毒病（disease）"],
    "noise_examples": ["- 病例报告、患犬、检查结果……"]
  }
}
```

## 校验规则（加载期 fail-fast）

- `entity_types` + `type_whitelist` - `type_blacklist` 后至少保留一个有效类型；
- 类型/关系名称不可重复；类型 label 不能与另一个类型 name/label 冲突；
- `aliases[].type` 必须是有效类型；
- 同一别名（大小写不敏感、空白归一）不能属于两个实体；
- 别名不能为空，且长度至少 2；
- 实体标准名/别名不能同时出现在 `noise_terms`；
- `relations` + `relation_whitelist` - `relation_blacklist` 为有效关系集合；
- `version` 可选；缓存 key 使用画像内容签名，内容变化后旧缓存自动失效。

## 使用边界

- **抽取**：LLM 看有效类型、有效关系、别名标准名和抽取规则；清单外概念仍可抽取，
  由 `graph_allow_unknown_types` / `graph_allow_unknown_relations` 控制是否入库。
- **过滤**：实体名命中 `noise_terms`，或实体类型/关系命中类型/关系黑名单后丢弃。
- **归一**：LLM 输出的别名/俗称按 `aliases` 归一为标准名；词典外实体保留原词。
- **类型**：优先解析为有效类型；未知类型按开关保留为合法标识或 `other`。
- **关系**：默认声明与白名单关系可抽取；黑名单关系过滤；未知关系按开关保留证据。
- **检索**：图下一跳不按画像关系枚举限制，所有入库合法关系都可参与扩展。
- **seed**：不扫描 `aliases` 生成图 seed；规划器/BM25 explainScore 提取的实体
  只做别名归一与噪声词过滤。

## 更换领域

1. 在 `config/domains/`（或 `IndexerConfig.domain_profile_dir`）新增主/子画像；
2. 设置 `IndexerConfig.master_profile` / `domain_profile`，或在知识库上绑定
   `domain_profile`（通常写子画像名）；
3. 对相关 KB 执行 REINDEX——图是在入库时写入 Neo4j 的，换画像不会自动改旧数据。

`medical.json` 是通用人类医疗参考主画像，不参与宠物医疗默认链路；只有知识库
显式绑定 `medical` 时才会使用。
