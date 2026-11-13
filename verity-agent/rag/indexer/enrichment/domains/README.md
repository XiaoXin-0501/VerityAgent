# 垂直领域画像目录

一个部署只加载 **一个** 领域画像；默认 `medical.json`。切换领域不需要改代码：

1. 在本目录（或 `IndexerConfig.domain_profile_dir` 指向的目录）新增 `<domain>.json`；
2. 修改 `IndexerConfig.domain_profile = "<domain>"`；
3. 重建相关 KB 的索引（REINDEX）——关键词是在入库时写入 Neo4j 的，换画像不会自动改旧数据。

也可以设置 `IndexerConfig.domain_profile_path` 直接指向一个 JSON 文件（优先级最高）。

## JSON 结构

```jsonc
{
  "name": "medical",                 // 必填，唯一标识
  "display_name": "医疗",
  "version": "1",
  "language": "zh",
  "description": "……",

  // 实体类型：name 必须被 entities[].type 引用；weight 参与关键词排序
  "entity_types": [
    {"name": "disease", "label": "疾病", "weight": 1.3},
    {"name": "drug", "label": "药物", "weight": 1.3}
  ],

  // 实体词典：标准名 + 别名（缩写/俗称/英文名）；写入图的是标准名
  "entities": [
    {"name": "高血压", "type": "disease", "aliases": ["HTN", "hypertension"]},
    {"name": "二甲双胍", "type": "drug", "aliases": ["metformin"]}
  ],

  // 领域停用词：无检索价值的通用词，不进入关键词
  "stopwords": ["患者", "诊断", "治疗"],

  // 关系白名单：LLM 只能从这里取值；词典模式通过 RelationExtractor 也可使用
  "relations": [
    {"name": "treated_by", "label": "疾病→治疗", "description": "疾病使用某药物/治疗"}
  ],

  // LLM 提示词片段；实体/关系类型清单由程序自动追加
  "prompt": {
    "role": "你是医疗知识图谱的实体与关系抽取器。",
    "entity_guidance": "只抽取文本中明确出现的医学概念……",
    "relation_guidance": "只抽取正文明确表达的关系……",
    "extra_instructions": ["……"]
  },

  // 领域词不足时是否补充通用名词
  "include_generic_terms": true,
  "generic_candidate_factor": 2
}
```

## 校验规则（加载期 fail-fast）

- `entity_types[].name` 不可重复，且至少声明一个类型；
- 至少声明一个实体；
- `entities[].name` 不可重复，`type` 必须是已声明类型；
- 同一别名（大小写不敏感、空白归一）不能属于两个实体；
- 别名不能为空，且长度至少 2（单字中文会高频命中子串，噪声大于收益）；
- 标准名与别名都会参与匹配。

## 匹配行为

- 标准名与别名按 **最长优先** 匹配：`2型糖尿病` 优先于 `糖尿病`；
- ASCII 词加字母数字边界，`HTN` 不会命中 `XHTNX`；
- 命中结果先归一为标准名，再进行关键词打分/关系过滤。

## 更换领域时

- 旧画像可以保留在目录里，但 `domain_profile` 只指向一个；
- 不同领域的关系名可以完全不同，无需复用 `is_a/part_of` 等通用名；
- 已建图数据不会自动清理；如需彻底切换，请对 KB 执行全量 REINDEX。
