# 垂直领域画像目录

领域画像**只辅助 LLM**：提供实体类型、补充词白名单、黑名单、关系白名单和提示词；
实体与关系必须由 LLM 从正文抽取，画像不会直接产出图节点。

> `whitelist` 不是领域全集，也不要求穷举；它主要用于补充 LLM 可能不认识的
> 新药物、新技术、专有名词和别名。内置列表可按需裁剪，不影响未登录实体的抽取。

默认 `pet_medical.json`（宠物医疗）；也可按知识库绑定不同画像。切换领域不需要改代码：

1. 在本目录（或 `IndexerConfig.domain_profile_dir` 指向的目录）新增 `<domain>.json`；
2. 修改 `IndexerConfig.domain_profile`，或在知识库上设置 `domain_profile`；
3. 重建相关 KB 的索引（REINDEX）——实体是在入库时写入 Neo4j 的，换画像不会自动改旧数据。

也可以设置 `IndexerConfig.domain_profile_path` 直接指向一个 JSON 文件（优先级最高）。

## JSON 结构

```jsonc
{
  "name": "pet_medical",            // 必填，唯一标识
  "display_name": "宠物医疗",
  "version": "1",
  "language": "zh",
  "description": "……",

  // 实体类型：name 是写入 LLM 输出的 type；examples 用来展示垂直领域粒度
  "entity_types": [
    {
      "name": "disease",
      "label": "疾病/综合征",
      "description": "具有明确诊断名称的疾病、综合征、感染或病理状态",
      "examples": ["犬瘟热", "犬细小病毒病", "猫传染性腹膜炎"]
    }
  ],

  // 补充词白名单：只放新药物/新技术/专有名词/别名等；不要求穷举领域。
  // 仅当当前 chunk 文本命中时才进入 reference_terms，用于统一标准名。
  "whitelist": [
    {"name": "犬细小病毒病", "type": "disease", "aliases": ["犬细小", "CPV"]}
  ],

  // 明确禁止进入图的关键词：泛化词、版式词、叙述词；提示词与结果后处理都会用
  "blacklist": ["宠物", "主人", "诊断", "治疗", "检查结果", "表1", "图2"],

  // 关系白名单：只有这里声明的关系会写入图；清单外关系只记录日志
  "relations": [
    {"name": "has_symptom", "label": "疾病→症状", "description": "疾病表现出某症状/体征"}
  ],

  // LLM 提示词片段；实体/关系类型清单与黑名单由程序自动追加，
  // 补充词按当前 chunk 命中情况放入 reference_terms
  "prompt": {
    "role": "你是宠物医疗知识图谱的实体与关系抽取器。",
    "entity_guidance": "只抽取文本中明确出现的宠物医疗概念……",
    "relation_guidance": "只抽取正文明确表达的关系……",
    "extra_instructions": ["……"],
    "entity_examples": ["- 犬细小病毒病（disease）……"],
    "noise_examples": ["- 病例报告、患犬、检查结果……"]
  }
}
```

## 校验规则（加载期 fail-fast）

- `entity_types[].name` 不可重复，且至少声明一个类型；
- `entity_types[].label` 不能与另一个类型的 `name/label` 冲突；
- `whitelist` 可以为空；非空时 `name` 不可重复，`type` 必须是已声明类型；
- `whitelist` 不是抽取边界：LLM 可以输出未登录实体；只有类型已声明的实体才入图；
- 同一别名（大小写不敏感、空白归一）不能属于两个补充词；
- 别名不能为空，且长度至少 2（单字中文会高频命中子串，噪声大于收益）；
- 补充词的标准名/别名不能同时出现在 `blacklist`（白名单优先，避免自相矛盾）；
- `relations[].name` 不可重复；只有这里声明的关系会入图，清单外关系仅记录日志；
- `version` 可选；缓存 key 使用画像**内容签名**，内容变化后旧缓存会自动失效。

## 使用边界

- **抽取**：LLM 看 `entity_types`、`blacklist` 与关系规则；`whitelist` 只把
  当前 chunk 文本中实际命中的补充词作为 `reference_terms` 注入，文本中没有
  出现的参考词不会被输出。
- **归一**：LLM 输出的缩写/俗称按 `whitelist` 映射为标准名；词典外实体保留原词。
- **检索**：query 命中 `whitelist` 时作为图 seed 实体；图扩展仍只走画像声明的关系。
- **类型**：LLM 给出的 type 先匹配类型名/标签；未声明类型暂不写入图，只记录
  聚合日志，便于后续扩展 `entity_types`。
- **关系**：未声明关系同样暂不写入图，只记录日志。

## 更换领域时

- 旧画像可以保留在目录里，`domain_profile` 只指向当前使用的画像；
- 不同领域的关系名可以完全不同，无需复用 `is_a/part_of` 等通用名；
- 已建图数据不会自动清理；如需彻底切换，请对 KB 执行全量 REINDEX。
