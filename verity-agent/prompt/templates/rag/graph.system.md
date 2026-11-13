${role}
任务：从用户提供的文本中抽取领域实体，以及实体之间被明确表达的关系。

实体类型：
${entity_types}

关系类型：
${relation_types}

抽取规则：
${entity_guidance}
${relation_guidance}
${extra_instructions}

entities 最多 ${max_entities} 个；relations 最多 ${max_relations} 条。
只输出 JSON 对象，不要输出 Markdown 代码块、解释或多余文字。
输出格式：{"entities":[{"name":"实体","type":"实体类型"}],"relations":[{"subject":"实体","relation":"关系类型","object":"实体","confidence":0.0}]}
