${role}
任务：从用户提供的文本中抽取领域实体，以及实体之间被明确表达的关系。
目标：以文本语义判断为主，领域词典只用于标准名参考，不要被词典范围限制。

领域词典（辅助参考，不是抽取范围限制；词典未覆盖但文本中明确出现的领域概念也要抽取）：
${dictionary_hint}

实体类型：
${entity_types}

关系类型：
${relation_types}

抽取规则：
${entity_guidance}
${relation_guidance}
${extra_instructions}
- 实体必须是原子领域概念（疾病、药物、症状、检查、治疗、物种、品种、解剖部位等），不能是整句、从句、章节标题、图表编号、文献信息、数值、单位、代词或泛化词。
- 同一实体只输出一次；优先使用文本中最完整的标准名；别名可归一到领域词典标准名。
- 关系必须由文本明确表达；不要根据常识推断；subject/object 必须是本次 entities 中的 name；relation 只能取关系类型清单中的值。
- 不确定时省略，不要编造。

好例子：
${entity_examples}

不要抽取的噪音（示例）：
${noise_examples}

禁止输出的黑名单词（命中即丢弃）：
${keyword_blacklist}

entities 最多 ${max_entities} 个；relations 最多 ${max_relations} 条。
只输出 JSON 对象，不要输出 Markdown 代码块、解释或多余文字。
输出格式：{"entities":[{"name":"实体","type":"实体类型"}],"relations":[{"subject":"实体","relation":"关系类型","object":"实体","confidence":0.0}]}
