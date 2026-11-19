${role}
任务：从用户提供的 chunks 数组中逐块抽取领域实体，以及实体之间被明确表达的关系。
输入是一个 JSON 数组，每个元素包含 chunk_id / heading / block_type / text。
必须逐 chunk 抽取，并在结果中回填与输入完全相同的 chunk_id。

实体类型（type 优先取以下名称，但不是穷举白名单）：
${entity_types}

关系类型（relation 优先取以下名称，但不是穷举白名单）：
${relation_types}

领域已知标准名（来自 aliases，可优先使用；不要求穷举，也不是抽取边界）：
${alias_names}

抽取规则：
${entity_guidance}
${relation_guidance}
${extra_instructions}
- 实体必须是文本中明确出现的领域概念；优先使用完整标准名；避免整句、章节标题、图表编号、文献信息、纯数值/单位、代词。
- type 尽量取实体类型清单中的名称；清单无法覆盖时可以给出最贴切的类型名称，系统会保留并记录日志，不会因为类型未登记而直接丢弃实体。
- 不要被类型清单或已知标准名限制；文本中明确出现的领域概念都要抽取。
- 同一 chunk 内重复实体只输出一次；同一实体有多个叫法时优先输出领域已知标准名。
- 关系必须由文本明确表达；不要根据常识推断；subject/object 必须是该 chunk entities 中的 name；relation 尽量取关系类型清单中的名称，未声明关系会记录日志，并按系统配置决定是否保留。
- 不确定时省略，不要编造；没有实体/关系的 chunk 也要返回一条空 results 项。

好例子：
${entity_examples}

不要抽取的噪音（示例）：
${noise_examples}

每个 chunk 的 entities 最多 ${max_entities} 个；relations 最多 ${max_relations} 条。
只输出 JSON 对象，不要输出 Markdown 代码块、解释或多余文字。
输出格式：{"results":[{"chunk_id":"输入中的 chunk_id","entities":[{"name":"实体","type":"实体类型"}],"relations":[{"subject":"实体","relation":"关系类型","object":"实体","confidence":0.0}]}]}
