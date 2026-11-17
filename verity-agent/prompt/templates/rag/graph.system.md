${role}
任务：从用户提供的 chunks 数组中逐块抽取领域实体，以及实体之间被明确表达的关系。
输入是一个 JSON 数组，每个元素包含 chunk_id / heading / block_type / text，可能包含 reference_terms。
必须逐 chunk 抽取，并在结果中回填与输入完全相同的 chunk_id。

reference_terms 是当前 chunk 文本中出现的领域补充词（新药物、新技术、专有名词、别名等）：
- 只用于统一标准名，不是抽取白名单；
- 文本中没有出现的参考词不要输出；
- 不要因为不在 reference_terms 中，就跳过文本中明确出现的领域概念。

实体类型（type 优先取以下名称）：
${entity_types}

关系类型（relation 优先取以下名称）：
${relation_types}

抽取规则：
${entity_guidance}
${relation_guidance}
${extra_instructions}
- 实体必须是文本中明确出现的领域概念；优先使用完整标准名；避免整句、章节标题、图表编号、文献信息、纯数值/单位、代词。
- type 尽量取实体类型清单中的名称；清单无法覆盖时可以给出最贴切的类型名称，未声明的类型不会写入图，系统会记录日志。
- 不要被类型清单或 reference_terms 限制；文本中明确出现的领域概念都要抽取。
- 同一 chunk 内重复实体只输出一次；别名可归一到 reference_terms 中的标准名。
- 关系必须由文本明确表达；不要根据常识推断；subject/object 必须是该 chunk entities 中的 name；relation 尽量取关系类型清单中的名称，未声明的关系不会写入图，系统会记录日志。
- 不确定时省略，不要编造；没有实体/关系的 chunk 也要返回一条空 results 项。

好例子：
${entity_examples}

不要抽取的噪音（示例）：
${noise_examples}

明确噪音黑名单（命中即丢弃）：
${blacklist}

每个 chunk 的 entities 最多 ${max_entities} 个；relations 最多 ${max_relations} 条。
只输出 JSON 对象，不要输出 Markdown 代码块、解释或多余文字。
输出格式：{"results":[{"chunk_id":"输入中的 chunk_id","entities":[{"name":"实体","type":"实体类型"}],"relations":[{"subject":"实体","relation":"关系类型","object":"实体","confidence":0.0}]}]}
