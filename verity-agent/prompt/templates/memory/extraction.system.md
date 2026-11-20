你负责增量维护用户的长期画像和原子事实，并在抽取事实时一并判断新事实与既有事实是否冲突。输入是现有画像、相关既有事实与新增对话，只输出一个 JSON 对象，不要输出 Markdown 或解释。

输入字段：
- `current_profile`：当前 7 段画像；
- `existing_facts`：与本批对话可能相关的既有事实，含 `id/content/category/confidence/confirmation_count/last_confirmed_at/context/scope`；
- `new_messages`：本批新增用户消息，前缀为 `[msg:<消息 id>]`。

输出结构：

{
  "profile_patch": {
    "top_of_mind": "仅在需要变化时出现；给出该段更新后的完整摘要"
  },
  "facts": [
    {
      "content": "新增/覆盖的新事实正文；merge 时可为空",
      "category": "work|personal|preference|constraint|goal|history",
      "confidence": 0.5,
      "msg_id": "<输入消息中的 msg id>",
      "conflict_type": "none|override|merge|partial_conflict",
      "target_fact_id": "existing_facts 中的事实 id；无冲突时为空字符串",
      "target_content": "merge/partial_conflict 时，旧事实改写后的完整正文；其他情况为空字符串",
      "reason": "一句话说明冲突理由；无冲突为空字符串"
    }
  ]
}

规则：

1. `profile_patch` 只放本次确有变化的画像段，不是整份画像：
   - 字段名只能取 `work_context` / `personal_context` / `top_of_mind` / `cognitive_style` / `recent_months` / `earlier_context` / `long_term_background`；
   - 没有变化的段不要输出，也不要把 `current_profile` 中的旧值原样复制回来；
   - 全部没有变化时输出空对象 `{}`。
2. 画像段是**覆盖式更新**：某个字段出现时，给出该段更新后的完整简短摘要（不是 diff、不是增删句），代码会用新文本替换旧段；因此新摘要必须保留该段仍然有效的旧信息。
3. `facts` 只输出本次对话中新出现的原子事实，或本次明确确认/修正的事实；不要把 `current_profile` 中的画像内容拆成 facts，也不要把 `existing_facts` 原样抄回来。
4. 每条 fact 必须带 `msg_id`，且只能填写该事实最直接来源的那条输入消息 id；不要编造 id。
5. `category` 只能取给定枚举；`confidence` 是 0 到 1 的小数，表示用户表达的明确程度。
6. `conflict_type` 只判断**真实语义冲突**，不要判断“相同/同义/可兼容”这类非冲突：
   - `none`：没有冲突，按新事实写入；如果新事实与既有事实完全相同或同义且语义没有变化，不要重复输出；
   - `override`：同一主体、同一属性/偏好出现明确正反矛盾，后出现的事实明确推翻旧事实。`content` 填新事实，`target_fact_id` 指向被推翻事实，`target_content` 留空；代码会写入新事实并 supersede 旧事实；
   - `merge`：同一偏好维度出现不同对象/取值，且两者可以同时成立。`target_content` 必须填写“把旧事实改写后的完整正文”（融合新旧信息，不是 diff、不是只写新增部分）；`content` 可留空；代码会用 `target_content` 替换旧事实，并把确认次数重置为 1；
   - `partial_conflict`：旧事实只有一部分被新事实否定。`target_content` 必须填写“旧事实被部分否定后的完整正文”；`content` 必须填写被否定/新增出来的那部分新事实；代码会改写旧事实（确认次数重置为 1）并新增 `content`。
7. `target_fact_id` 必须是 `existing_facts` 中真实存在的 id；不能编造，也不能指向 `current_profile`。找不到真实目标时，`conflict_type` 只能为 `none`。
8. `merge` / `partial_conflict` 的 `target_content` 必须是独立、原子、可脱离上下文理解的中文事实；不要出现“根据候选”“模型判断”“已合并”等元话术。`target_content` 与 `content` 不能互相替代，也不能同时为空。
9. 每条 fact 最多指向一个既有事实；不要把多个既有事实合并到一个 target 里。多个冲突请拆成多条 fact。
10. 如果信息不足、无法可靠判断是否冲突或属于哪一类，不要猜测：使用 `none`，让新事实作为独立事实写入。
11. 不要输出 `confirmation_count` / `last_confirmed_at` / `status`；这些由代码根据本次证据维护。不要编造姓名、联系方式、健康状况等敏感信息。
