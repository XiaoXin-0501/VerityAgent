判断新事实与候选事实的关系。只输出一个 JSON 对象，不要输出 Markdown 或解释。

输出结构：

{"decision": "compatible|duplicate|supersede|unresolved", "target_id": "候选 id 或空字符串", "reason": "简短理由"}

判定规则：

- duplicate：新事实与某个候选表达相同或同义，target_id 指向该候选。
- supersede：新事实明确替代、纠正或更新某个候选，target_id 指向被替代的候选。
- compatible：两者适用背景不同或可以同时成立，target_id 为空字符串。
- unresolved：信息不足，无法可靠判断，target_id 为空字符串。

约束：

1. 只依据输入的事实文本判断，不引入外部知识。
2. 新旧事实主体不同、时间不同或适用对象不同时，优先判为 compatible。
3. 不确定时输出 unresolved，不要猜测 target_id。
4. reason 控制在 50 字以内。
