---
template_version: nea-formal8-v1
stage_id: formal_defect_audit
stage_kind: llm_prompt
input_schema_version: 1
output_schema_version: 1
---

对question和candidate_answer做独立盲化诊断，不知道实验臂、原诊断或是否受到扰动。仅标注真实妨碍本题解释的五类缺陷：term_gap（必要术语未解释）、step_gap（必要中间推导步骤缺失）、logical_jump（结论衔接缺少成立条件或依据）、term_excess（无助于本题的术语解释）、step_excess（重复或无助于本题的推导）。相关知识不自动成为本题必须展开的内容；不要以篇幅或术语数量制造缺陷。
status为issues/no_issue/uncertain，后两者defects为空。每个缺陷给type、reason和定位：span逐字复制candidate_answer的一个连续子串；遗漏用邻近句作为插入位置，也可用question_anchor引用题目中的对应义务。不使用的字段填空字符串；不能用“全篇”“第几段”或概述代替逐字引文。证据不足时用uncertain，不编造缺陷。
只输出符合schema的JSON实例。
