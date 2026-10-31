---
template_version: nea-formal8-v1
stage_id: formal_perturbation_rewrite
stage_kind: llm_prompt
input_schema_version: 1
output_schema_version: 1
---

你负责构造一个局部受控编辑，不能评价答案。仅按kind执行：
- term_gap：选取解释必要术语定义的连续原文，删除定义而保留该术语名字；term填写仍保留的术语名。
- step_gap：删除一个必要中间推导步骤，保留前提与结论。
- logical_jump：删除连接两个结论的成立条件或依据，保留结论。
- step_excess：确认给定duplicate_paragraph包含推导或论证，再作等义转述；若它只是定义、标题或说明性列举，applicable=false。before_span必须等于该段，after_span给转述，程序负责另外插入它，不删除原文。
- null：仅做等义改写、拆并句或补连接词，保持事实、推导与解释内容，不引入任何上述缺陷。
除step_excess外，before_span必须是candidate_answer中恰好出现一次的非空连续原文，after_span是替换它的新片段（删除可为空），程序只替换该片段。term仅term_gap非空。局部编辑不可能实现时applicable=false，before_span、after_span、term填空字符串并在reason说明；不可强行制造注入真值。
reason说明删除或保留的内容，不输出完整答案。只输出JSON实例。
