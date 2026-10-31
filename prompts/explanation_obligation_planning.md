---
template_version: nea-0022-v1
stage_id: explanation_obligation_planning
stage_kind: llm_prompt
input_schema_version: 2
output_schema_version: 2
---

按原任务安排必须解释的obligations，区分given与uncertain。图候选和needed_prerequisites仅供支撑，不能改变任务或强制覆盖概念。取消固定概念槽位及每概念句数，不推测读者知识。route=direct/decompose仅兼容旧调用。
兼容D1输出gap_concepts时只列任务确需解释者；兼容D3逐候选labels使用gap=确需解释、given=题面已充分给出、not_needed=本题无需展开、uncertain=无法确定。禁止以assumed推断读者掌握；按本次schema输出。
