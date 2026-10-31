---
template_version: nea-0022-v1
stage_id: target_grounding
stage_kind: llm_prompt
input_schema_version: 2
output_schema_version: 2
---

先理解用户任务，再从candidates选择真正支持任务的0到2个概念ID。候选均不贴切、证据不足或不确定时返回空grounded_target_ids，保留原问题，不硬选相近概念。不能从候选之外编造ID，不回答问题。
