---
template_version: nea-0022-v1
stage_id: query_task_analysis
stage_kind: llm_prompt
input_schema_version: 2
output_schema_version: 2
---

分析原问题，提取必须回答的任务义务obligations、题面已给定given和未能确定uncertain。每条义务应可据问题定位，尤其保留比较方向、解释原因、条件及限制。needed_prerequisites仅是支撑建议；无需补充时为空。route只是兼容标记，不决定段落或句数。不回答问题。
