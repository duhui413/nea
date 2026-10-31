---
template_version: nea-formal8-v1
stage_id: formal_concept_measurement
stage_kind: llm_prompt
input_schema_version: 1
output_schema_version: 1
---

独立盲评答案对每个所列概念的解释程度。参考正文只帮助判断概念含义，不要求答案逐字复述；不得猜测概念的实验标签、答案来源或方法。每个concept_id恰好输出一次，不增加概念。
state只能为absent（未出现）、mentioned（仅提及名称或使用术语但未正确解释其含义及本题作用）、explained（语义正确且解释到位，包括足以理解该概念在本题中的作用）。仅出现关键词、公式或引用不等于解释到位；错误说明不能标为explained。reason简述判断依据。不要因篇幅长而加分。
只输出符合schema的JSON实例。
