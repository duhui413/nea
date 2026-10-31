---
template_version: nea-0022-v1
stage_id: answer_quality_evaluation
stage_kind: llm_prompt
input_schema_version: 2
output_schema_version: 2
---

独立盲评question与candidate_answer，依据rubric六维各给1—5整数。评价依据不得包含臂名、内部诊断或修订通过状态。不因篇幅、概念数量或行文流畅掩盖关键错误。
每维reasons说明可定位的span（答案原文）或question_anchor（遗漏任务）、判断reason、evidence_status与evidence_ids。无法确定时明确uncertain。findings记录关键错误跨度/违反任务及依据；无问题可为空。status=no_issue/issues/uncertain与发现一致。回退答案中的错误照常计入。

定位是逐字子串校验，不是语义相似匹配。每个reasons条目和每个finding至少填写一个定位字段：
- span必须从candidate_answer复制一段短的、连续的原文，保持标点、空格、换行、Markdown和数学符号。不要转述、拼接多处、用省略号替代中间文本，或填写“全篇”“第几节”等描述。优先引用无需跨行的短语。
- question_anchor只复制question中的连续原文。无需该定位字段时填空字符串；若span与question_anchor都非空，两者都必须分别逐字匹配。一个合法字段不能抵消另一个字段的伪造引用。
- 正面评价也给出实际短引文；遗漏任务可用question_anchor并把span留空。解释与概括写在reason，不写入定位字段。
- evidence_ids只能使用knowledge_chunks给出的chunk_id；PE填空数组。不要把模型知识或问题文字编成证据ID。

只返回JSON实例，顶层恰为scores、reasons、status、findings。reasons的每个维度及每个finding恰为span、question_anchor、reason、evidence_status、evidence_ids。不要输出additionalProperties、properties、required、type等schema定义字段。提交前逐项核对定位字段确实存在于对应输入中。
