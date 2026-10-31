---
template_version: nea-0032-v1
stage_id: revision_validation
stage_kind: llm_prompt
input_schema_version: 2
output_schema_version: 2
---

核对父答案answer与具体候选candidate_answer，结合原任务、obligations、actions和相同允许证据。检查义务完成、有效条件的语义保留、原缺陷是否修正、是否新增错误或无支持主张。按具体span/question_anchor说明理由与证据。
status=pass/fail/uncertain；任何未修错误、新增错误、证据冲突为fail或uncertain，无支持不能通过。程序负责绑定父/候选hash；不要自行编造身份。只做核验，不修改答案。
分别检查父答案已有错误与候选新引入错误；不要因为动作已执行就放行。新增内容须核对question_anchor和necessity_reason是否真正构成题面必要义务：仅与chunk、图或计划相关不是必要性证据；非空理由不代表理由成立。特别保留“最容易、常见、必须”等限定，不把无频率依据的相关知识包装成常见误区。scope_effect=correction的候选若实际增加解释义务，也按相同标准核对。

findings中的定位按逐字子串校验。span只从父答案answer或候选candidate_answer复制一段短的连续原文，保持标点、空格、Markdown和数学符号；不要转述、拼接多段、省略中间文本或写“全篇”。question_anchor只复制question中的连续原文；不用的字段填空字符串。两字段至少一项非空；若都填写，两者都必须合法，不能用正确的anchor掩盖改写后的span。解释写在reason。evidence_ids只使用本题knowledge_chunks的chunk_id，PE填空数组。
只输出JSON实例，顶层恰为status、reason、findings；每个finding恰为span、question_anchor、reason、evidence_status、evidence_ids。不要混入schema定义字段。无发现可用空findings，不能为通过校验而隐瞒已发现的问题。
