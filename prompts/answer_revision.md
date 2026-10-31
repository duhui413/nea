---
template_version: nea-0032-v1
stage_id: answer_revision
stage_kind: llm_prompt
input_schema_version: 2
output_schema_version: 2
---

依据已核准actions（SR使用issues）作一次有界修订，输出完整答案正文。遵守max_chars及allowed_citations；保留已经核对的有效条件的语义与支持关系，允许改写措辞。旧required_sentences/required_citations不是逐字或逐ID保护依据。
有依据时改正事实、推导、错误成立条件及错引，不能只润色连接句。不能使用允许证据之外的正文。no_issue/uncertain时保持父答案，不能强行修订。该候选必须交由独立核验接口，不自称已通过。
动作附有question_anchor及necessity_reason时，只补充完成该题面要求所必需的最小内容；检查必要性理由是否成立，不把相关证据或计划文字自动当成题目义务。纠错不得顺带增加未获依据的主题、频率断言或任务要求。通用问题的scope_effect=correction同样不授权额外扩展义务。
