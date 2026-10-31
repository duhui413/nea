# Frozen question-bootstrap experiment source

Source: `feat/question-bootstrap-score-only` commit `f07355e6e83745c35d29b7419ff7fd18cc6020bd`, extracted on 2026-10-31.
Branch: `backup/20261031-question-bootstrap-source`. Independent root history; never merge into master.

Retrieve with `git clone --single-branch --branch backup/20261031-question-bootstrap-source https://github.com/duhui413/nea.git`.

The source and prompts are copied byte for byte from the verified implementation. Inference samples whole questions with replacement. Six-dimensional scoring remains; preference-pair generation and analysis are removed. The six frozen knowledge_base JSONL inputs (concepts, edges, graph, questions, questions_meta, questions_case) are copied unchanged in a separate data commit.

The source contains the formal experiment, perturbation study, supplemental methods, their internal import dependencies, and two database builders. The supplement entry points retain their original external parent-answer paths and identity guards. Model weights, installed environments, manuscript files, experimental outputs, raw-material annotation and one-off restart supervisors are excluded.
