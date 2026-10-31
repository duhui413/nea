# NEA experimental source

Frozen source supporting the paper. Statistical inference resamples whole questions (`qid`) with replacement, 20,000 replicates and seed 20260927. Each question first averages its two judges and the four generator/context cells equally; methods stay paired within the question. Length sensitivity resamples the complete observation block for each question. Quality evaluation uses the six-dimensional rubric; preference-pair generation and analysis have been removed. Includes the six frozen knowledge-base JSONL inputs. No manuscript, experiment results, model weights, virtual environment, preprocessing/annotation pipeline, or one-off restart supervisor is included.

- `nea/method/`: retrieval, grounding, planning, diagnosis, revision and comparison methods.
- `nea/pipeline/formal_*.py`: formal generation, evaluation and perturbation workflow.
- `nea/evaluation/`: scoring and statistical analysis implementation.
- `nea/providers/config.py`: sole definition of models, endpoints, sampling and local precision.
- `prompts/`: method and evaluation templates and schema-related configuration.
- `scripts/run_formal_experiment.py`: formal experiment entry point.
- `scripts/run_diagnosis_perturbation.py`: perturbation entry point.
- `scripts/run_gplan_cove_supplement.py`, `scripts/run_plan_supplement.py`: supplemental comparison implementations. These preserve their original parent-record paths and checks; they are not standalone fresh-run entry points.
- `scripts/database/`: Milvus and Neo4j builders from already structured inputs.

Internal runtime modules needed by these sources are retained, including normal request accounting and restart safety. Run from this checkout so the prompts remain available alongside the Python package.

## Environment

Frozen environment: Windows, Python 3.13.9, CUDA 13.0 PyTorch (`torch==2.13.0+cu130`), CUDA GPU; local inference uses FP16 and seed 23. Install a matching CUDA-enabled PyTorch wheel from the PyTorch CUDA 13.0 distribution or your original wheel cache. Other runtime packages and the Windows transitive closure are pinned in the dependency files. No installed environment or binaries are bundled.

```powershell
py -3.13 -m venv .venv
# After making the matching CUDA torch wheel available:
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pip install --no-deps -e .
.venv/Scripts/python.exe -B scripts/run_formal_experiment.py --phase plan --run-id paper
```

The plan command is offline. The six frozen inputs are included under `knowledge_base/`: `concepts.jsonl`, `edges.jsonl`, `graph.jsonl`, `questions.jsonl`, `questions_meta.jsonl`, `questions_case.jsonl`. They are copied unchanged from the original project and are not regenerated or annotated by this source snapshot. Place the complete pretrained model folders in `models/BAAI/bge-small-zh-v1.5/` and `models/BAAI/bge-reranker-base/`. The hierarchy comes from `graph.jsonl`.

Supply `MIMO_API_KEY` and `MINIMAX_API_KEY` through process environment or a local ignored `.env` (empty template: `.env.example`). Model and endpoint settings are in `nea/providers/config.py`. Real experiment commands call the configured model APIs; default request parameters include seed 23, temperature 0.2, at most three structural attempts and bounded transport retries. Use the same run-id and options when resuming; outputs remain outside Git.

```powershell
.venv/Scripts/python.exe -B scripts/run_formal_experiment.py --phase test --run-id paper
```

## Database builders

Milvus server: 2.6.6, `http://localhost:19530`, database `nea`, collection `concepts`, 512-dimensional `FLOAT16_VECTOR`, HNSW/COSINE. Build from existing `concepts.jsonl` and `graph.jsonl` using the local CUDA embedding model:

```powershell
.venv/Scripts/python.exe -B scripts/database/build_milvus_concepts.py --upsert --create-database
```

An existing collection is replaced only with the additional explicit `--drop` flag. Experiment runtime uses read-only queries.

The optional Neo4j maintenance copy uses Enterprise 2025.12.1 with a separate `nea` database, and reads `concepts.jsonl`, `edges.jsonl`, `graph.jsonl`. Set `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` locally and run:

```powershell
.venv/Scripts/python.exe -B scripts/database/build_neo4j_graph.py --upsert --create-database
```

Neo4j is not used by the experiment runtime. Neither builder annotates electronic documents or produces chunks/edges from raw materials.

Supplementary analyses use the same question-level bootstrap. Their archived reference check compares the point effect only: intervals and p values from the previous target-cluster protocol are not comparable to the new uncertainty estimates. Current reports use `need_metrics` and do not require preference artifacts. Existing run manifests remain immutable; this version has a new source/prompt identity.
