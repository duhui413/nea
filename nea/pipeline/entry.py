"""The single real experiment entry point."""
from __future__ import annotations
import argparse
import json
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable
from nea.providers.config import (experiment_role_models_for, local_runtime_identity,
                                  LOCAL_MODEL_DEVICE, LOCAL_MODEL_DTYPE, EXPERIMENT_GENERATORS)
from nea.pipeline.sampling import scope_question_set, validate_sample_size

def default_embedder(models_dir: Path, models=None) -> Any:
    """Use the same CUDA fp16 BGE-small for grounding and retrieval."""
    from nea.domain_models import HuggingFaceDomainModels

    class _Embedder:
        def __init__(self) -> None:
            self.models = models if models is not None else HuggingFaceDomainModels(models_dir)

        def embed(self, texts, *, device, dtype):
            if (device, dtype) != (LOCAL_MODEL_DEVICE, LOCAL_MODEL_DTYPE):
                raise ValueError("grounding embedder must run CUDA fp16")
            return self.models.embed(texts, purpose="grounding")

    return _Embedder()

def default_components(graph: Any) -> tuple[Callable[[str], float], Callable[[str], float], Any]:
    from nea.cognitive import question_structure_completeness, terminology_precision

    from nea.method.graph import link_concepts, to_knowledge_base
    from nea.method.signals import SignalComponents

    kb = to_knowledge_base(graph)
    return (lambda question: float(terminology_precision(question, kb, None)["value"]),
            lambda question: float(question_structure_completeness(question, None)["value"]),
            SignalComponents.default(lambda text: link_concepts(text, graph)))

def build_parser(repo_root: Path):
    parser = argparse.ArgumentParser(description="NEA 0022: live development; formal test -> analyze")
    parser.add_argument("--phase", choices=("live", "test", "analyze", "compare", "dev", "freeze"), required=True)
    parser.add_argument("--protocol", choices=("operator", "legacy"), default="operator",
                        help="operator: 0022 manifest pipeline; legacy: historical read-only analysis")
    parser.add_argument("--comparison-phase", choices=("live", "test"), default="live",
                        help="phase read by descriptive compare")
    parser.add_argument("--generator", default="mimo", help="mimo, minimax, or all (shared concurrent scheduler)")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume-from", help="import compatible completed requests from another live/test run")
    parser.add_argument("--reuse-generation-from", help="live only: reuse unchanged generation after judge/validate repair")
    parser.add_argument("--resume-endpoint-expansion-from", help="live only: preserve journals when adding configured credential endpoints")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="fixed first-three subset of this live/test phase")
    def sample_size(text):
        try:
            value = int(text)
            if not 1 <= value <= 270:
                raise ValueError("sample_size must be between 1 and 270")
            return value
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc
    mode.add_argument("--sample-size", type=sample_size, help="fixed first-N subset; live <=30, test <=270")
    mode.add_argument("--qids", nargs="+", help="operator live only: explicitly selected development qids")
    def minimax_attempts(text):
        try:
            value = int(text)
            if not 1 <= value <= 12:
                raise ValueError("minimax structure attempts must be between 1 and 12")
            return value
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc
    parser.add_argument("--minimax-structure-attempts", type=minimax_attempts,
                        help="operator live only: bounded MiniMax structure attempts; revise remains one")
    parser.add_argument("--output-root", type=Path, default=repo_root / "outputs")
    parser.add_argument("--question-set", type=Path, default=repo_root / "knowledge_base" / "questions.jsonl")
    parser.add_argument("--knowledge-root", type=Path, default=repo_root / "knowledge_base")
    parser.add_argument("--models-dir", type=Path, default=repo_root / "models")
    parser.add_argument("--prompts-dir", type=Path, default=repo_root / "prompts")
    parser.add_argument("--env-file", type=Path, default=repo_root / ".env")
    parser.add_argument("--mimo-batch", action="store_true",
                        help="MiMo uses its dedicated Batch key; other providers retain realtime concurrency")
    parser.add_argument("--rag-pool-path", type=Path, help="optional existing shared candidate pools; otherwise read Milvus")
    from nea.providers import config
    config.add_mimo_arguments(parser)
    return parser

def _rag_pools(args, graph, rows, phase_root, models=None):
    from nea.pipeline.workflow import read_json, write_once
    saved = phase_root / "rag_pools.json"
    if args.rag_pool_path or saved.exists():
        document = read_json(args.rag_pool_path or saved)
        if not isinstance(document, dict) or document.get("local_runtime") != local_runtime_identity():
            raise ValueError("RAG pool runtime differs or is missing; regenerate with CUDA fp16 in a new run")
        pools = document["pools"]
    else:
        from nea.domain_models import HuggingFaceDomainModels
        from nea.method.graph import to_knowledge_base
        from nea.storage.milvus import MilvusConceptStore
        models = models if models is not None else HuggingFaceDomainModels(args.models_dir)
        store = MilvusConceptStore(to_knowledge_base(graph), models)
        if store.healthcheck()["row_count"] != 6600:
            raise ValueError("Milvus must contain 6600 concepts")
        pools = {}
        for row in rows:
            hits = store.search(row["question"], top_k=12, query_id=row["qid"])
            scores = models.rerank(row["question"], [hit.chunk.content for hit in hits])
            pools[row["qid"]] = [{**hit.chunk.to_dict(), "dense_score": hit.dense_score, "rerank_score": score}
                                  for hit, score in zip(hits, scores)]
    # Never silently run RAG with absent or invented evidence.
    from nea.method.rag import select_shared_rag_context
    for row in rows:
        pool = pools.get(row["qid"], [])
        select_shared_rag_context(pool)
        for item in pool:
            node = graph.nodes.get(item["chunk_id"])
            if node is None or item["content"] != node["explain"]:
                raise ValueError(f"RAG evidence differs from concepts: {row['qid']}")
    write_once(saved, {"local_runtime": local_runtime_identity(),
                       "pools": {row["qid"]: pools[row["qid"]] for row in rows}})
    return pools


def repository_cli_lock(repo_root: Path):
    """One rendezvous point for the main checkout and its Git worktrees."""
    from nea.pipeline.artifacts import RunDirectoryLock
    git = repo_root / ".git"
    if git.is_file():
        text = git.read_text(encoding="utf8").strip()
        if not text.startswith("gitdir: "):
            raise ValueError("Malformed worktree gitdir")
        git = (repo_root / text[8:]).resolve()
        if (git / "commondir").is_file():
            git = (git / (git / "commondir").read_text(encoding="utf8").strip()).resolve()
    shared_root = git.parent if git.is_dir() else repo_root
    return RunDirectoryLock(shared_root / "outputs" / "provider-cli")


def _run_generation(args, repo_root: Path, run_root: Path, qset, generators):
    from nea.pipeline.artifacts import RunDirectoryLock
    from nea.pipeline.workflow import Environment, Phase, load_freeze, run_phases, write_once
    from nea.pipeline.planner import prepare_question
    from nea.pipeline.sampling import check_mode
    from nea.runtime.identity import phase_directory, validate_run_root
    from nea.method.graph import load_graph
    from nea.method.grounding import CandidateIndex
    from nea.providers.router import RoleRouter
    from nea.providers.pool import ProviderPool
    from nea.domain_models import HuggingFaceDomainModels

    with repository_cli_lock(repo_root), ExitStack() as locks:
        source_root = validate_run_root(repo_root, args.output_root, args.resume_from) if args.resume_from else None
        if source_root == run_root:
            raise ValueError("--resume-from must name a different run")
        phase_roots = {generator: phase_directory(run_root, generator, args.phase) for generator in generators}
        for path in phase_roots.values():
            locks.enter_context(RunDirectoryLock(path.with_name(path.name + "-runner")))
        frozen = load_freeze(run_root) if args.phase == "test" else None
        if frozen:
            check_mode(frozen["freeze"], smoke=args.smoke, sample_size=args.sample_size)
            if not (args.smoke or args.sample_size is not None) and not frozen["freeze"]["release_gate"]["release_test"]:
                raise ValueError("Development release criteria have not passed")
        env_file = args.env_file if args.env_file.is_absolute() else repo_root / args.env_file
        pool = ProviderPool(env_file)
        models = HuggingFaceDomainModels(args.models_dir)
        graph = load_graph(args.knowledge_root)
        for row in qset.rows:
            if {row["authoring_target_id"], *row["gold_target_ids"], *row["gold_labels"]} - graph.nodes.keys():
                raise ValueError(f"Unknown concept in {row['qid']}")
        rows = sorted(qset.split_rows(args.phase), key=lambda row: row["qid"])
        shared_root = run_root / "shared" / args.phase
        pool_args = argparse.Namespace(**vars(args))
        if not args.rag_pool_path and not (shared_root / "rag_pools.json").exists():
            prior_pools = [path / "rag_pools.json" for path in phase_roots.values() if (path / "rag_pools.json").exists()]
            if prior_pools:
                pool_args.rag_pool_path = prior_pools[0]
        pools = _rag_pools(pool_args, graph, rows, shared_root, models=models)
        for path in phase_roots.values():
            write_once(path / "rag_pools.json", {"local_runtime": local_runtime_identity(),
                       "pools": {row["qid"]: pools[row["qid"]] for row in rows}})
        terminology, structure, components = default_components(graph)
        index = CandidateIndex(graph, default_embedder(args.models_dir, models))
        prepared = {row["qid"]: prepare_question(row, graph, index, terminology, structure, pools[row["qid"]])
                    for row in rows}
        phases = []
        for generator in generators:
            env = Environment(run_root=run_root, run_id=args.run_id, generator=generator,
                question_set=qset, graph=graph, index=index, terminology=terminology, structure=structure,
                components=components, rag_pools=pools, prompts_dir=args.prompts_dir,
                router=RoleRouter(generator, env_file, pool=pool), smoke=args.smoke,
                sample_size=args.sample_size, prepared_questions=prepared)
            phases.append(Phase(env, args.phase, frozen))
        if source_root is not None:
            from nea.runtime.recovery import import_phase
            for phase in phases:
                receipt = import_phase(source_root, phase.run)
                print(json.dumps({"recovery": {"source_run_id": receipt["source_run_id"],
                    "target_run_id": receipt["target_run_id"], "generator": receipt["generator"],
                    "split": receipt["split"], "copied_files": len(receipt["copied_files"]),
                    "omitted_requests": len(receipt["omitted_requests"])}}, ensure_ascii=False, sort_keys=True), flush=True)
        results = run_phases(phases)
        return results[generators[0]] if len(generators) == 1 else {"generators": results}


def main(argv=None, repo_root: Path | None = None):
    repo_root = repo_root or Path(__file__).resolve().parents[2]
    args = build_parser(repo_root).parse_args(argv)
    from nea.question_set.contract import load_question_set
    from nea.runtime.identity import validate_run_root
    from nea.pipeline.artifacts import RunDirectoryLock
    from nea.pipeline.workflow import freeze, write_once, PhaseGroupError
    try:
        if (args.qids is not None or args.minimax_structure_attempts is not None) and (
                args.protocol != "operator" or args.phase != "live"):
            raise ValueError("--qids and --minimax-structure-attempts require operator live")
        generators = list(EXPERIMENT_GENERATORS) if args.generator == "all" else [experiment_role_models_for(args.generator)["student1"]]
        run_root = validate_run_root(repo_root, args.output_root, args.run_id)
        if args.protocol == "operator":
            from nea.pipeline.operator_entry import run_operator_cli
            result = run_operator_cli(args, repo_root, run_root, generators)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
            return 2 if result.get("complete") is False or result.get("status") == "NOT_READY_MISSING_RESULTS" else 0
        if args.reuse_generation_from or args.resume_endpoint_expansion_from:
            raise ValueError("--reuse-generation-from requires the operator protocol")
        if (run_root / "operator-v1").exists():
            raise ValueError("Operator run id cannot receive legacy artifacts")
        if args.phase in {"dev", "test", "freeze"}:
            raise RuntimeError("Historical generation requires its original code and prompt version; use operator for 0022")
        if args.phase == "live":
            raise ValueError("live belongs to the operator protocol")
        if args.resume_from and args.phase not in {"dev", "test"}:
            raise ValueError("--resume-from is only supported for dev/test")
        if args.resume_from == args.run_id:
            raise ValueError("--resume-from must name a different run")
        generators = list(EXPERIMENT_GENERATORS) if args.generator == "all" else [experiment_role_models_for(args.generator)["student1"]]
        run_root = validate_run_root(repo_root, args.output_root, args.run_id)
        if args.phase == "analyze" and (args.smoke or args.sample_size is not None
                or (run_root / "smoke_scope.json").exists() or (run_root / "sample_scope.json").exists()):
            raise ValueError("Smoke and exploratory runs do not support formal statistical analysis")
        if args.phase == "compare":
            if args.smoke:
                raise ValueError("Comparison requires an exploratory sample")
            from nea.evaluation.comparison import compare_run
            result = compare_run(run_root)
            if args.sample_size is not None and result["sample_size"] != args.sample_size:
                raise ValueError("Requested sample_size differs from comparison scope")
            with RunDirectoryLock(run_root):
                write_once(run_root / "exploratory_comparison.json", result)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
            return 0
        qset = load_question_set(args.question_set)
        if qset.sealed_problems() or qset.counts != {"dev": 60, "test": 240}:
            raise ValueError("Historical analysis requires its original sealed dev60/test240 question set")
        with RunDirectoryLock(run_root):
            qset = scope_question_set(run_root, qset, smoke=args.smoke, sample_size=args.sample_size)
        if args.phase in {"dev", "test"}:
            result = _run_generation(args, repo_root, run_root, qset, generators)
        elif args.phase == "freeze":
            result = freeze(run_root, qset, smoke=args.smoke, sample_size=args.sample_size)
        else:
            from nea.evaluation.analysis import analyze_run
            result = analyze_run(run_root, qset)
            with RunDirectoryLock(run_root):
                write_once(run_root / "analysis.json", result)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        if isinstance(exc, PhaseGroupError):
            print(json.dumps({"failed_groups": exc.failures, "completed_groups": sorted(exc.results)},
                             ensure_ascii=False, sort_keys=True), file=sys.stderr)
        elif getattr(exc, "retry_at", None) is not None:
            print(f"RETRY_AT_UNIX_UTC: {exc.retry_at}; CODE={getattr(exc, 'code', 'transport_error')}", file=sys.stderr)
        from nea.providers.client import LLMTransportError
        cause = exc.__cause__
        if isinstance(cause, LLMTransportError):
            print(f"PROVIDER: {cause.code}; HTTP={cause.details.get('http_status', '-')}", file=sys.stderr)
        elif cause is not None:
            print(f"CAUSE: {type(cause).__name__}", file=sys.stderr)
        return 2
