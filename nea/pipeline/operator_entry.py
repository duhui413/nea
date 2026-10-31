"""CLI dispatch for the manifest-bound 0022 live/test protocol."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from nea.pipeline.artifacts import RunDirectoryLock
from nea.runtime.records import ConflictError, digest, write_once


def check_namespace(run_root: Path) -> None:
    """A legacy run id can never become a new-protocol run id."""
    if any((run_root / name).exists() for name in ("freeze.json", "smoke_scope.json", "sample_scope.json")):
        raise ConflictError("Legacy run id cannot receive operator artifacts; choose a new run id")
    for path in run_root.glob("*/dev/run_manifest.json"):
        raise ConflictError("Legacy run id cannot receive operator artifacts; choose a new run id")
    for path in run_root.glob("*/test/run_manifest.json"):
        raise ConflictError("Legacy run id cannot receive operator artifacts; choose a new run id")


def asset_identity(knowledge_root: Path, question_set: Path) -> dict[str, str]:
    """Hash the actual loaded question sources alongside the knowledge assets."""
    sources = {name: knowledge_root / (name + ".jsonl")
               for name in ("concepts", "edges", "graph", "questions_case")}
    sources.update(questions=question_set, questions_meta=question_set.with_name("questions_meta.jsonl"))
    return {name: digest(path.read_bytes()) for name, path in sources.items()}


def prepare_retrieval_cache(args: Any, graph: Any, rows: list[dict], shared_root: Path,
                            assets: dict[str, str], *, models: Any = None) -> dict:
    """Reuse BGE's implementation with an explicit operator cache identity."""
    import json
    from nea.pipeline.entry import _rag_pools
    from nea.providers.config import local_runtime_identity
    identity = {"protocol": "nea-0022-v1", "phase": args.phase,
                "qids": [row["qid"] for row in rows],
                "questions_hash": digest([{k: row[k] for k in ("qid", "split", "question")} for row in rows]),
                "assets": assets, "local_runtime": local_runtime_identity(),
                "evidence_policy": "shared-top8-no-extra-body"}
    cache = shared_root / "rag_pools.json"
    sidecar = shared_root / "retrieval_manifest.json"
    if args.rag_pool_path:
        source = args.rag_pool_path.parent / "retrieval_manifest.json"
        if not source.is_file() or json.loads(source.read_text(encoding="utf8")) != identity:
            raise ConflictError("RAG cache protocol/phase/data identity differs or is missing")
    if cache.exists() and not sidecar.is_file():
        raise ConflictError("An unversioned RAG cache cannot enter the operator protocol")
    with RunDirectoryLock(shared_root):
        write_once(sidecar, identity)
    return _rag_pools(args, graph, rows, shared_root, models=models)


def run_operator_cli(args: Any, repo_root: Path, run_root: Path, generators: list[str]) -> dict[str, Any]:
    from nea.evaluation.operator_analysis import analyze_operator_run, compare_operator_run
    from nea.question_set.contract import load_question_set

    if args.phase not in {"live", "test", "analyze", "compare"}:
        raise ValueError("0022 uses live for development and test -> analyze for formal runs")
    from nea.providers import config
    config.resolve_mimo_api(getattr(args, "mimo_api", None), mimo_batch=getattr(args, "mimo_batch", False))
    config.batch_size(getattr(args, "mimo_batch_size", 5))
    config.mimo_concurrency(getattr(args, "mimo_concurrency", None))
    config.minimax_concurrency(getattr(args, "minimax_concurrency", None))
    check_namespace(run_root)
    reuse_source = getattr(args, "reuse_generation_from", None)
    endpoint_source = getattr(args, "resume_endpoint_expansion_from", None)
    selected_qids = getattr(args, "qids", None)
    minimax_attempts = getattr(args, "minimax_structure_attempts", None)
    if (selected_qids is not None or minimax_attempts is not None) and args.phase != "live":
        raise ValueError("explicit qids and provider structure attempts require live")
    if selected_qids is not None and (args.smoke or args.sample_size is not None):
        raise ValueError("explicit qids cannot combine with smoke or sample_size")
    if minimax_attempts is not None and (type(minimax_attempts) is not int or not 1 <= minimax_attempts <= 12):
        raise ValueError("MiniMax structure attempts must be between 1 and 12")
    if minimax_attempts is not None and any((reuse_source, endpoint_source, args.resume_from)):
        raise ValueError("custom structure budgets require same-directory resume; cross-directory recovery is disabled")
    if sum(bool(x) for x in (reuse_source, endpoint_source, args.resume_from)) > 1:
        raise ValueError("Choose exactly one recovery mode")
    if (reuse_source or endpoint_source) and (args.phase != "live" or args.resume_from):
        raise ValueError("--reuse-generation-from requires live and cannot combine with --resume-from")
    if args.phase in {"analyze", "compare"}:
        if args.resume_from:
            raise ValueError("--resume-from is only available for live/test generation")
        if args.phase == "analyze":
            if args.smoke or args.sample_size is not None:
                raise ValueError("analyze reads the locked formal run; use compare for a sample")
            result = analyze_operator_run(run_root)
            path = run_root / "operator-v1" / "analysis.json"
        else:
            result = compare_operator_run(run_root, phase=getattr(args, "comparison_phase", "live"))
            path = run_root / "operator-v1" / ("comparison-" + getattr(args, "comparison_phase", "live") + ".json")
        with RunDirectoryLock(run_root):
            write_once(path, result)
        return result
    qset = load_question_set(args.question_set)
    if qset.sealed_problems() or qset.counts != {"dev": 30, "test": 270}:
        raise ValueError("0022 requires the accepted sealed 30/270 question set")
    if (qset.manifest or {}).get("version") != "nea-questions-0022-v1":
        raise ValueError("0022 question-set version is missing or differs")
    size = 3 if args.smoke else args.sample_size
    split = "dev" if args.phase == "live" else "test"
    if size is not None and not 1 <= size <= qset.counts[split]:
        raise ValueError("sample_size exceeds the phase's question count")
    from nea.pipeline.operator_workflow import select_operator_rows
    select_operator_rows(qset.split_rows(split), args.phase, size, selected_qids)
    return _generate(args, repo_root, run_root, qset, generators, size)


def _generate(args: Any, repo_root: Path, run_root: Path, qset: Any,
              generators: list[str], sample_size: int | None) -> dict[str, Any]:
    from nea.pipeline.entry import default_embedder, repository_cli_lock
    from nea.pipeline.operator_workflow import OperatorEnvironment, OperatorPhase, select_operator_rows
    from nea.method.graph import load_graph
    from nea.method.grounding import CandidateIndex
    from nea.domain_models import HuggingFaceDomainModels
    from nea.providers.pool import ProviderPool
    from nea.providers.router import RoleRouter
    from nea.runtime.identity import operator_phase_directory, validate_run_root
    from nea.runtime.recovery import import_phase

    with repository_cli_lock(repo_root), ExitStack() as locks:
        reuse_name = getattr(args, "reuse_generation_from", None) or getattr(args, "resume_endpoint_expansion_from", None)
        reuse_source = validate_run_root(repo_root, args.output_root, reuse_name) if reuse_name else None
        if reuse_source is not None:
            check_namespace(reuse_source)
        source = validate_run_root(repo_root, args.output_root, args.resume_from) if args.resume_from else None
        if source == run_root:
            raise ValueError("--resume-from must name a different run")
        if source is not None:
            check_namespace(source)
        for generator in generators:
            phase_root = operator_phase_directory(run_root, generator, args.phase)
            locks.enter_context(RunDirectoryLock(phase_root.with_name(phase_root.name + "-runner")))
        env_file = args.env_file if args.env_file.is_absolute() else repo_root / args.env_file
        pool = ProviderPool(env_file, mimo_batch=getattr(args, "mimo_batch", False),
            mimo_api=getattr(args, "mimo_api", None), mimo_batch_size=getattr(args, "mimo_batch_size", 5),
            mimo_concurrency=getattr(args, "mimo_concurrency", None),
            minimax_concurrency=getattr(args, "minimax_concurrency", None))
        models = HuggingFaceDomainModels(args.models_dir)
        graph = load_graph(args.knowledge_root)
        split = "dev" if args.phase == "live" else "test"
        selected_qids = getattr(args, "qids", None)
        rows = select_operator_rows(qset.split_rows(split), args.phase, sample_size, selected_qids)
        shared_root = run_root / "operator-v1" / "shared" / args.phase
        assets = asset_identity(args.knowledge_root, args.question_set)
        # Retrieval and BGE are shared between generator groups, as in 0015.
        pools = prepare_retrieval_cache(argparse.Namespace(**vars(args)), graph, rows, shared_root, assets, models=models)
        index = CandidateIndex(graph, default_embedder(args.models_dir, models))
        candidates = {row["qid"]: index.rank(row["question"]) for row in rows}
        phases = [OperatorPhase(OperatorEnvironment(run_root=run_root, run_id=args.run_id,
            generator=generator, question_set=qset, graph=graph, candidates=candidates, rag_pools=pools,
            prompts_dir=args.prompts_dir, router=RoleRouter(generator, env_file, pool=pool),
            phase=args.phase, sample_size=sample_size, asset_identity=assets,
            selected_qids=selected_qids,
            structure_attempts_by_provider=(
                {"mimo": 3, "minimax": args.minimax_structure_attempts}
                if getattr(args, "minimax_structure_attempts", None) is not None else None)))
            for generator in generators]
        if reuse_source is not None:
            # Bind provenance before publishing manifests or sending any request.
            from nea.runtime.generation_recovery import GenerationRecovery, STAGES
            for phase in phases:
                phase.run.identity["generation_reuse"] = {"source_run_id": reuse_source.name,
                    "version": 2, "mode": "endpoint_expansion" if getattr(args, "resume_endpoint_expansion_from", None) else "generation_only",
                    "stages": list(STAGES) + (["validate", "judge"] if getattr(args, "resume_endpoint_expansion_from", None) else [])}
                phase.generation_recovery = GenerationRecovery(reuse_source, phase.run,
                    endpoint_expansion=bool(getattr(args, "resume_endpoint_expansion_from", None)))
        # Every phase manifest is fixed before the first model request.
        for phase in phases:
            with RunDirectoryLock(phase.run.root):
                phase.run.initialize()
                if source is not None:
                    import_phase(source, phase.run)
        results = {}
        with ThreadPoolExecutor(max_workers=len(phases), thread_name_prefix="nea-0022") as executor:
            futures = {executor.submit(phase.run_phase): phase.env.generator for phase in phases}
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        return {"protocol": "nea-0022-v1", "phase": args.phase,
                "complete": all(result["complete"] for result in results.values()),
                "generators": dict(sorted(results.items()))}
