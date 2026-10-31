"""One batch directory for the complete formal protocol and every restart."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
from pathlib import Path

from nea.pipeline.artifacts import RunDirectoryLock
from nea.providers import config
from nea.runtime.formal_protocol import PROTOCOL, DIRECTORY, phase_directory, aggregate_telemetry
from nea.runtime.identity import validate_run_root
from nea.runtime.records import ConflictError, digest, write_once


def build_parser(repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Eight-arm experiment; restart with the same run-id")
    parser.add_argument("--phase", choices=("plan", "live", "test", "case", "perturb", "analyze"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--generator", default="all")
    parser.add_argument("--comparison-phase", choices=("live", "test"), default="test")
    parser.add_argument("--sample-size", type=int, help="fixed first-N sample; never a formal result")
    parser.add_argument("--output-root", type=Path, default=repo_root / "outputs")
    parser.add_argument("--question-set", type=Path, default=repo_root / "knowledge_base/questions.jsonl")
    parser.add_argument("--knowledge-root", type=Path, default=repo_root / "knowledge_base")
    parser.add_argument("--models-dir", type=Path, default=repo_root / "models")
    parser.add_argument("--prompts-dir", type=Path, default=repo_root / "prompts")
    parser.add_argument("--env-file", type=Path, default=repo_root / ".env")
    parser.add_argument("--mimo-batch", action="store_true",
                        help="MiMo uses its dedicated Batch key; MiniMax stays realtime; resume with the same flag")
    parser.add_argument("--rag-pool-path", type=Path)
    from nea.providers import config
    config.add_mimo_arguments(parser)
    return parser


def execution_plan(*, mimo_batch: bool = False, mimo_api: str | None = None, mimo_batch_size: int = 5, mimo_concurrency: int | None = None, minimax_concurrency: int | None = None,
                   minimax_m3_api: str | None = None, minimax_relay_concurrency: int | None = None,
                   mimo_official_concurrency: int | None = None) -> dict:
    mode = config.resolve_mimo_api(mimo_api, mimo_batch=mimo_batch)
    mimo_batch = mode == "batch"
    size = config.batch_size(mimo_batch_size)
    concurrency = config.mimo_concurrency(mimo_concurrency, mode)
    mini = config.minimax_concurrency(minimax_concurrency) or 2
    result = {"mimo_api": mode, "minimax_max_in_flight": mini, "mimo_rate_policy": dict(config.MIMO_RATE_POLICY) if mode == "standard" else None, "protocol": PROTOCOL, "status": "NOT_RUN", "calls_made": 0,
        "models": {generator: config.experiment_role_models_for(generator) for generator in config.EXPERIMENT_GENERATORS},
        "endpoints": dict(config.EXPERIMENT_PROVIDER_ENDPOINTS),
        "request_options": dict(config.REQUEST_OPTIONS), "max_in_flight": config.EXPERIMENT_MAX_IN_FLIGHT,
        "structure_attempts": 3, "revision_structure_attempts": 1,
        "transport_max_failures": config.TRANSPORT_MAX_FAILURES,
        "formal_answer_positions": 8640, "formal_judgment_positions": 17280,
        "estimated_call_upper_bound_without_retries": 115300,
        "price_status": "NOT_CONFIGURED", "resume": "same run-id, immutable successes, cumulative attempts",
        "case_policy": "independent Q410/Q240/Q213, excluded from aggregate statistics"}
    if mode in ("standard", "ezlook"):
        result.update(mimo_concurrency=concurrency, max_in_flight=concurrency + mini)
    if mimo_batch:
        result.update(mimo_transport="batch", mimo_batch_endpoint=config.MIMO_BATCH_ENDPOINT,
            endpoints={**config.EXPERIMENT_PROVIDER_ENDPOINTS, "mimo": config.MIMO_BATCH_ENDPOINT},
            mimo_batch_policy={**config.MIMO_BATCH_POLICY, "max_requests": size},
            realtime_providers=[p for p in config.EXPERIMENT_PROVIDER_ORDER if p != "mimo"],
            mimo_batch_credential=config.MIMO_BATCH_CREDENTIAL_NAME,
            realtime_max_in_flight=mini,
            max_in_flight=mini,
            resume="same run-id and --mimo-batch; resume saved server tasks before retrying failed rows")
    if mode == "token-plan":
        result.update(endpoints={**config.EXPERIMENT_PROVIDER_ENDPOINTS, "mimo": config.MIMO_TOKEN_PLAN_ENDPOINT},
                      max_in_flight=1 + mini, mimo_credential=config.MIMO_TOKEN_PLAN_CREDENTIAL_NAME)
    official = config.mimo_official_concurrency(mimo_official_concurrency)
    if official:
        result.update(mimo_official_concurrency=official, mimo_official_endpoint=config.MIMO_STANDARD_ENDPOINT,
                      mimo_official_rate_policy=dict(config.MIMO_RATE_POLICY))
        if isinstance(result.get("max_in_flight"), int) and isinstance(result.get("mimo_concurrency"), int):
            result["max_in_flight"] += official
    if config.resolve_minimax_m3_api(minimax_m3_api) in config.MINIMAX_RELAY_MODE_MODELS:
        relay = config.minimax_relay_concurrency(minimax_relay_concurrency)
        relay_mode = config.resolve_minimax_m3_api(minimax_m3_api)
        result.update(minimax_m3_api=relay_mode, minimax_relay_models=list(config.MINIMAX_RELAY_MODE_MODELS[relay_mode]),
                      minimax_relay_endpoint=config.MINIMAX_RELAY_ENDPOINT, minimax_relay_concurrency=relay,
                      minimax_relay_credential=config.MINIMAX_RELAY_CREDENTIAL_NAME)
        if "max_in_flight" in result and isinstance(result.get("mimo_concurrency"), int):
            result["max_in_flight"] = result["mimo_concurrency"] + mini + relay
    if mode == "ezlook":
        result.update(endpoints={**config.EXPERIMENT_PROVIDER_ENDPOINTS, "mimo": config.EZLOOK_ENDPOINT},
                      mimo_credential=config.EZLOOK_CREDENTIAL_NAME)
    return result


def _read_trusted_questions(path: Path, *, cases: bool = False):
    """Use accepted assets directly; do not audit labels or revalidate their truth."""
    from nea.question_set.contract import QuestionSet
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
    if cases:
        rows = [{"qid": str(row.get("qid", row.get("id"))), "split": "case",
                 "question": row.get("question", row.get("query")), "style": "case",
                 "authoring_target_id": row.get("authoring_target_id",
                     row.get("target_concept", {}).get("id", "")
                     if isinstance(row.get("target_concept"), dict) else row.get("target_concept", "")),
                 "gold_target_ids": row.get("gold_target_ids", []), "gold_labels": row.get("gold_labels", {}),
                 "status": "sealed", "case_source": row} for row in rows]
        manifest = {"version": "trusted-independent-case-v1", "sha256": digest(raw)}
    else:
        manifest = json.loads(path.with_name("questions_meta.jsonl").read_text(encoding="utf-8-sig"))
    return QuestionSet(tuple(sorted(rows, key=lambda row: row["qid"])), digest(raw), manifest)


def prepare_retrieval(args, graph, rows, shared_root, assets, models):
    from nea.pipeline.entry import _rag_pools
    identity = {"protocol": PROTOCOL, "phase": args.phase, "qids": [row["qid"] for row in rows],
        "questions_hash": digest([{key: row[key] for key in ("qid", "split", "question")} for row in rows]),
        "assets": assets, "local_runtime": config.local_runtime_identity(), "evidence_policy": "shared-top8-no-extra-body"}
    sidecar = shared_root / "retrieval_manifest.json"
    if args.rag_pool_path:
        source = args.rag_pool_path.parent / "retrieval_manifest.json"
        if not source.is_file() or json.loads(source.read_text(encoding="utf8")) != identity:
            raise ConflictError("shared retrieval cache differs from the formal scope")
    if (shared_root / "rag_pools.json").exists() and not sidecar.is_file():
        raise ConflictError("retrieval cache has no formal identity")
    with RunDirectoryLock(shared_root):
        write_once(sidecar, identity)
    return _rag_pools(args, graph, rows, shared_root, models=models)


def build_phases(args, repo_root, run_root, generators, *, pool, graph, models):
    from nea.pipeline.entry import default_embedder
    from nea.pipeline.formal_workflow import FormalEnvironment, FormalPhase
    from nea.method.grounding import CandidateIndex
    from nea.providers.router import RoleRouter
    from nea.pipeline.operator_entry import asset_identity
    qset = _read_trusted_questions(args.knowledge_root / "questions_case.jsonl" if args.phase == "case"
                                   else args.question_set, cases=args.phase == "case")
    split = "dev" if args.phase == "live" else args.phase
    rows = sorted(qset.split_rows(split), key=lambda row: row["qid"])
    if args.sample_size is not None:
        if args.phase == "case" or not 1 <= args.sample_size <= len(rows):
            raise ValueError("sample-size must fit live/test and cannot select independent cases")
        rows = rows[:args.sample_size]
    # Hashes bind runtime inputs, without evaluating or changing trusted assets.
    assets = asset_identity(args.knowledge_root, args.question_set)
    pools = prepare_retrieval(args, graph, rows, run_root / DIRECTORY / "shared" / args.phase, assets, models)
    index = CandidateIndex(graph, default_embedder(args.models_dir, models))
    candidates = {row["qid"]: index.rank(row["question"]) for row in rows}
    phases = [FormalPhase(FormalEnvironment(run_root=run_root, run_id=args.run_id,
        generator=generator, question_set=qset, graph=graph, candidates=candidates, rag_pools=pools,
        prompts_dir=args.prompts_dir, router=RoleRouter(generator, args.env_file, pool=pool),
        phase=args.phase, sample_size=args.sample_size, asset_identity=assets)) for generator in generators]
    for phase in phases:
        from nea.runtime.formal_recovery import bind_imported_case
        bind_imported_case(phase)
        with RunDirectoryLock(phase.run.root):
            phase.run.initialize()
    return phases


def run_phase_code(phase) -> dict:
    """Shared real/fixture path; callers supply either a real or a synthetic router."""
    return evaluate_phase_code(phase, phase.run_phase())


def evaluate_phase_code(phase, records) -> dict:
    """Evaluation half of ``run_phase_code``, given the phase's completed generation records."""
    from nea.pipeline.formal_evaluation import run_evaluations
    from nea.runtime.formal_recovery import enabled
    if enabled(phase.run) and not records.get("summary", {}).get("generation_complete"):
        return {"records": records, "evaluation": {"summary": {"status": "NOT_RUN",
                "reason": "generation_recovery_pending"}}}
    evaluation = run_evaluations(phase, records)
    return {"records": records, "evaluation": evaluation}


def publish_phase(phase, result: dict, *, perturbations=None, cases=None) -> dict:
    records, evaluation = result["records"], result["evaluation"]
    generation = records.get("summary", {})
    eval_summary = evaluation.get("summary", evaluation)
    generated = bool(generation.get("generation_complete"))
    evaluated = eval_summary.get("status") == "COMPLETE"
    required = {"E1": generated and evaluated,
                "E2": evaluated, "E3": generated,
                "E5": evaluated, "E7": evaluated,
                "E8": cases is not None and all(item.get("complete") for item in cases)}
    if phase.env.phase == "live":
        required["E6"] = generated
        required.pop("E8")
    elif phase.env.phase == "test":
        required["E4"] = bool(perturbations and perturbations.get("status") == "COMPLETE")
    else:
        required = {"E8": generated and evaluated}
    summary = {**generation, "protocol": PROTOCOL, "origin": phase.env.origin,
        "phase": phase.env.phase, "generator": phase.run.identity["generator"],
        "complete": all(required.values()) and not phase.stop_reason,
        "required_experiments": required, "evaluation": eval_summary,
        "stop_reason": phase.stop_reason, "telemetry": aggregate_telemetry(phase.run.root)}
    summary["status"] = "COMPLETE" if summary["complete"] else "NOT_RUN"
    with RunDirectoryLock(phase.run.root):
        if cases is not None:
            case_record = {"protocol": PROTOCOL, "origin": phase.env.origin,
                "status": "COMPLETE" if required.get("E8") else "NOT_RUN",
                "rows": [row for group in cases if group["generator"] == phase.run.identity["generator"]
                         for row in group["answers"]]}
            write_once(phase.run.root / "cases" / (digest(case_record) + ".json"), case_record)
            if required.get("E8"):
                write_once(phase.run.root / "cases.json", case_record)
        write_once(phase.run.root / "snapshots" / (digest(summary) + ".json"), summary)
        if summary["complete"]:
            write_once(phase.run.root / "phase_summary.json", summary)
    return summary


def run_cli(args, repo_root: Path) -> dict:
    config.resolve_mimo_api(getattr(args, "mimo_api", None), mimo_batch=getattr(args, "mimo_batch", False))
    config.batch_size(getattr(args, "mimo_batch_size", 5))
    config.mimo_concurrency(getattr(args, "mimo_concurrency", None), config.resolve_mimo_api(
        getattr(args, "mimo_api", None), mimo_batch=getattr(args, "mimo_batch", False)))
    config.minimax_concurrency(getattr(args, "minimax_concurrency", None))
    config.resolve_minimax_m3_api(getattr(args, "minimax_m3_api", None))
    config.mimo_official_concurrency(getattr(args, "mimo_official_concurrency", None))
    config.minimax_relay_concurrency(getattr(args, "minimax_relay_concurrency", None))
    run_root = validate_run_root(repo_root, args.output_root, args.run_id)
    if args.phase == "plan":
        return execution_plan(mimo_batch=getattr(args, "mimo_batch", False),
            mimo_api=getattr(args, "mimo_api", None), mimo_batch_size=getattr(args, "mimo_batch_size", 5),
            mimo_concurrency=getattr(args, "mimo_concurrency", None),
            minimax_concurrency=getattr(args, "minimax_concurrency", None),
            minimax_m3_api=getattr(args, "minimax_m3_api", None),
            minimax_relay_concurrency=getattr(args, "minimax_relay_concurrency", None),
            mimo_official_concurrency=getattr(args, "mimo_official_concurrency", None))
    if args.phase == "analyze":
        from nea.evaluation.formal_analysis import analyze_formal_run, export_paper_tables
        report = analyze_formal_run(run_root, phase=args.comparison_phase)
        exported = export_paper_tables(report, run_root)
        return {"status": report.get("status"), "paper_tables": str(exported), "analysis": report}
    generators = list(config.EXPERIMENT_GENERATORS) if args.generator == "all" else [config.experiment_role_models_for(args.generator)["student1"]]
    if args.sample_size is not None and args.sample_size < 1:
        raise ValueError("sample-size must be positive")
    if args.phase == "perturb":
        # Reconstruct locked parents with the same manifest; no generation calls.
        args = argparse.Namespace(**{**vars(args), "phase": "test"})
        perturb_only = True
    else:
        perturb_only = False
    from nea.pipeline.entry import repository_cli_lock
    from nea.method.graph import load_graph
    from nea.domain_models import HuggingFaceDomainModels
    from nea.providers.pool import ProviderPool
    from nea.pipeline.formal_perturbation import run_perturbations
    with repository_cli_lock(repo_root), ExitStack() as locks:
        phases_to_lock = [args.phase] + (["case"] if args.phase == "test" and not perturb_only else [])
        for name in phases_to_lock:
            for generator in generators:
                path = phase_directory(run_root, generator, name)
                locks.enter_context(RunDirectoryLock(path.with_name(path.name + "-runner")))
        pool = ProviderPool(args.env_file, mimo_batch=getattr(args, "mimo_batch", False),
            mimo_api=getattr(args, "mimo_api", None), mimo_batch_size=getattr(args, "mimo_batch_size", 5),
            mimo_concurrency=getattr(args, "mimo_concurrency", None),
            minimax_concurrency=getattr(args, "minimax_concurrency", None),
            minimax_m3_api=getattr(args, "minimax_m3_api", None),
            minimax_relay_concurrency=getattr(args, "minimax_relay_concurrency", None),
            mimo_official_concurrency=getattr(args, "mimo_official_concurrency", None))
        models = HuggingFaceDomainModels(args.models_dir)
        graph = load_graph(args.knowledge_root)
        phases = build_phases(args, repo_root, run_root, generators, pool=pool, graph=graph, models=models)
        if perturb_only:
            return run_perturbations(phases)
        perturbations, cases = None, None
        with ThreadPoolExecutor(max_workers=len(phases), thread_name_prefix="formal8") as executor:
            generated = list(executor.map(lambda phase: phase.run_phase(), phases))
            all_generated = all(item.get("summary", {}).get("generation_complete") for item in generated)
            # Controlled perturbations need only the generated answers, so they run beside evaluation
            # (over disjoint request keys) instead of waiting for it.
            if args.phase == "test" and all_generated:
                with ThreadPoolExecutor(max_workers=1, thread_name_prefix="formal8-perturb") as side:
                    pending = side.submit(run_perturbations, phases)
                    evaluations = [executor.submit(evaluate_phase_code, phase, records)
                                   for phase, records in zip(phases, generated)]
                    results = [future.result() for future in evaluations]
                    perturbations = pending.result()
            else:
                results = [future.result() for future in [executor.submit(evaluate_phase_code, phase, records)
                                                         for phase, records in zip(phases, generated)]]
        if perturbations is not None:
            case_args = argparse.Namespace(**{**vars(args), "phase": "case", "sample_size": None, "rag_pool_path": None})
            case_phases = build_phases(case_args, repo_root, run_root, generators, pool=pool, graph=graph, models=models)
            with ThreadPoolExecutor(max_workers=len(case_phases), thread_name_prefix="formal8-case") as executor:
                case_results = list(executor.map(run_phase_code, case_phases))
            cases = []
            for phase, item in zip(case_phases, case_results):
                case_summary = publish_phase(phase, item)
                cases.append({"generator": phase.run.identity["generator"], "complete": case_summary["complete"],
                              "answers": item["records"]["answers"]})
        summaries = {phase.env.generator: publish_phase(phase, item, perturbations=perturbations, cases=cases)
                     for phase, item in zip(phases, results)}
        return {"protocol": PROTOCOL, "phase": args.phase,
                "complete": all(row["complete"] for row in summaries.values()), "generators": summaries}


def main(argv=None, repo_root: Path | None = None) -> int:
    repo_root = repo_root or Path(__file__).resolve().parents[2]
    args = build_parser(repo_root).parse_args(argv)
    try:
        result = run_cli(args, repo_root)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 2 if result.get("complete") is False or result.get("status") == "NOT_RUN" and args.phase != "plan" else 0
    except (ValueError, ConflictError, OSError) as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc)}, ensure_ascii=False))
        return 2
