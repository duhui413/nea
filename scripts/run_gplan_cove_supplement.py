"""GPlan+CoVe supplement (spec 0045): CoVe on the 1008 Host+GPlan answers, scored like 1008.

The formal batch (run_id 1008, directory outputs/formal-test270-20261008) keeps its eight-arm
records unchanged. This script builds a FormalPhase with its own ledger in
outputs/formal-test270-20261008/supplement-gplan-cove/, feeds it the formal Host+GPlan answers as
parents, runs the same three CoVe steps and the same two-judge six-dimension scoring, then appends
the new group (recorded as ``CVL``) to the formal answers/judgments files.

  plan     print scope and transport, no API calls
  run      generate and score; rerun the same command to resume (successes are reused)
  analyze  write supplement-gplan-cove/analysis.json (no API calls)
  merge    append the CVL rows to the formal answers.json/judgments.json (idempotent)
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nea.providers import config  # noqa: E402
from nea.runtime.records import answer_sha256, canonical_bytes, digest, write_once  # noqa: E402

RUN_ID = "1015-gplan-cove-r3"
SOURCE = ROOT / "outputs" / "formal-test270-20261008"
SUPPLEMENT_ROOT = SOURCE / "supplement-gplan-cove"
# Paths in source-1008-manifest.json were registered before the batch was renamed from outputs/1008.
LEGACY_PREFIX = "outputs/1008/"
SOURCE_SHA = "693b89a9c22dbbbbab73dba003d88c09baaa389bd941f91a500424806ac4af66"
ARM, PARENT = "CVL", "L"
# Spec 0046 merged the Host+Plan groups into the formal files too; the 0045 registration excludes all merged groups.
MERGED_ARMS = (ARM, "SRP", "CVP", "FP")
TRANSPORT = {"mimo_api": "standard", "mimo_concurrency": 16, "minimax_concurrency": 2, "minimax_m3_api": "official"}
DIMENSIONS = ("answer_correctness", "query_fulfillment", "information_sufficiency", "explanatory_clarity",
              "instructional_utility", "overall_quality")
COMPARISONS = (("F-CVL", "F", ARM), ("CVL-L", ARM, "L"), ("CVL-SRL", ARM, "SRL"), ("CVL-CV", ARM, "CV"))


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def source_phase_dir(generator: str) -> Path:
    return SOURCE / "operator-v2" / generator / "test"


def original_sha256(path: Path) -> str:
    """SHA256 of a formal file with any merged CVL rows removed (the files are canonical JSON)."""
    data = path.read_bytes()
    if path.name in {"answers.json", "judgments.json"} and path.parent.name == "test":
        document = json.loads(data)
        data = canonical_bytes({**document, "rows": [row for row in document["rows"] if row.get("arm") not in MERGED_ARMS]})
    return hashlib.sha256(data).hexdigest()


def source_manifest() -> dict:
    """Hashes of the formal files this supplement reads, keyed by their registered outputs/1008 paths."""
    files = []
    for generator in config.EXPERIMENT_GENERATORS:
        files += sorted(source_phase_dir(generator).glob("*.json"))
    files += sorted((SOURCE / "operator-v2" / "shared" / "test").glob("*.json"))
    files += sorted((SOURCE / "operator-v2" / "analysis" / SOURCE_SHA).glob("*.json"))
    files += sorted((SOURCE / "paper-tables" / "test" / SOURCE_SHA).glob("*"))
    return {LEGACY_PREFIX + path.relative_to(SOURCE).as_posix(): original_sha256(path) for path in files}


def build_phases(args, run_root: Path, pool):
    from nea.method.graph import load_graph
    from nea.pipeline.formal_entry import _read_trusted_questions
    from nea.pipeline.formal_workflow import FormalEnvironment, FormalPhase
    from nea.pipeline.operator_entry import asset_identity
    from nea.pipeline.artifacts import RunDirectoryLock
    from nea.providers.router import RoleRouter
    qset = _read_trusted_questions(args.question_set)
    graph = load_graph(args.knowledge_root)
    assets = asset_identity(args.knowledge_root, args.question_set)
    pools = read(SOURCE / "operator-v2" / "shared" / "test" / "rag_pools.json")["pools"]
    qids = [row["qid"] for row in qset.split_rows("test")]
    phases = []
    for generator in args.generators:
        phase = FormalPhase(FormalEnvironment(run_root=run_root, run_id=args.run_id, generator=generator,
            question_set=qset, graph=graph, candidates={qid: [] for qid in qids}, rag_pools=pools,
            prompts_dir=args.prompts_dir, router=RoleRouter(generator, args.env_file, pool=pool),
            phase="test", sample_size=args.sample_size, asset_identity=assets))
        with RunDirectoryLock(phase.run.root):
            phase.run.initialize()
        phases.append(phase)
    return phases


def cove_on_parent(phase, parents):
    """FormalPhase._cove with an explicit parent answer per cell (the 1008 Host+GPlan answer)."""
    from nea.method import formal_operators as op
    jobs, contexts, stages = {}, {}, {}
    for cell, parent in parents.items():
        key = (*cell, "CV")
        contexts[key] = phase._context(key, parent, generic=True)
        payload = {**phase._payloads[cell], "answer": parent, "max_verification_questions": 3}
        jobs[key] = phase.job(phase._by_qid[cell[0]], cell[1], "CV", "cove_verification_questions",
                              payload, parent=answer_sha256(parent))
        stages[key] = {"cv_questions": jobs[key]}
    questions = phase._wave(jobs)
    independent = {}
    for key, result in questions.items():
        rows = result["verification_questions"]
        if len({item["question_id"] for item in rows}) != len(rows):
            continue
        payload = {**phase._payloads[key[:2]], "verification_questions": rows}
        independent[key] = phase.job(phase._by_qid[key[0]], key[1], "CV", "cove_verification_answer",
                                     payload, parent=contexts[key].parent_sha256)
        stages[key]["cv_answer"] = independent[key]
    verification = phase._wave(independent)
    revisions = {}
    for key, result in verification.items():
        expected = [item["question_id"] for item in questions[key]["verification_questions"]]
        if sorted(expected) != sorted(item["question_id"] for item in result["verification_answers"]):
            continue
        ctx = contexts[key]
        payload = {**phase._payloads[key[:2]], "answer": ctx.parent_answer,
                   "verification_questions": questions[key]["verification_questions"],
                   "verification_answers": result["verification_answers"], "max_chars": ctx.budget["max_chars"],
                   "allowed_citations": list(ctx.allowed_citations)}
        revisions[key] = phase.job(phase._by_qid[key[0]], key[1], "CV", "cove_revision", payload,
                                   parent=ctx.parent_sha256)
        stages[key]["revise"] = revisions[key]
    candidates = phase._wave(revisions)
    final = {}
    for key, ctx in contexts.items():
        candidate = candidates.get(key)
        valid, reason = op.validate_candidate(ctx, candidate)
        if candidate is None:
            reason = "cove_stage_failed"
        answer = candidate if valid else ctx.parent_answer
        record = None if candidate is None else {"answer": candidate, "answer_sha256": answer_sha256(candidate),
                                                 "status": "accepted" if valid else reason, "semantic": None}
        final[key[:2]] = {"answer": answer, "answer_sha256": answer_sha256(answer),
            "parent_sha256": ctx.parent_sha256, "context_sha256": ctx.context_sha256,
            "operator_version": "nea-cove-v1", "parent_arm": PARENT,
            "status": "accepted" if valid else "parent_retained", "revised": valid,
            "fallback_reason": None if valid else reason, "candidate_record": record,
            "verification_questions": questions.get(key), "verification_answers": verification.get(key),
            "validation_policy": "independent-verification-answers-and-structural-rules",
            "request_keys": {stage: phase.request_keys.get(job) for stage, job in stages[key].items()},
            "calls": {stage: len(list((phase.run.root / "attempts" / phase.request_keys[job]).glob("*-reserved.json")))
                      if job in phase.request_keys else 0 for stage, job in stages[key].items()},
            "comparison": "main"}
    return final


def long_path_safe(write):
    """Address paths near Windows MAX_PATH in extended-length form; same file, same bytes.

    The MiniMax generator directory plus an incomplete-judgment snapshot name exceeds 260 characters.
    """
    def wrapped(path, payload):
        full = str(Path(path).resolve())
        if sys.platform == "win32" and len(full) >= 240 and not full.startswith("\\\\?\\"):
            path = Path("\\\\?\\" + full)
        return write(path, payload)
    return wrapped


def run_generator(phase, generation_only: bool = False) -> dict:
    from nea.pipeline.artifacts import RunDirectoryLock
    from nea.pipeline import formal_evaluation
    from nea.pipeline.formal_evaluation import run_evaluations
    from nea.pipeline.registry import BASES
    if not getattr(formal_evaluation.write_once, "long_path_safe", False):
        formal_evaluation.write_once = long_path_safe(formal_evaluation.write_once)
        formal_evaluation.write_once.long_path_safe = True
    generator = phase.run.identity["generator"]
    source = {(row["qid"], row["base"]): row for row in read(source_phase_dir(generator) / "answers.json")["rows"]
              if row["arm"] == PARENT and row.get("comparison", "main") == "main"}
    phase._scheduled_jobs = set()
    phase._cells = [(row["qid"], base) for row in phase.rows for base in BASES]
    phase._payloads = {cell: phase.base_payload(phase._by_qid[cell[0]], cell[1]) for cell in phase._cells}
    phase._tasks = {}
    parents = {}
    for cell in phase._cells:
        row = source[cell]
        if row.get("status") != "ok" or row["evidence"] != phase._payloads[cell]["knowledge_chunks"]:
            raise RuntimeError(f"1008 Host+GPlan parent or evidence differs for {cell}")
        parents[cell] = row["answer"]
    # The transport registers CoVe stages under arm CV with parent arm B; point that slot at the L parent.
    phase.parent_index = {(*cell, "B"): answer_sha256(text) for cell, text in parents.items()}
    final = cove_on_parent(phase, parents)
    rows = []
    for cell in phase._cells:
        record = final.get(cell)
        failed = record is None or record["fallback_reason"] == "cove_stage_failed"
        row = phase._answer_row((*cell, ARM), None if failed else record["answer"], revision=record)
        row["parent_arm"] = PARENT
        rows.append(row)
    complete = all(row["status"] == "ok" for row in rows) and not phase.stop_reason
    with RunDirectoryLock(phase.run.root):
        if complete:
            write_once(phase.run.root / "supplement-answers.json", {"rows": rows})
        else:
            write_once(phase.run.root / "snapshots" / digest(rows) / "supplement-answers.json", {"rows": rows})
    summary = {"generator": generator, "answers": sum(row["status"] == "ok" for row in rows),
               "positions": len(rows), "generation_complete": complete, "stop_reason": phase.stop_reason}
    if complete and not generation_only:
        evaluation = run_evaluations(phase, rows, scoring_only=True)["summary"]
        summary.update(judgments=evaluation["judgments"], judgments_missing=evaluation["judgments_missing"],
                       evaluation_status=evaluation["status"])
    summary["complete"] = complete and summary.get("evaluation_status") == "COMPLETE"
    return summary


def supplement_judgments(phase_root: Path) -> list[dict]:
    found = sorted(phase_root.glob("quality/*/judgments.json"))
    if len(found) != 1:
        raise RuntimeError(f"expected one complete supplement judgment file in {phase_root}, found {len(found)}")
    return read(found[0])["rows"]


def analyze(run_root: Path) -> dict:
    from statistics import fmean
    from nea.evaluation.formal_analysis import _estimate, _round
    from nea.evaluation.statistics import BOOTSTRAP_REPLICATES, SEED
    quality, chars, revised, calls = {}, {}, {}, {}
    for generator in config.EXPERIMENT_GENERATORS:
        # The formal files may already hold the merged CVL rows; take them from the supplement ledger only.
        answers = [row for row in read(source_phase_dir(generator) / "answers.json")["rows"] if row["arm"] not in MERGED_ARMS]
        judgments = [row for row in read(source_phase_dir(generator) / "judgments.json")["rows"]
                     if row.get("comparison", "main") == "main" and row["arm"] not in MERGED_ARMS]
        phase_root = run_root / "operator-v2" / generator / "test"
        extra = read(phase_root / "supplement-answers.json")["rows"]
        answers += extra
        judgments += supplement_judgments(phase_root)
        for row in answers:
            if row.get("comparison", "main") == "main":
                chars.setdefault(row["arm"], []).append(len(row["answer"]))
        for row in extra:
            revised.setdefault(ARM, []).append(row["answer_sha256"] != row["parent_sha256"])
            for stage, count in row["revision"]["calls"].items():
                calls[stage] = calls.get(stage, 0) + count
        for row in judgments:
            if row["status"] != "ok":
                raise RuntimeError("missing judgment")
            quality.setdefault((row["qid"], row["arm"]), []).append(row["scores"])
    per = {key: {d: fmean(s[d] for s in scores) for d in DIMENSIONS} for key, scores in quality.items()}
    if any(len(scores) != 8 for scores in quality.values()):
        raise RuntimeError("each question needs four cells x two judges per group")
    arms = sorted({arm for _, arm in per})
    qids = sorted({qid for qid, _ in per})

    def delta(left, right, dimension="overall_quality"):
        return {qid: per[qid, left][dimension] - per[qid, right][dimension] for qid in qids}

    reference = json.loads(next(row["value"] for row in __import__("csv").DictReader(
        (SOURCE / "paper-tables" / "test" / SOURCE_SHA / "table-06.csv").open(encoding="utf-8"))
        if row["record_type"] == "inference"))["primary"]["comparisons"]["F-L"]
    check = _estimate(delta("F", "L"))
    # The archived reference used target clusters; only its point effect is comparable.
    # New uncertainty is computed from whole-question resampling, never matched to old intervals.
    reproduced = _round(check["effect"]) == _round(reference["effect"])
    comparisons = {label: {"overall_quality": _estimate(delta(left, right)),
                           "answer_correctness": _estimate(delta(left, right, "answer_correctness"))}
                   for label, left, right in COMPARISONS}
    report = {"protocol": "nea-formal8-gplan-cove-supplement-v1", "source_analysis_sha256": SOURCE_SHA,
        "bootstrap": {"unit": "qid", "replicates": BOOTSTRAP_REPLICATES, "seed": SEED},
        "n_qids": len(qids), "reference_check": {"comparison": "F-L", "reproduced": reproduced,
            "scope": "point_effect_only", "historical_uncertainty_comparable": False},
        "means": {arm: {d: _round(fmean(per[qid, arm][d] for qid in qids)) for d in DIMENSIONS} for arm in arms},
        "mean_chars": {arm: _round(fmean(values)) for arm, values in sorted(chars.items())},
        "comparisons": comparisons,
        "revision": {"positions": len(revised[ARM]), "revised": sum(revised[ARM]),
                     "revised_rate": _round(sum(revised[ARM]) / len(revised[ARM]))},
        "requests": {"by_stage": dict(sorted(calls.items())), "per_answer": _round(sum(calls.values()) / len(revised[ARM]))},
        "status": "ESTIMATED" if reproduced else "REFERENCE_MISMATCH",
        "note": "Supplementary comparisons; not part of the 1008 fixed sequence or its Holm family."}
    return report


def merge(run_root: Path) -> dict:
    """Append the CVL answers and judgments to the formal answers/judgments files (idempotent).

    The original rows stay byte-identical: dropping the CVL rows and re-serializing reproduces the
    SHA256 recorded in source-1008-manifest.json before the supplement ran.
    """
    from nea.runtime.records import canonical_bytes
    registered = read(run_root / "source-1008-manifest.json")
    summary = {}
    for generator in config.EXPERIMENT_GENERATORS:
        phase_root = run_root / "operator-v2" / generator / "test"
        additions = {"answers": read(phase_root / "supplement-answers.json")["rows"],
                     "judgments": supplement_judgments(phase_root)}
        for name, extra in additions.items():
            path = source_phase_dir(generator) / f"{name}.json"
            document = read(path)
            present = [row for row in document["rows"] if row.get("arm") == ARM]
            key = next(k for k in registered if k.endswith(f"/{generator}/test/{name}.json"))
            if original_sha256(path) != registered[key]:
                raise RuntimeError(f"original rows of {path} differ from the registered SHA256")
            if present:
                if present != extra:
                    raise RuntimeError(f"{path} already holds different {ARM} rows")
                summary[f"{generator}/{name}"] = "already merged"
                continue
            data = canonical_bytes({**document, "rows": document["rows"] + extra})
            temp = path.with_suffix(".json.tmp")
            temp.write_bytes(data)
            temp.replace(path)
            summary[f"{generator}/{name}"] = f"+{len(extra)} rows"
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("plan", "run", "analyze", "merge"))
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--sample-size", type=int, help="first-N smoke run in its own run-id; never a result")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--question-set", type=Path, default=ROOT / "knowledge_base/questions.jsonl")
    parser.add_argument("--knowledge-root", type=Path, default=ROOT / "knowledge_base")
    parser.add_argument("--prompts-dir", type=Path, default=ROOT / "prompts")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--generator", choices=("all", *config.EXPERIMENT_GENERATORS), default="all",
                        help="limit generation to one generator (judging always needs both judge routes)")
    parser.add_argument("--max-rounds", type=int, default=40,
                        help="recovery rounds (each grants 8 attempts per unfinished request; quota 429 waits 1 h)")
    parser.add_argument("--generation-only", action="store_true",
                        help="generate and store CoVe answers without scoring; rerun without it to score")
    args = parser.parse_args(argv)
    args.generators = list(config.EXPERIMENT_GENERATORS) if args.generator == "all" else [args.generator]
    if args.sample_size is not None and args.run_id == RUN_ID:
        raise SystemExit("a sample run needs its own --run-id")
    from nea.runtime.identity import validate_run_root
    run_root = SUPPLEMENT_ROOT if args.run_id == RUN_ID else validate_run_root(ROOT, args.output_root, args.run_id)
    if args.command == "plan":
        from nea.pipeline.formal_entry import execution_plan
        n = 270 if args.sample_size is None else args.sample_size
        print(json.dumps({"run_root": str(run_root), "transport": execution_plan(**TRANSPORT),
            "positions_per_generator": n * 2, "generation_requests_without_retries": n * 2 * 2 * 3,
            "judgment_requests_without_retries": n * 2 * 2 * 2}, ensure_ascii=False, indent=1))
        return 0
    if args.command == "merge":
        print(json.dumps(merge(run_root), ensure_ascii=False))
        return 0
    if args.command == "analyze":
        first, second = analyze(run_root), analyze(run_root)
        if canonical_bytes(first) != canonical_bytes(second):
            raise RuntimeError("analysis is not deterministic")
        target = run_root / "analysis.json"
        if target.exists() and target.read_bytes() != canonical_bytes(first):
            raise RuntimeError("analysis.json exists with different content")
        if not target.exists():
            target.write_bytes(canonical_bytes(first))
        print(json.dumps({k: first[k] for k in ("status", "reference_check", "revision", "requests")}, ensure_ascii=False))
        return 0 if first["status"] == "ESTIMATED" else 2
    from nea.pipeline.entry import repository_cli_lock
    from nea.providers.pool import ProviderPool
    run_root.mkdir(parents=True, exist_ok=True)
    manifest = source_manifest()
    write_once(run_root / "source-1008-manifest.json", manifest)
    if read(run_root / "source-1008-manifest.json") != manifest:
        raise RuntimeError("formal inputs changed since the supplement was registered")
    # The 1008 recovery mechanism (policy authorized 2026-10-07): per round, each unfinished request gets
    # eight more attempts; quota-type 429s defer the batch for one hour instead of marking it failed.
    import time
    from nea.runtime.formal_recovery import POLICY, retry_at
    write_once(run_root / "recovery-policy.json", POLICY)
    rounds = run_root / "supervision" / "rounds"
    summaries, complete = [], False
    for _ in range(args.max_rounds):
        wait = max(0.0, max((json.loads(p.read_text(encoding="utf8"))["retry_at"]
                             for p in (run_root / "supervision" / "cooldowns").glob("*.json")), default=0) - time.time())
        if wait:
            print(json.dumps({"event": "cooldown", "seconds": round(wait)}), flush=True)
            time.sleep(min(wait, 600))
            continue
        number = len(list(rounds.glob("*.json"))) + 1
        write_once(rounds / f"{number:06d}.json", {"round": number, "time": time.time(), "transport": TRANSPORT,
                                                  "generators": args.generators, "generation_only": args.generation_only})
        with repository_cli_lock(ROOT):
            pool = ProviderPool(args.env_file, **TRANSPORT)
            phases = build_phases(args, run_root, pool)
            with ThreadPoolExecutor(max_workers=len(phases), thread_name_prefix="gplan-cove") as executor:
                summaries = list(executor.map(lambda phase: run_generator(phase, args.generation_only), phases))
        complete = all(item["complete"] or (args.generation_only and item["generation_complete"]) for item in summaries)
        print(json.dumps({"round": number, "complete": complete, "generators": summaries}, ensure_ascii=False), flush=True)
        if complete:
            break
        time.sleep(30)
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
