"""Build the ``nea.concepts`` Milvus collection from the final knowledge base.

Under the new data contract a concept *is* the retrieval unit — there is no
separate chunk any more — so this collection replaces the old chunk-based one
for anything reading ``knowledge_base/``.

Structural values (``depth`` / ``graph_level`` / ancestor counts) are stored as
scalar fields so retrieval can filter by cognitive level without reading
``graph.jsonl`` back.

Document vectors are encoded with the same fixed runtime as queries and grounding:
CUDA FP16 forward (``nea.domain_models.configure_local_runtime``), L2-normalised,
then stored as binary half values in the FLOAT16_VECTOR field.  There is no CPU/FP32 fallback.

Choose ``--stats`` to read collection state or ``--upsert`` to build it.
Rebuilding an existing collection also requires ``--drop``.

Specification and acceptance criteria: docs/milvus.md

Usage (from the repository root):

    .venv\\Scripts\\python.exe scripts/database/build_milvus_concepts.py --stats
    .venv\\Scripts\\python.exe scripts/database/build_milvus_concepts.py --upsert
    .venv\\Scripts\\python.exe scripts/database/build_milvus_concepts.py --upsert --drop
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nea.domain_models import LOCAL_MODEL_ATTENTION, MODEL_PATHS, configure_local_runtime  # noqa: E402
from nea.storage.milvus import DEFAULT_COLLECTION, DEFAULT_DB, DEFAULT_URI, fp16_payload

KB_DIR = REPO_ROOT / "knowledge_base"
CONCEPTS_PATH = KB_DIR / "concepts.jsonl"
GRAPH_PATH = KB_DIR / "graph.jsonl"
DEFAULT_MODEL = MODEL_PATHS["retrieval_embedding"]
METRIC_TYPE = "COSINE"
INDEX_TYPE = "HNSW"
INDEX_PARAMS = {"M": 16, "efConstruction": 200}

SCALAR_FIELDS = ("name", "chapter", "aliases", "explain",
                 "depth", "height", "pi", "graph_level", "n_ancestors", "n_descendants")


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"missing input: {path}")
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def embed_text(concept: dict) -> str:
    """Deterministic text sent to the encoder. Keep in sync with docs/milvus.md."""
    lines = [f"{concept['name']}｜{concept['chapter']}"]
    aliases = concept.get("aliases") or []
    if aliases:
        lines.append("别名：" + "、".join(aliases))
    lines.append(concept["explain"])
    return "\n".join(lines)


def load_rows() -> list[dict]:
    concepts = read_jsonl(CONCEPTS_PATH)
    graph = {row["id"]: row for row in read_jsonl(GRAPH_PATH)}
    missing = {c["id"] for c in concepts} ^ set(graph)
    if missing:
        raise SystemExit(
            f"concepts.jsonl and graph.jsonl disagree on {len(missing)} ids, e.g. "
            f"{sorted(missing)[:3]}"
        )
    rows = []
    for concept in concepts:
        node = graph[concept["id"]]
        rows.append(
            {
                "id": concept["id"],
                "name": concept["name"],
                "chapter": concept["chapter"],
                "aliases": json.dumps(concept.get("aliases") or [], ensure_ascii=False),
                "explain": concept["explain"],
                "depth": int(node["depth"]),
                "height": int(node["height"]),
                "pi": float(node["pi"]),
                "graph_level": int(node["graph_level"]),
                "n_ancestors": int(node["n_ancestors"]),
                "n_descendants": int(node["n_descendants"]),
                "_text": embed_text(concept),
            }
        )
    return rows


def build_schema(client, dimension: int):
    from pymilvus import DataType

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.VARCHAR, max_length=64, is_primary=True)
    schema.add_field("name", DataType.VARCHAR, max_length=256)
    schema.add_field("chapter", DataType.VARCHAR, max_length=256)
    schema.add_field("aliases", DataType.VARCHAR, max_length=1024)
    schema.add_field("explain", DataType.VARCHAR, max_length=8192)
    schema.add_field("depth", DataType.INT64)
    schema.add_field("height", DataType.INT64)
    schema.add_field("pi", DataType.DOUBLE)
    schema.add_field("graph_level", DataType.INT64)
    schema.add_field("n_ancestors", DataType.INT64)
    schema.add_field("n_descendants", DataType.INT64)
    schema.add_field("vector", DataType.FLOAT16_VECTOR, dim=dimension)
    return schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--upsert", action="store_true", help="build the collection with real embeddings")
    action.add_argument("--stats", action="store_true", help="read current collection state and exit")
    parser.add_argument("--create-database", action="store_true",
                        help="with --upsert, create the database if it does not exist yet")
    parser.add_argument("--drop", action="store_true", help="with --upsert, replace an existing collection")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="embedding model directory under models/ (default: %(default)s)")
    parser.add_argument("--models-dir", type=Path, default=REPO_ROOT / "models")
    parser.add_argument("--batch-size", type=int, default=32, help="encoder batch size")
    parser.add_argument("--insert-batch", type=int, default=500, help="Milvus insert batch size")
    args = parser.parse_args()
    if (args.drop or args.create_database) and not args.upsert:
        parser.error("--drop/--create-database require --upsert")
    args.uri, args.database, args.collection = DEFAULT_URI, DEFAULT_DB, DEFAULT_COLLECTION
    return args


def connect(args):
    from pymilvus import MilvusClient

    root = MilvusClient(uri=args.uri)
    databases = root.list_databases()
    if args.database not in databases and args.create_database:
        root.create_database(args.database)
    elif args.database not in databases:
        raise SystemExit(
            f"database {args.database!r} does not exist on {args.uri} "
            f"(found {databases}); pass --create-database with --upsert to create it"
        )
    return MilvusClient(uri=args.uri, db_name=args.database)


def report_state(client, collection: str) -> dict:
    if collection not in client.list_collections():
        return {"exists": False}
    stats = client.get_collection_stats(collection)
    described = client.describe_collection(collection)
    return {
        "exists": True,
        "row_count": int(stats.get("row_count", 0)),
        "fields": [f["name"] for f in described.get("fields", [])],
    }


def main() -> int:
    args = parse_args()
    rows = load_rows()
    client = connect(args)
    state = report_state(client, args.collection)

    if args.stats:
        print(json.dumps(
            {
                "uri": args.uri, "database": args.database, "collection": args.collection,
                "source_rows": len(rows), "collection_state": state,
            },
            ensure_ascii=False, indent=2))
        return 0

    if state["exists"] and not args.drop:
        print(json.dumps(
            {
                "status": "blocked",
                "reason": f"collection {args.collection!r} already exists in "
                          f"{args.database!r}; pass --drop to rebuild it",
                "collection_state": state,
            },
            ensure_ascii=False, indent=2), file=sys.stderr)
        return 2

    import torch
    from sentence_transformers import SentenceTransformer

    model_dir = args.models_dir / args.model
    if not model_dir.is_dir():
        raise SystemExit(f"embedding model not found: {model_dir}")
    configure_local_runtime()
    encoder = SentenceTransformer(str(model_dir), device="cuda", local_files_only=True,
                                  model_kwargs={"dtype": torch.float16, "attn_implementation": LOCAL_MODEL_ATTENTION})
    encoder.half().eval()
    max_tokens = encoder.max_seq_length
    tokenizer = encoder.tokenizer

    texts = [row["_text"] for row in rows]
    truncated = [
        row["id"] for row, text in zip(rows, texts)
        if len(tokenizer(text, add_special_tokens=True)["input_ids"]) > max_tokens
    ]

    print(f"encoding {len(texts)} concepts with {args.model} on CUDA FP16 ...", file=sys.stderr)
    vectors = encoder.encode(
        texts, batch_size=args.batch_size, normalize_embeddings=True,
        show_progress_bar=True, convert_to_numpy=True,
    ).astype("float16")
    dimension = int(vectors.shape[1])

    if state["exists"]:
        client.drop_collection(args.collection)

    schema = build_schema(client, dimension)
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", index_type=INDEX_TYPE,
                           metric_type=METRIC_TYPE, params=INDEX_PARAMS)
    client.create_collection(args.collection, schema=schema, index_params=index_params)

    inserted = 0
    for start in range(0, len(rows), args.insert_batch):
        batch = []
        for row, vector in zip(rows[start:start + args.insert_batch],
                               vectors[start:start + args.insert_batch]):
            record = {key: row[key] for key in ("id", *SCALAR_FIELDS)}
            record["vector"] = fp16_payload(vector)
            batch.append(record)
        client.insert(args.collection, batch)
        inserted += len(batch)

    # Milvus counts only sealed segments, so an unflushed collection reports 0 rows.
    client.flush(args.collection)
    client.load_collection(args.collection)
    row_count = int(client.get_collection_stats(args.collection).get("row_count", 0))
    if row_count != len(rows):
        raise SystemExit(f"read-back mismatch: Milvus reports {row_count} rows, expected {len(rows)}")

    probe_ids = [rows[i]["id"] for i in range(0, len(rows), max(1, len(rows) // 5))][:5]
    probe_vectors = [fp16_payload(vectors[[r["id"] for r in rows].index(pid)]) for pid in probe_ids]
    hits = client.search(args.collection, data=probe_vectors, limit=1,
                         output_fields=["id"], search_params={"metric_type": METRIC_TYPE})
    for pid, hit in zip(probe_ids, hits):
        top = hit[0]
        if top["id"] != pid or float(top["distance"]) < 0.99:
            raise SystemExit(
                f"self-retrieval failed for {pid}: got {top['id']} at {top['distance']}")

    print(json.dumps({"status": "ready", "inserted": inserted,
                      "collection": args.collection, "database": args.database,
                      "row_count": row_count, "embedding_dimension": dimension, "vector_dtype": "FLOAT16_VECTOR",
                      "pi_dtype": "DOUBLE", "self_retrieval_probes": len(probe_ids),
                      "truncated_count": len(truncated)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
