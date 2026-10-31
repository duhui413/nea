"""Maintain the independent Neo4j nea graph; runtime still reads local JSONL.

Credentials: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD from process environment
then .env. Defaults are local Bolt and neo4j user; password has no default.
Reads are --stats / --verify. Writes require --upsert; database creation also
requires --create-database. Never clears databases or touches other graphs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
DATABASE = "nea"
GRAPH_FIELDS = ("depth", "height", "pi", "betweenness", "n_ancestors",
                "n_descendants", "scc_id", "graph_level")


def connection_settings():
    from dotenv import dotenv_values
    values = dotenv_values(ROOT / ".env") if (ROOT / ".env").exists() else {}
    def get(key, default=None):
        return os.environ[key] if key in os.environ else values.get(key, default)
    uri = get("NEO4J_URI", "bolt://localhost:7687")
    user = get("NEO4J_USER", "neo4j")
    password = get("NEO4J_PASSWORD")
    if not password:
        raise ValueError("NEO4J_PASSWORD is missing from process environment/.env")
    from urllib.parse import urlsplit
    parsed = urlsplit(uri)
    if parsed.scheme not in {"bolt", "bolt+s", "bolt+ssc", "neo4j", "neo4j+s", "neo4j+ssc"} or parsed.username or parsed.password:
        raise ValueError("NEO4J_URI must be a Bolt/Neo4j URI without credentials")
    return uri, user, password


def source_rows():
    def rows(name):
        return [json.loads(line) for line in (ROOT / f"knowledge_base/{name}.jsonl").read_bytes().splitlines() if line]
    concepts, edges = rows("concepts"), rows("edges")
    graph = {row["id"]: row for row in rows("graph")}
    nodes = {row["id"]: {key: row[key] for key in ("id", "name", "chapter", "aliases", "explain")} for row in concepts}
    if len(nodes) != len(concepts) or set(nodes) != set(graph) or len(nodes) != 6600:
        raise ValueError("source concept/graph ids violate the 6600 node contract")
    for cid, node in nodes.items():
        node.update({key: graph[cid][key] for key in GRAPH_FIELDS})
    relations = {row["id"]: row for row in edges}
    if len(relations) != len(edges) or len(edges) != 13949 or any(
        row["type"] != "prerequisite" or row["node1"] not in nodes or row["node2"] not in nodes for row in edges
    ):
        raise ValueError("source violates the 13949 prerequisite relation contract")
    hashes = {name: hashlib.sha256((ROOT / f"knowledge_base/{name}.jsonl").read_bytes()).hexdigest()
              for name in ("concepts", "edges", "graph")}
    return nodes, relations, hashes


def read_state(session):
    nodes = session.run("MATCH (n) RETURN labels(n) AS labels, properties(n) AS props").data()
    edges = session.run("MATCH (a)-[r]->(b) RETURN a.id AS node1,b.id AS node2,type(r) AS kind,properties(r) AS props").data()
    return nodes, edges


def validate_subset(state, wanted_nodes, wanted_edges, *, complete):
    nodes, edges = state
    node_ids, edge_ids = [], []
    for row in nodes:
        props = row["props"]
        cid = props.get("id")
        if row["labels"] != ["Concept"] or cid not in wanted_nodes or props != wanted_nodes[cid]:
            raise ValueError("existing node does not match the current source; stopped without clearing")
        node_ids.append(cid)
    for row in edges:
        props = row["props"]
        eid = props.get("id")
        expected = wanted_edges.get(eid)
        if expected is None or row["kind"] != "PREREQUISITE" or row["node1"] != expected["node1"] or row["node2"] != expected["node2"] or props != {key: expected[key] for key in ("id", "chapter", "type", "explain")}:
            raise ValueError("existing relation does not match current source; stopped without clearing")
        edge_ids.append(eid)
    if len(set(node_ids)) != len(node_ids) or len(set(edge_ids)) != len(edge_ids):
        raise ValueError("existing ids are duplicated")
    if complete and (set(node_ids) != set(wanted_nodes) or set(edge_ids) != set(wanted_edges)):
        raise ValueError(f"incomplete graph: nodes={len(nodes)} edges={len(edges)}")


def import_rows(session, nodes, edges):
    session.run("CREATE CONSTRAINT nea_concept_id IF NOT EXISTS FOR (n:Concept) REQUIRE n.id IS UNIQUE").consume()
    def batches(rows):
        for start in range(0, len(rows), 500):
            yield rows[start:start + 500]
    for batch in batches(list(nodes.values())):
        session.run("UNWIND $rows AS row MERGE (n:Concept {id:row.id}) SET n = row", rows=batch).consume()
    for batch in batches(list(edges.values())):
        session.run("UNWIND $rows AS row MATCH (a:Concept {id:row.node1}), (b:Concept {id:row.node2}) "
                    "MERGE (a)-[r:PREREQUISITE {id:row.id}]->(b) "
                    "SET r.chapter=row.chapter,r.type=row.type,r.explain=row.explain", rows=batch).consume()


def verify(session, database_row, nodes, edges):
    outcomes = []
    def check(name, function):
        try:
            function()
        except Exception as exc:
            outcomes.append(False)
            print(f"FAIL {name}: {exc}", flush=True)
        else:
            outcomes.append(True)
            print(f"PASS {name}", flush=True)
    def require(ok):
        if not ok:
            raise AssertionError("contract mismatch")
    state = read_state(session)
    check("nea online and read-write", lambda: require(database_row["currentStatus"] == "online" and database_row["access"] == "read-write"))
    check("6600 total/Concept/unique nodes", lambda: require(len(state[0]) == 6600 and len({r["props"].get("id") for r in state[0]}) == 6600 and all(r["labels"] == ["Concept"] for r in state[0])))
    check("13949 total/PREREQUISITE/unique relations", lambda: require(len(state[1]) == 13949 and len({r["props"].get("id") for r in state[1]}) == 13949 and all(r["kind"] == "PREREQUISITE" for r in state[1])))
    check("all properties, endpoints and FP64 values match JSONL", lambda: validate_subset(state, nodes, edges, complete=True))
    check("graph levels 643/1401/4556", lambda: require(Counter(r["props"].get("graph_level") for r in state[0]) == {0: 643, 1: 1401, 2: 4556}))
    constraints = session.run("SHOW CONSTRAINTS YIELD type, labelsOrTypes, properties RETURN type, labelsOrTypes, properties").data()
    check("Concept.id uniqueness constraint", lambda: require(any(r["type"] == "UNIQUENESS" and r["labelsOrTypes"] == ["Concept"] and r["properties"] == ["id"] for r in constraints)))
    passed = sum(outcomes)
    print(f"VERDICT: {'PASS' if all(outcomes) else 'FAIL'} ({passed}/{len(outcomes)})")
    return 0 if all(outcomes) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--stats", action="store_true")
    action.add_argument("--verify", action="store_true")
    action.add_argument("--upsert", action="store_true")
    parser.add_argument("--create-database", action="store_true")
    args = parser.parse_args()
    if args.create_database and not args.upsert:
        parser.error("--create-database requires --upsert")
    uri, user, password = connection_settings()
    from neo4j import GraphDatabase
    with GraphDatabase.driver(uri, auth=(user, password), connection_timeout=10) as driver:
        driver.verify_connectivity()
        with driver.session(database="system") as system:
            version = system.run("CALL dbms.components() YIELD versions,edition RETURN versions,edition").data()
            databases = system.run("SHOW DATABASES YIELD name,currentStatus,access RETURN name,currentStatus,access").data()
            row = next((r for r in databases if r["name"] == DATABASE), None)
            if args.upsert and row is None and args.create_database:
                if not version or version[0]["edition"] != "enterprise":
                    raise ValueError("service does not support an independent nea database (Enterprise required)")
                system.run("CREATE DATABASE nea IF NOT EXISTS WAIT 30 SECONDS").consume()
                row = system.run("SHOW DATABASE nea YIELD name,currentStatus,access RETURN name,currentStatus,access").single().data()
        if args.stats:
            print(json.dumps({"service": version, "nea": row}, ensure_ascii=False))
            if row is not None:
                with driver.session(database=DATABASE, default_access_mode="READ") as session:
                    state = read_state(session)
                    print(json.dumps({"nodes": len(state[0]), "relationships": len(state[1])}))
            return 0
        if row is None:
            raise ValueError("nea does not exist; creation requires --upsert --create-database")
        if row["currentStatus"] != "online" or row["access"] != "read-write":
            raise ValueError("nea must be online and read-write")
        nodes, edges, hashes = source_rows()
        print(json.dumps({"source_sha256": hashes, "database": DATABASE}, ensure_ascii=False))
        with driver.session(database=DATABASE, default_access_mode="WRITE" if args.upsert else "READ") as session:
            if args.upsert:
                validate_subset(read_state(session), nodes, edges, complete=False)
                import_rows(session, nodes, edges)
                validate_subset(read_state(session), nodes, edges, complete=True)
                # Second same-source import verifies actual idempotence.
                import_rows(session, nodes, edges)
                validate_subset(read_state(session), nodes, edges, complete=True)
                print("PASS repeated same-source import: full content/counts unchanged")
            return verify(session, row, nodes, edges)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # Driver connection errors may include a credential-bearing endpoint.
        # Report class only; never print secrets or raw driver diagnostics.
        print(f"FAIL Neo4j: {type(exc).__name__}; check service/configuration and source compatibility", file=sys.stderr)
        print("VERDICT: FAIL (0/6)")
        raise SystemExit(1)
