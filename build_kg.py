# kg_retrieval.py — FINAL FIXED VERSION

from neo4j import GraphDatabase
import json

URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "password"

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))


# ------------------------------------------------
# TOPIC LOOKUP BY ID (MOST RELIABLE)
# ------------------------------------------------
def topic_id(topic_key):
    """
    Convert canonical topic_key → Neo4j node ID
    Example: DLG_Cap → Topic::DLG_Cap
    """
    return f"Topic::{topic_key}"


# ------------------------------------------------
# 1. Get nodes (Chunk/Clause) linked to Topic
# ------------------------------------------------
def get_topic_related_nodes(topic_key):

    tid = topic_id(topic_key)

    query = """
    MATCH (t:Topic {id: $tid})
    OPTIONAL MATCH (c:Chunk)-[:pertainsTo]->(t)
    OPTIONAL MATCH (cl:Clause)-[:pertainsTo]->(t)
    RETURN COLLECT(DISTINCT c.id) + COLLECT(DISTINCT cl.id) AS nodes
    """

    with driver.session() as session:
        result = session.run(query, tid=tid)
        nodes = result.single()["nodes"]

        # remove None
        return [n for n in nodes if n]


# ------------------------------------------------
# 2. Expand KG facts: each node’s relations
# ------------------------------------------------
def get_kg_facts(node_ids):

    if not node_ids:
        return []

    query = """
    UNWIND $ids AS nid
    MATCH (n {id: nid})-[r]->(x)
    RETURN nid AS source, type(r) AS relation,
           x.id AS target, x.label AS label
    """

    with driver.session() as session:
        result = session.run(query, ids=node_ids)
        facts = []

        for row in result:
            facts.append({
                "source": row["source"],
                "relation": row["relation"],
                "target": row["target"],
                "label": row["label"]
            })

        return facts
