# kg_retrieval.py — FIXED VERSION (canonical stored inside meta JSON string)

from neo4j import GraphDatabase
import json

URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "password"

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))


# ------------------------------------------------
# 1. Get chunks/clauses connected to a Topic
# ------------------------------------------------
def get_topic_related_nodes(topic_key):

    # Neo4j stores canonical inside meta JSON string → must search inside meta text
    topic_json = f'"canonical": "{topic_key}"'

    query = """
    MATCH (t:Topic)
    WHERE t.meta CONTAINS $topic_json

    OPTIONAL MATCH (c:Chunk)-[:pertainsTo]->(t)
    OPTIONAL MATCH (cl:Clause)-[:pertainsTo]->(t)

    RETURN COLLECT(DISTINCT c.id) + COLLECT(DISTINCT cl.id) AS nodes
    """

    with driver.session() as session:
        result = session.run(query, topic_json=topic_json)
        nodes = result.single()["nodes"]
        return [n for n in nodes if n]


# ------------------------------------------------
# 2. Expand KG context for given chunk/clause IDs
# ------------------------------------------------
def get_kg_facts(node_ids):
    if not node_ids:
        return []

    query = """
    UNWIND $ids AS nid
    MATCH (n {id: nid})-[r]->(x)
    RETURN nid AS source, type(r) AS relation, x.id AS target, x.label AS label
    """

    with driver.session() as session:
        result = session.run(query, ids=node_ids)
        return [
            {
                "source": row["source"],
                "relation": row["relation"],
                "target": row["target"],
                "label": row["label"]
            }
            for row in result
        ]
