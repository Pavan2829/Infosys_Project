from neo4j import GraphDatabase
import re

class Neo4jLoader:

    def __init__(self):
        self.driver = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "nani1234")
        )

    def close(self):
        self.driver.close()

    # -------- CREATE PAPER NODE --------
    def load_metadata(self, metadata):
        with self.driver.session() as session:
            session.execute_write(self._create_paper, metadata)

    @staticmethod
    def _create_paper(tx, metadata):
        tx.run("""
        MERGE (p:Paper {title:$title})
        SET p.year=$year,
            p.conference=$conference,
            p.domain=$domain
        """,
        title=metadata.get("title","Unknown"),
        year=metadata.get("year","Unknown"),
        conference=metadata.get("conference","Unknown"),
        domain=metadata.get("domain","Unknown")
        )

    # -------- CREATE RELATIONSHIPS --------
    def create_relationship(self, subject, relation, obj):

        relation = self.clean_relation(relation)

        with self.driver.session() as session:
            session.execute_write(self._create_rel, subject, relation, obj)

    @staticmethod
    def clean_relation(rel):
        rel = rel.upper()
        rel = re.sub(r'[^A-Z_]', '', rel)

        if rel == "":
            rel = "RELATED_TO"

        return rel

    @staticmethod
    def _create_rel(tx, subject, relation, obj):

        query = f"""
        MERGE (a:Entity {{name:$subject}})
        MERGE (b:Entity {{name:$object}})
        MERGE (a)-[:{relation}]->(b)
        """

        tx.run(query, subject=subject, object=obj)


# -------- RETRIEVE GRAPH KNOWLEDGE --------
def retrieve_graph_knowledge():

    try:
        driver = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "nani1234")
        )

        query = """
        MATCH (a)-[r]->(b)
        RETURN a.name AS subject, type(r) AS relation, b.name AS object
        LIMIT 10
        """

        with driver.session() as session:
            result = session.run(query)
            knowledge = ""

            for record in result:
                knowledge += f"{record['subject']} {record['relation']} {record['object']}. "

        driver.close()
        return knowledge

    except Exception:
        return "Graph Knowledge Not Available"