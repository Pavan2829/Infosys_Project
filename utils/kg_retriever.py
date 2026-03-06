from neo4j import GraphDatabase

class KGRetriever:

    def __init__(self):
        try:
            self.driver = GraphDatabase.driver(
                "bolt://localhost:7687",
                auth=("neo4j","nani1234"),
                connection_timeout=2.0  # Add timeout to avoid long waits
            )
            self.available = True
        except Exception:
            self.driver = None
            self.available = False

    def retrieve_graph_context(self):
        if not self.available or self.driver is None:
            return []

        try:
            with self.driver.session() as session:

                result = session.run("""
                MATCH (a:Entity)-[r]->(b:Entity)
                RETURN a.name AS subject,
                       type(r) AS relation,
                       b.name AS object
                LIMIT 20
                """)

                triples = []

                for r in result:
                    triples.append(
                        f"{r['subject']} {r['relation']} {r['object']}"
                    )

                return triples
        except Exception:
            # If query fails, return empty list
            return []