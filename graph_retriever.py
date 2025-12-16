from neo4j import GraphDatabase
from baseline_retriever import BaselineRetriever
from embedding_retrieval import EmbeddingRetriever



class GraphRetriever:

    def __init__(self, uri, user, password, embed_dim=128):
        self.driver = GraphDatabase.driver(
            uri,
            auth=(user, password),
            notifications_min_severity="OFF"
        )

        self.baseline = BaselineRetriever(self.driver)
        self.embedder = EmbeddingRetriever(self.driver, embed_dim=embed_dim)


    def build_embeddings(self, method="node2vec"):
        G = self.embedder.export_graph()

        if method == "node2vec":
            emb = self.embedder.train_node2vec(G)
            name = "node2vec_embed"

        elif method == "graphsage":
            emb = self.embedder.train_graphsage(G)
            name = "sage_embed"

        else:
            raise ValueError("Unknown embedding method")

        self.embedder.store_embeddings(name, emb)

        for label in ["Passenger", "Journey", "Flight", "Airport"]:
            self.embedder.create_vector_index(name, label)

        return f"Embedding model '{name}' trained and indexed."


    def retrieve(self, intent, entities):

        if intent == "flights_from":
            return self.baseline.flights_from_airport(
                entities["origin"]
            )

        if intent == "flights_to":
            return self.baseline.flights_to_airport(
                entities["destination"]
            )

        if intent == "passenger_journeys":
            return self.baseline.passenger_journeys(
                entities["record_locator"]
            )

        if intent == "journey_flight":
            return self.baseline.journey_flight(
                entities["feedback_id"]
            )

        if intent == "flights_between":
            return self.baseline.flights_between(
                entities["origin"],
                entities["destination"]
            )

        if intent == "passengers_on_flight":
            return self.baseline.passengers_on_flight(
                entities["flight_number"]
            )

        if intent == "flights_by_fleet":
            return self.baseline.flights_by_fleet(
                entities["fleet_type"]
            )

        if intent == "similar_nodes":
            return self.embedder.query_similar(
                name=entities["embedding_name"],
                label=entities["label"],
                node_eid=entities["node_eid"],
                k=entities.get("k", 5)
            )

        return {"error": "Unknown intent"}
