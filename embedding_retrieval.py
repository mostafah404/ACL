from neo4j import GraphDatabase
import networkx as nx
from node2vec import Node2Vec
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


class EmbeddingRetriever:

    def __init__(self, driver, embed_dim=128):
        self.driver = driver
        self.embed_dim = embed_dim

  
    def export_graph(self):
        with self.driver.session() as session:
            result = session.run("""
                MATCH (a)-[]->(b)
                RETURN elementId(a) AS src, elementId(b) AS dst
            """)
            edges = [(r["src"], r["dst"]) for r in result]

        G = nx.DiGraph()
        G.add_edges_from(edges)
        return G

    # Model 1: Node2Vec

    def train_node2vec(self, G):
        node2vec = Node2Vec(
            G,
            dimensions=self.embed_dim,
            walk_length=10,
            num_walks=50,
            workers=2
        )
        model = node2vec.fit(window=10, min_count=1)
        return {node: model.wv[node] for node in G.nodes()}

   
    # Model 2: GraphSAGE
 
    class SAGEModel(torch.nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim):
            super().__init__()
            self.conv1 = SAGEConv(in_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, out_dim)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            x = self.conv2(x, edge_index)
            return x

    def train_graphsage(self, G):
        nodes = list(G.nodes())
        mapping = {node: idx for idx, node in enumerate(nodes)}

        edge_index = torch.tensor(
            [[mapping[u], mapping[v]] for u, v in G.edges()],
            dtype=torch.long
        ).t().contiguous()

        x = torch.eye(len(nodes))
        data = Data(x=x, edge_index=edge_index)

        model = self.SAGEModel(
            in_dim=len(nodes),
            hidden_dim=128,
            out_dim=self.embed_dim
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        for _ in range(50):
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = ((out @ out.t()) - torch.eye(len(nodes))).pow(2).mean()
            loss.backward()
            optimizer.step()

        emb = out.detach().numpy()
        return {node: emb[mapping[node]] for node in nodes}

    
    def store_embeddings(self, name, embeddings):
        with self.driver.session() as session:
            for node_eid, vector in embeddings.items():
                session.run("""
                    MATCH (n)
                    WHERE elementId(n) = $eid
                    SET n[$prop] = $vec
                """, eid=node_eid, prop=name, vec=vector.tolist())

    
    def create_vector_index(self, name, label):
      with self.driver.session() as session:
        session.run(f"""
            CREATE VECTOR INDEX {label.lower()}_{name}_index IF NOT EXISTS
            FOR (n:{label})
            ON (n.{name})
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {self.embed_dim},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
        """)

   
    def query_similar(self, name, label, node_eid, k=5):
        with self.driver.session() as session:
            result = session.run(f"""
                MATCH (n:{label}) WHERE elementId(n) = $eid
                WITH n.{name} AS embed
                CALL db.index.vector.queryNodes(
                    '{label.lower()}_{name}_index',
                    $k,
                    embed
                )
                YIELD node, score
                RETURN elementId(node) AS id, score
            """, eid=node_eid, k=k)

            return result.data()
