import torch
from torch_geometric.utils import dense_to_sparse

from utils import get_ae_model
from vae.helper_types import TupleGraph

ENCODER_PARAMS = {
    "autoencoder_type": "VAE",
    "features": 178,
    "edge_dim": None,
    "common": {
        "num_layers": 2,
        "hidden_channels": 384,
        "layer_type": "GDN",
        "norm_type": "GraphNorm",
        "random_embedding_size": 16,
    },
    "mu": {
        "num_layers": 1,
        "hidden_channels": 384,
        "layer_type": "GDN",
        "norm_type": "GraphNorm",
    },
    "logstd": {
        "num_layers": 1,
        "hidden_channels": 384,
        "layer_type": "GDN",
        "norm_type": "GraphNorm",
    },
    "name": "GDN_VAE"
}

DECODER_PARAMS = {
    "type": "CompositeDecoder",
    "adj": {
        "type": "DirectedInnerProductDecoder",
        "name": "INAAM",
        "n_layers": 1
    },
    "feature": {
        "type": "GNNDecoder",
        "layer_type": "GDN"
    }
}

model = get_ae_model(ENCODER_PARAMS, DECODER_PARAMS)
model.load("checkpoint")


def autoencode(model, graph):
    num_nodes = graph.num_nodes
    device = graph.x.device

    latent = model.encode(graph)
    latent = latent["mu"] + torch.randn_like(latent["logstd"]) * torch.exp(latent["logstd"])

    decoded = model.decode(TupleGraph(
        x=latent,
        num_nodes=num_nodes,
        batch=torch.zeros((num_nodes,), device=device, dtype=torch.long)
    ))
    decoded.adj = torch.sigmoid(decoded.adj)

    rand_sample = torch.rand_like(decoded.adj)
    decoded.adj[decoded.adj >= rand_sample] = 1
    decoded.adj[decoded.adj < rand_sample] = 0
    decoded.edge_index, _ = dense_to_sparse(decoded.adj.detach())
    decoded.adj = None

    return decoded

autoencode(model, )