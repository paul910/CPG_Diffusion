from vae.autoencoder import AutoEncoder
from vae.decoder import CompositeDecoder
from vae.decoders import GATDecoder, GDNDecoder, MLPAdjDecoder
from vae.decoders import InnerProductDecoder, NodeVAEDecoder, DirectedInnerProductDecoder
from vae.encoders import GCNEncoder, GATEncoder, NodeVAEEncoder, GDNEncoder
from vae.encoders import GGNNEncoder, GINEncoder


def get_ae_model(encoder_params, decoder_params):
    encoder = get_encoder(encoder_params)
    decoder_params["hidden_channels"] = encoder_params["hidden_channels"]
    decoder_params["features"] = encoder_params["features"]
    decoder_params["edge_dim"] = encoder_params["edge_dim"]
    decoder = get_decoder(decoder_params)

    return AutoEncoder(encoder, decoder)


def get_decoder(params):
    if params["type"] == "VAEDecoder":
        params["decoder"]["hidden_channels"] = params["hidden_channels"]
        params["decoder"]["features"] = params["features"]
        return NodeVAEDecoder(get_decoder(params["decoder"]))
    if params["type"] == "InnerProductDecoder":
        return InnerProductDecoder()
    if params["type"] == "DirectedInnerProductDecoder":
        return DirectedInnerProductDecoder(params)
    if params["type"] == "MLPAdj":
        return MLPAdjDecoder(params)
    if params["type"] == "CompositeDecoder":
        split = min(params["hidden_channels"] // 2, 128)

        params["adj"]["hidden_channels"] = split
        params["adj"]["features"] = params["features"]
        adj_decoder = get_decoder(params["adj"])

        params["feature"]["hidden_channels"] = params["hidden_channels"] - split
        params["feature"]["features"] = params["features"]
        feature_decoder = get_decoder(params["feature"])

        return CompositeDecoder(adj_decoder, feature_decoder)
    if params["type"] == "GNNDecoder":
        if params["layer_type"] == "GAT":
            return GATDecoder(params)
        if params["layer_type"] == "GDN":
            return GDNDecoder(params)

    raise ValueError(f"Could not construct decoder for {params}")


def get_encoder(params):
    if params.get("autoencoder_type") == "VAE":
        params["hidden_channels"] = params["mu"]["hidden_channels"]
        params["common"]["features"] = params["features"]
        params["common"]["edge_dim"] = params["edge_dim"]
        common_encoder = get_encoder(params["common"])

        params["mu"]["features"] = params["common"]["hidden_channels"]
        params["logstd"]["features"] = params["common"]["hidden_channels"]
        params["mu"]["edge_dim"] = params["common"]["edge_dim"]
        params["logstd"]["edge_dim"] = params["common"]["edge_dim"]

        mu_encoder = get_encoder(params["mu"])
        logstd_encoder = get_encoder(params["logstd"])

        return NodeVAEEncoder(common_encoder, mu_encoder, logstd_encoder)
    if params["layer_type"] == "GCN":
        return GCNEncoder(**params)
    if params["layer_type"] == "GAT":
        return GATEncoder(**params)
    if params["layer_type"] == "GDN":
        return GDNEncoder(**params)
    if params["layer_type"] == "GIN":
        return GINEncoder(**params)
    if params["layer_type"] == "GGNN":
        return GGNNEncoder(**params)

    raise ValueError(f"Could not construct encoder for {params}")
