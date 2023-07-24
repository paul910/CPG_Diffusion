import os
import sys

from torch_geometric.data import Data
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import tokenizers
import torch.nn.functional as F

from cpg_reconstruction.dataloader import CPGReconstructionDataset
from cpg_reconstruction.model import Transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = tokenizers.Tokenizer.from_pretrained("Salesforce/codegen-350M-multi")
tokenizer.add_special_tokens(["<|startoftext|>", "<|pad|>"])

PAD_TOK = tokenizer.token_to_id("<|pad|>")
END_TOK = tokenizer.token_to_id("<|endoftext|>")

model = Transformer(num_tokens=tokenizer.get_vocab_size(),
                    dim_model=128,
                    num_heads=2,
                    num_encoder_layers=6,
                    num_decoder_layers=6,
                    dropout_p=0.1,
                    padding_token=PAD_TOK).to(device)

model.load_state_dict(torch.load("cpg_reconstruction/transformer_model.chkpt", map_location=device))
model.eval()

load_set = CPGReconstructionDataset(None, tokenizer)


def decode_tokens(tokens):
    return tokenizer.decode(tokens, skip_special_tokens=False)


@torch.no_grad()
def beam_decode(src, seq_len, first_token, end_token=END_TOK, pad_token=PAD_TOK, beam_width=5):
    tgt = first_token.unsqueeze(0).unsqueeze(0)
    alpha = 1.0

    pred = model.pred_nodes(src, tgt, model.get_tgt_mask(1))  # TODO
    pred = F.log_softmax(pred.squeeze(), dim=-1)
    pred = torch.topk(pred, k=beam_width)
    tgts = pred.indices.reshape(beam_width, 1)
    probs = pred.values
    tgt = torch.cat([tgt.repeat(beam_width, 1), tgts], axis=1)

    padding = torch.empty_like(tgt[:, 0:1])
    padding.fill_(pad_token)

    src = torch.repeat_interleave(src, beam_width, 0)
    for i in tqdm(range(2, seq_len + 1), desc="Reconstruction"):
        pred = model.pred_nodes(src, tgt, model.get_tgt_mask(i))  # TODO

        probs_local = torch.repeat_interleave(probs, beam_width)
        tgt = torch.cat([tgt, padding], axis=1)
        tgts_local = torch.repeat_interleave(tgt, beam_width, 0)
        for j, seq in enumerate(pred):
            seq = seq[-1]
            seq = F.log_softmax(seq, dim=-1)
            preds = torch.topk(seq, k=beam_width)
            tgts_local[j * beam_width:((j + 1) * beam_width), -1] = preds.indices
            probs_local[j * beam_width:((j + 1) * beam_width)] += preds.values
            if tgt[j, -2] == end_token or tgt[j, -2] == pad_token:
                tgts_local[(j + 1) * beam_width - 1, -1] = pad_token
                probs_local[(j + 1) * beam_width - 1] -= preds.values[-1]

        # length penalty
        ends = torch.cat([(t == end_token).unsqueeze(0) for t in tgts_local[:, -1]], dim=0)
        pads = torch.cat([(t == pad_token).unsqueeze(0) for t in tgts_local[:, -1]], dim=0)
        probs_local[torch.logical_not(pads)] /= (5 + i) ** alpha / (5 + 1) ** alpha

        # after end filtering
        after_ends = torch.cat([(t == end_token).unsqueeze(0) for t in tgts_local[:, -2]], dim=0)
        after_pads = torch.cat([(t == pad_token).unsqueeze(0) for t in tgts_local[:, -2]], dim=0)
        after_mask = torch.logical_and(
            torch.logical_or(after_ends, after_pads),
            torch.logical_not(pads)
        )
        probs_local.masked_fill_(after_mask, -1e20)

        # probs, indices = torch.topk(probs_local, k=beam_width)
        _, indices = torch.topk(probs_local, k=beam_width)

        # remove length penalty
        probs_local[torch.logical_not(torch.logical_or(pads, ends))] *= (5 + i) ** alpha / (5 + 1) ** alpha

        probs = probs_local[indices]
        tgt = tgts_local[indices, :]
        text = decode_tokens(tgt[0].detach().cpu().numpy())
        if tgt[0, -1] == end_token:
            break
    text = decode_tokens(tgt[0].detach().cpu().numpy())
    return text


def process(graph: Data):
    X = graph.x[..., 50:].unsqueeze(0).to(device)
    y = load_set.START_TOK
    y = y.to(device)
    sequence_length = 256
    beam_width = 5
    return beam_decode(X, sequence_length, y[0], beam_width=beam_width)
