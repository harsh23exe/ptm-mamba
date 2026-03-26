import torch
import torch.nn as nn

from ptm_classification.models.cnn_seq_models import CNNGRUClassifier


class PTMConditionedCNNGRU(CNNGRUClassifier):
    """
    CNN + GRU backbone with a learned PTM-type embedding concatenated before
    the classification head (class-conditioned logits).

    Forward: forward(x, ptm_ids) where ptm_ids are long tensors of shape [B].
    """

    def __init__(
        self,
        num_ptm_types: int,
        ptm_embed_dim: int = 32,
        num_classes: int = 2,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        kwargs.pop("num_classes", None)
        super().__init__(num_classes=num_classes, dropout=dropout, **kwargs)
        if num_ptm_types < 1:
            raise ValueError("num_ptm_types must be >= 1")

        self.num_ptm_types = num_ptm_types
        self.ptm_embed_dim = ptm_embed_dim
        self.ptm_emb = nn.Embedding(num_ptm_types, ptm_embed_dim)

        rnn_hidden = self.gru.hidden_size
        self.head = nn.Sequential(
            nn.Linear(rnn_hidden + ptm_embed_dim, rnn_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor, ptm_ids: torch.Tensor) -> torch.Tensor:
        cnn_out = self.forward_cnn(x)
        rnn_in = cnn_out.transpose(1, 2)
        rnn_out, _ = self.gru(rnn_in)
        rnn_out_t = rnn_out.transpose(1, 2)
        pooled = self.pool(rnn_out_t).squeeze(-1)
        pooled = self.rnn_ln(pooled)

        pe = self.ptm_emb(ptm_ids.long())
        h = torch.cat([pooled, pe], dim=-1)
        return self.head(h)
