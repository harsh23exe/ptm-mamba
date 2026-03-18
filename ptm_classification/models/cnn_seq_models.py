import torch
import torch.nn as nn


class _BaseSeqClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        seq_len: int = 51,
        conv_channels: int = 256,
        kernel_size: int = 3,
        conv_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len

        self.input_ln = nn.LayerNorm(input_dim)

        layers = []
        in_channels = input_dim
        padding = kernel_size // 2
        for _ in range(conv_layers):
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=conv_channels,
                kernel_size=kernel_size,
                padding=padding,
            )
            layers.append(conv)
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_channels = conv_channels

        self.conv = nn.Sequential(*layers)
        self.conv_out_channels = conv_channels

    def forward_cnn(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, input_dim]
        returns: [batch, C, T] after conv stack.
        """
        x = self.input_ln(x)
        x = x.transpose(1, 2)  # [B, input_dim, T]
        x = self.conv(x)       # [B, C, T]
        return x


class CNNGRUClassifier(_BaseSeqClassifier):
    def __init__(
        self,
        input_dim: int = 768,
        seq_len: int = 51,
        conv_channels: int = 256,
        kernel_size: int = 3,
        conv_layers: int = 2,
        rnn_hidden: int = 256,
        rnn_layers: int = 1,
        num_classes: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            seq_len=seq_len,
            conv_channels=conv_channels,
            kernel_size=kernel_size,
            conv_layers=conv_layers,
            dropout=dropout,
        )

        self.gru = nn.GRU(
            input_size=self.conv_out_channels,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=False,
        )

        self.rnn_ln = nn.LayerNorm(rnn_hidden)
        self.pool = nn.AdaptiveMaxPool1d(1)

        self.head = nn.Sequential(
            nn.Linear(rnn_hidden, rnn_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, input_dim]
        """
        cnn_out = self.forward_cnn(x)          # [B, C, T]
        rnn_in = cnn_out.transpose(1, 2)       # [B, T, C]

        rnn_out, _ = self.gru(rnn_in)          # [B, T, H]

        # max-pool over temporal dimension (T)
        rnn_out_t = rnn_out.transpose(1, 2)    # [B, H, T]
        pooled = self.pool(rnn_out_t).squeeze(-1)  # [B, H]

        pooled = self.rnn_ln(pooled)
        logits = self.head(pooled)
        return logits


class CNNBiLSTMClassifier(_BaseSeqClassifier):
    def __init__(
        self,
        input_dim: int = 768,
        seq_len: int = 51,
        conv_channels: int = 256,
        kernel_size: int = 3,
        conv_layers: int = 2,
        rnn_hidden: int = 256,
        rnn_layers: int = 1,
        num_classes: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            seq_len=seq_len,
            conv_channels=conv_channels,
            kernel_size=kernel_size,
            conv_layers=conv_layers,
            dropout=dropout,
        )

        self.lstm = nn.LSTM(
            input_size=self.conv_out_channels,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.out_dim = 2 * rnn_hidden
        self.rnn_ln = nn.LayerNorm(self.out_dim)
        self.pool = nn.AdaptiveMaxPool1d(1)

        self.head = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.out_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, input_dim]
        """
        cnn_out = self.forward_cnn(x)          # [B, C, T]
        rnn_in = cnn_out.transpose(1, 2)       # [B, T, C]

        rnn_out, _ = self.lstm(rnn_in)         # [B, T, 2H]

        rnn_out_t = rnn_out.transpose(1, 2)    # [B, 2H, T]
        pooled = self.pool(rnn_out_t).squeeze(-1)  # [B, 2H]

        pooled = self.rnn_ln(pooled)
        logits = self.head(pooled)
        return logits


