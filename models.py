"""
LSTM과 Transformer를 파라미터 수가 거의 같도록(oo 0.5% 이내) 구성.
LSTM: input=7, hidden=256, layers=2, dropout=0.1
Transformer: d_model=256, heads=8, encoder layers=2, dropout=0.1
"""
import math
import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    def __init__(self, n_features=7, hidden=256, layers=2, dropout=0.1, horizon=24):
        super().__init__()
        self.horizon = horizon
        self.n_features = n_features
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden, num_layers=layers,
            batch_first=True, dropout=dropout if layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden, horizon * n_features)

    def forward(self, x):
        # x: (B, L, C)
        out, _ = self.lstm(x)
        last = out[:, -1, :]  # (B, hidden)
        pred = self.fc(last)  # (B, horizon*C)
        return pred.view(-1, self.horizon, self.n_features)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerForecaster(nn.Module):
    def __init__(self, n_features=7, d_model=256, nhead=8, num_layers=2,
                 dim_ff=256, dropout=0.1, horizon=24):
        super().__init__()
        self.horizon = horizon
        self.n_features = n_features
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, horizon * n_features)
        self._last_attn = None  # 해석가능성 실험용 hook

    def forward(self, x, return_attn=False):
        h = self.input_proj(x)
        h = self.pos_enc(h)
        if return_attn:
            # 마지막 레이어의 self-attention weight를 직접 뽑기 위해 수동 forward
            attn_weights = None
            out = h
            for i, layer in enumerate(self.encoder.layers):
                if i == len(self.encoder.layers) - 1:
                    out2, attn_weights = layer.self_attn(
                        out, out, out, need_weights=True, average_attn_weights=True
                    )
                    out = layer.norm1(out + layer.dropout1(out2))
                    ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(out))))
                    out = layer.norm2(out + layer.dropout2(ff))
                else:
                    out = layer(out)
            h = out
            self._last_attn = attn_weights  # (B, L, L)
        else:
            h = self.encoder(h)
        last = h[:, -1, :]
        pred = self.fc(last)
        return pred.view(-1, self.horizon, self.n_features)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    lstm = LSTMForecaster()
    tf = TransformerForecaster()
    print("LSTM params:", count_params(lstm))
    print("Transformer params:", count_params(tf))
