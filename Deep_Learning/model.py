import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# -----------------------
# Attention Layer
# -----------------------
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, outputs):
        weights = torch.softmax(self.attn(outputs), dim=1)
        context = torch.sum(outputs * weights, dim=1)
        return context, weights


# -----------------------
# Multimodal ICU Model
# -----------------------
class MultimodalICUModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=16,
            hidden_size=64,
            batch_first=True,
            bidirectional=True
        )

        # MUST stay attn (not attention)
        self.attn = Attention(128)

        self.static_net = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(144, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1)
        )

    def forward(self, temporal, static, seq_len):

        packed = pack_padded_sequence(
            temporal,
            seq_len.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_out, _ = self.lstm(packed)

        outputs, _ = pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=72
        )

        # FIXED LINE
        temporal_embedding, attention = self.attn(outputs)

        static_embedding = self.static_net(static)

        fused = torch.cat([temporal_embedding, static_embedding], dim=1)

        prediction = self.classifier(fused).squeeze(1)

        return prediction, attention