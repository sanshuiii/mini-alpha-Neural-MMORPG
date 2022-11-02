import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class EntityEncoder(nn.Module):
    def __init__(self, input_size=16, drop=0.0, d_model=256, d_feedforward=1024, n_head=2, n_layers=3, max_seq_length=512):
        super().__init__()

        self.max_seq_length = max_seq_length

        self.d_model = d_model
        self.dropout = nn.Dropout(drop)
        self.embedd = nn.Linear(input_size, d_model)
        self.group_id_embedd = nn.Embedding(num_embeddings=17, embedding_dim=d_model)
        self.group_id_embedd.weight.data.normal_(0, 1e-3)

        # input = [batch, seq=number_of_agents, feature=d_model]
        # output = [batch, seq=number_of_agents, feature=d_model]
        encoder_layer = TransformerEncoderLayer(d_model=d_model,
                                                nhead=n_head,
                                                dim_feedforward=d_feedforward,
                                                dropout=drop,
                                                layer_norm_eps=1e-6,
                                                batch_first=True)
        self.encoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_layers)

        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc1 = nn.Linear(d_model, d_model)

    def forward(self, x, group_id, spatial_embedd, entity_num):
        """
            x:           (batch_size, seq_size, feature_size)
            group_id:    (batch_size, seq_size)
            spatial_embedd: (batch_size, seq_size, d_model)
            entity_num:  (batch_size)
        """
        batch_size, seq_size, feature_size = x.shape

        mask = torch.arange(0, self.max_seq_length).float()
        mask = mask < entity_num.view(batch_size, 1) # mask.shape = (batch_size, seq_size)

        x = self.embedd(x)  # x.shape = (batch_size, seq_size, d_model)
        group_id = self.group_id_embedd(group_id)
        x = x + group_id + spatial_embedd
        out = self.encoder(x, src_key_padding_mask=mask) # x.shape = (batch_size, seq_size, d_model)
        entity_embeddings = F.relu(self.conv1(F.relu(out).transpose(1, 2))).transpose(1, 2)
        entity_embeddings = entity_embeddings * mask.unsqueeze(-1) # mask the output for padding tokens
        return entity_embeddings



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = EntityEncoder()
    encoder.to(device)
