import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class AttackHead(nn.Module):
    '''
        Inputs:
            lstm_output     (batch_size, lstm_dim),
            entity_embedd   (batch_size, seq_length, en_embed_dim),
            self_embedd     (batch_size, 8, en_embed_dim) 
            available       (batch_size, 8, seq_length)
        Outputs:
            output_act      (batch_size, 8, seq_length * 3)
            output_problog  (batch_size, 8, seq_length * 3)
        '''
    def __init__(self, lstm_dim = 256, embed_dim = 256, en_embed_dim = 256, action_space = 3):
        super().__init__()
        self.fc_lstm = nn.Linear(lstm_dim, embed_dim)
        self.fc_en = nn.Linear(en_embed_dim, embed_dim)
        self.fc_self = nn.Linear(en_embed_dim, embed_dim)
        self.fc1 = nn.Linear(embed_dim*3, embed_dim)
        self.fc2 = nn.Linear(embed_dim, action_space)


    def forward(self, lstm_output, entity_embedd, self_embedd, available, mode='eval'):
        lstm_embed = self.fc_lstm(lstm_output).relu() # (batch_size, embed_dim)
        lstm_extended = lstm_embed.unsqueeze(axis=1).unsqueeze(axis=1).repeat(1, 8, entity_embedd.shape[1], 1) # (batch_size, 8, 512, embed_dim)
        en_embed = self.fc_en(entity_embedd).relu() # (batch_size, seq_length, embed_dim)
        en_extended = en_embed.unsqueeze(axis=1).repeat(1,8,1,1) # (batch_size, 8, 512, embed_dim)
        se_embed = self.fc_self(self_embedd).relu() # (batch_size, 8, embed_dim)
        se_extended = se_embed.unsqueeze(axis=2).repeat(1, 1, entity_embedd.shape[1], 1) # (batch_size, 8, 512, embed_dim)
        input_tensor = torch.cat([lstm_extended, en_extended, se_extended], dim=-1) # (batch_size, 8, seq_length, embed_dim*3)
        hidd = self.fc1(input_tensor).relu() # (batch_size, 8, seq_length, embed_dim)
        output = self.fc2(hidd).squeeze(-1) # (batch_size, 8, seq_length, action_space)
        available = available.unsqueeze(axis=-1).repeat(1,1,1,3) # (batch_size, 8, seq_length, action_space)
        output_masked = output.masked_fill_(~available, -1e9)
        output_masked = output_masked.view(output_masked.shape[0], output_masked.shape[1], -1)
        output_prob = output_masked.softmax(dim=-1) # (batch_size, seq_length, seq_length * action_space)

        if mode == 'eval':
            output_act = output.argmax(-1) # (batch_size, seq_length)
        elif mode =='train':
            dist = Categorical(output_prob)
            output_act = dist.sample() # (batch_size, seq_length)
        else:
            print("UNKNOWN mode in AttackHead.forward")
            assert(0)

        return output_act, output_prob


        

        