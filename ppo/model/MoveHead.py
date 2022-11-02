import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class MoveHead(nn.Module):
    '''
        Inputs:
            lstm_output     (batch_size, lstm_dim),
            spatial_embedd  (batch_size, seq_length, sp_embed_dim),
            move_possible   (batch_size, seq_length, action_space=5)
        Outputs:
            output_act      (batch_size, seq_length)
            output_problog  (batch_size, seq_length)
        '''
    def __init__(self, lstm_dim = 256, embed_dim = 256, sp_embed_dim = 256, action_space = 5):
        super().__init__()
        self.fc_lstm = nn.Linear(lstm_dim, embed_dim)
        self.fc_sp = nn.Linear(sp_embed_dim, embed_dim)
        self.fc1 = nn.Linear(embed_dim*2, embed_dim)
        self.fc2 = nn.Linear(embed_dim, action_space)
        

    def forward(self, lstm_output, spatial_embedd, move_possible, mode='eval'):
        lstm_embed = self.fc_lstm(lstm_output).relu() # (batch_size, embed_dim)
        lstm_extended = lstm_embed.unsqueeze(axis=1).repeat(1, spatial_embedd.shape[1], 1) # (batch_size, seq_length, embed_dim)
        sp_embed = self.fc_sp(spatial_embedd).relu() # (batch_size, seq_length, embed_dim)
        input_tensor = torch.cat([lstm_extended,sp_embed], dim=-1) # (batch_size, seq_length, embed_dim*2)
        hidd = self.fc1(input_tensor).relu() # (batch_size, seq_length, embed_dim)
        output = self.fc2(hidd) # (batch_size, seq_length, action_space)
        output_masked = output.masked_fill_(~move_possible, -1e9)
        output_prob = output_masked.softmax(dim=-1) # (batch_size, seq_length, action_space)

        if mode == 'eval':
            output_act = output.argmax(-1) # (batch_size, seq_length)
        elif mode =='train':
            dist = Categorical(output_prob)
            output_act = dist.sample() # (batch_size, seq_length)
        else:
            print("UNKNOWN mode in MoveHead.forward")
            assert(0)

        return output_act, output_prob


        

        