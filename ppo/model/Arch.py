from tracemalloc import Statistic
from .AttackHead import AttackHead
from .MoveHead import MoveHead
from .EntityEncoder import EntityEncoder
from .SpatialEncoder import SpatialEncoder
from .Core import Core

import torch
import torch.nn as nn
import torch.nn.functional as F

class Archtecture(nn.Module):
    '''
    input:
        tiles:          (number of sample I's alive agents[which is a list], input channel, h, w)
        group_id:       (batch_size, seq_size)
        entities:       (batch_size, seq_size, feature_size)
        cnt:            (batch_size)
    output:
    '''
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.spatialEncoder = SpatialEncoder(args)
        self.entityEncoder = EntityEncoder()
        self.core = Core()
        self.attackHead = AttackHead()
        self.moveHead = MoveHead()

    def forward(self, tiles, group_id, global_id, entities, cnt, lstm_lst_hidd, move_possible, available, mode='eval'):
        sp_embed = self.spatialEncoder(tiles, group_id)
        # print(sp_embed.shape) # batch_size, seq_length, feature_size
        en_embed = self.entityEncoder(entities, group_id, sp_embed, cnt)
        # print(en_embed.shape) # batch_size, seq_length, feature_size
        en_embed_avg = en_embed.sum(axis = 1)/(cnt.view(-1,1))
        # print(en_embed_avg.shape) # batch_size, feature_size
        lstm_output, lstm_hidd = self.core(en_embed_avg, lstm_lst_hidd)
        # print(lstm_output.shape) # batch_size, lstm_size
        # print(lstm_hidd[0].shape) # n_layer, batch_size, hidden_size
        move_act, move_prob = self.moveHead(lstm_output, sp_embed, move_possible, mode=mode)
        # print(act.shape) # batch_size, seq_length
        # print(problog.shape) # batch_size, seq_length, action_space
        self_en_embed = self.get_self_embed(en_embed, group_id, global_id)
        attack_act, attack_prob = self.attackHead(lstm_output, en_embed, self_en_embed, available)
        # print(attack_act.shape) # batch_size, seq_length
        # print(attack_prob.shape) # batch_size, seq_length

        return lstm_hidd, move_act, attack_act, move_prob, attack_prob

    def get_self_embed(self, en_embed, group_id, global_id):
        ret = torch.zeros((en_embed.shape[0], 8, 256))
        for i in range(en_embed.shape[0]):
            idx = (group_id[i]==self.args.self_group_id).nonzero().view(-1)
            offset = global_id[i,idx]%8
            ret[i,offset] = en_embed[i,idx]
        return ret
        