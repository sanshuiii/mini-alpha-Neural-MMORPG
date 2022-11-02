if __name__ == '__main__':
    import torch
    import numpy as np
    import nmmo
    from nmmo import config
    from neurips2022nmmo import CompetitionConfig, TeamBasedEnv
    from nmmo.io import action

    import yaml
    from easydict import EasyDict

    from model.Arch import Archtecture

    with open("/home/jiayou.zhang/hom/mmo/neurips2022-nmmo-starter-kit/my-submission/NeurIPS22NMMO/ppo/scripts/config/config.yml", 'r') as f:
        arg_file = yaml.full_load(f)
    args = EasyDict(arg_file)

    class TrainConfig(CompetitionConfig):
        MAP_N = 2

    env = TeamBasedEnv(TrainConfig())
    obs = env.reset()

    for _ in range(5):
        obs,reward,done,info = env.step({})

    from utils.envParser import ObservationParser
    myteam=args.self_group_id

    parser = ObservationParser()
    entities1, available1, order_in_obs1, group_id1, global_id1, cnt1, tiles1, move_possible1  = parser.parse(observations=obs[myteam], args=args)

    obs,reward,done,info = env.step({})
    entities2, available2, order_in_obs2, group_id2, global_id2, cnt2, tiles2, move_possible2  = parser.parse(observations=obs[myteam], args=args)

    entities = torch.Tensor([entities1, entities2])
    tiles = [torch.Tensor(tiles1), torch.Tensor(tiles2)]
    cnt = torch.tensor([cnt1, cnt2], dtype=int)
    available = torch.tensor([available1, available2], dtype=bool).transpose(1,2)
    group_id = torch.tensor([group_id1, group_id2], dtype=int)
    global_id = torch.tensor([global_id1, global_id2], dtype=int)
    move_possible = torch.tensor([move_possible1, move_possible2], dtype=bool)

    # print(entities.shape)
    # print(tiles[0].shape)
    # print(tiles[1].shape)
    # print(cnt.shape)
    # print(available.shape)
    # print(group_id.shape)
    # print(global_id.shape)

    arch = Archtecture(args)
    lstm_hidden = arch.core.init_hidden_state(batch_size=2)
    lstm_hidd, move_act, attack_act, move_prob, attack_prob = arch(tiles, group_id, global_id, entities, cnt, lstm_hidden, move_possible, available)

# import torch
# a = torch.zeros((3,3), dtype=float)
# b = torch.tensor([[1,2,3],[4,5,6]], dtype=float)
# id1 = torch.tensor([1,2], dtype=int)
# id2 = torch.tensor([0,1], dtype=int)
# a[id1] = b[id2]
# print(a)