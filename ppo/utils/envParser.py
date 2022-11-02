import numpy as np


class ObservationParser(object):
    def parse(self, observations, args):
        entities = np.zeros((512, 16), dtype="float")  # 512 entities with 16 features
        available = np.zeros((512, 8), dtype="int")  # is the entity attack-able for each agent (if L_inf distance <= 3)
        order_in_obs = np.zeros((512, 8), dtype="int")  # global id to local id
        group_id = np.zeros(512, dtype="int")  # group id
        global_id = np.zeros(512, dtype="int")  # global id
        tiles = []  # np.zeros((8,1024,100), dtype="float") # map

        move_possible = np.zeros((512,5), dtype="float") 
        ###
        # show if direction is available (only self_group will be possible)
        # 0: i-1
        # 1: i+1
        # 2: j+1
        # 3: j-1
        # 4: stay still
        ###

        glob_id2offset = {}  # global id to offset in seq_len
        offset2id_in_team = {}

        cnt = 0

        if "stat" in observations:
            observations.pop("stat")
        for idx, obs in observations.items():
            cur_i = obs['Entity']['Continuous'][0][7]
            cur_j = obs['Entity']['Continuous'][0][8]
            for i in range(obs['Entity']['N'][0]):
                ent = obs['Entity']['Continuous'][i]
                glob_id = int(ent[1] - 1)  # global id
                if glob_id not in glob_id2offset:
                    glob_id2offset[glob_id] = cnt
                    offset2id_in_team[cnt] = glob_id%8
                    entities[cnt] = self._entity_parse(ent)
                    group_id[cnt] = int(ent[6])  # -1 ~ 15
                    group_id[cnt] = (group_id[cnt] + 17) % 17  # move -1 to 16
                    global_id[cnt] = glob_id  # -inf ~ 127, not used
                    cnt += 1

                dis = max(abs(ent[7] - cur_i), abs(ent[8] - cur_j))
                attackable = (group_id[glob_id2offset[glob_id]] != args.self_group_id and dis <= 3)

                available[glob_id2offset[glob_id]][idx] = attackable
                order_in_obs[glob_id2offset[glob_id]][idx] = i

        for offset in range(cnt):
            if group_id[offset] == args.self_group_id:
                tmp = np.zeros((8, 15, 15))
                for r in range(15):
                    for c in range(15):
                        t = int(observations[offset2id_in_team[offset]]['Tile']['Continuous'][15 * r + c][1]) # type of tile
                        nent = int(observations[offset2id_in_team[offset]]['Tile']['Continuous'][15 * r + c][0]) # if someone stands here
                        tmp[0,r,c] = (t not in {0,1,5,14,15}) and (nent == 0) # passable
                        tmp[1,r,c] = (t == 1) # is water
                        tmp[2,r,c] = (t == 4) # is forest
                        tmp[3,r,c] = (t == 7) # is ore
                        tmp[4,r,c] = (t == 9) # is tree
                        tmp[5,r,c] = (t == 11) # is Crystal
                        tmp[6,r,c] = (t == 13) # is Herb
                        tmp[7,r,c] = (t == 15) # is fish
                move_possible[offset][0] = tmp[0,6,7]
                move_possible[offset][1] = tmp[0,8,7]
                move_possible[offset][2] = tmp[0,7,8]
                move_possible[offset][3] = tmp[0,7,6]
                move_possible[offset][4] = 1
                tiles.append(tmp)
        tiles = np.array(tiles)

        return entities, available, order_in_obs, group_id, global_id, cnt, tiles, move_possible

    @staticmethod
    def _entity_parse(ent):
        ret = np.zeros(16, dtype=float)

        # r, c (7,8)
        ret[0] = ent[7] / 1024.0
        ret[1] = ent[8] / 1024.0

        # level, item_level (3,4)
        ret[2] = ent[3] / 10.0
        ret[3] = ent[4] / 10.0

        # 4-7 Gold, Health, Food, Water (12-15)
        for i in [4, 5, 6, 7]:
            ret[i] = ent[i + 8] / 100.0

        # 8-15 8 skills (16-23)
        for i in [8, 9, 10, 11, 12, 13, 14, 15]:
            ret[i] = ent[i + 8] / 10.0

        return ret


if __name__ == "__main__":
    parser = ObservationParser()
