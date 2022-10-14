import numpy as np


class ObservationParser(object):
    def parse(self, observations):
        entities = np.zeros((512,16), dtype="float") # 512 entities with 16 features
        available = np.zeros((512,8), dtype="int8") # is the entity attack-able for each agent
        order_in_obs = np.zeros((512,8), dtype="int8") # global id to local id
        group_id = np.zeros((512,1), dtype="int8") # group id
        global_id = np.zeros((512,1), dtype="int8") # global id
        tiles = np.zeros((1024,1024,100), dtype="float") # map

        pos_dict = {}

        cnt = 0

        if "stat" in observations:
            observations.pop("stat")
        for idx, obs in observations.items():
            for i in range(obs['Entity']['N'][0]):
                ent = obs['Entity']['Continuous'][i]
                pos = int(ent[1]-1)
                if pos not in pos_dict:
                    pos_dict[pos] = cnt
                    entities[cnt] = self._entity_parse(ent)
                    group_id[cnt] = int(ent[6])
                    global_id[cnt] = pos
                    cnt += 1
                available[pos_dict[pos]][idx] = 1
                order_in_obs[pos_dict[pos]][idx] = i

        return entities, available, order_in_obs, group_id, global_id, cnt, tiles

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
        for i in [4,5,6,7]:
            ret[i] = ent[i+8] / 100.0

        # 8-15 8 skills (16-23)
        for i in [8,9,10,11,12,13,14,15]:
            ret[i] = ent[i+8] / 10.0

        return ret

if __name__ == "__main__":
    parser = ObservationParser()