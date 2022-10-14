import numpy as np


class ObservationParser(object):
    def parse(self, observations):
        entities = np.zeros((512,25), dtype="float") # 512 entities with 25 features
        available = np.zeros((8,512), dtype="int8") # is the entity attack-able for each agent
        order_in_obs = np.zeros((8,512), dtype="int8") # global id to local id
        tiles = np.zeros((1024,1024,100), dtype="float") # map

        pos_dict = {}

        cnt = 0

        if "stat" in observations:
            observations.pop("stat")
        for idx, obs in observations.items():
            for i in range(obs['Entity']['N']):
                ent = obs['Entity']['Continuous'][i]
                pos = ent[1]
                if pos not in pos_dict:
                    pos_dict[pos] = cnt
                    entities[cnt] = self._entity_parse(ent)
                    cnt += 1
                available[idx-1][pos_dict[pos]] = 1
                order_in_obs[idx-1][pos_dict[pos]] = i

        return entities, available, order_in_obs, tiles

    @staticmethod
    def _entity_parse(ent):
        ret = np.zeros(25, dtype=float)

        # 0-16 population (6)
        pop = max(ent[6], 0)
        ret[pop] = 1

        # 17,18 r,c (7,8)
        ret[17] = ent[7] / 1024.0
        ret[18] = ent[8] / 1024.0

        # 19,20 level,item_level (3,4)
        ret[19] = ent[3] / 10.0
        ret[20] = ent[4] / 10.0

        # 21-24 Gold, Health, Food, Water (12-15)
        for i in [21,22,23,24]:
            ret[i] = ent[i-9] / 100.0

        # 25-32 8 skills (16-23)
        for i in [25,26,27,28,29,30,31,32]:
            ret[i] = ent[i-9] / 10.0

        return ret

if __name__ == "__main__":
    parser = ObservationParser()