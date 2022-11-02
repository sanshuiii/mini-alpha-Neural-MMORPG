import lzma
import json
import os

from copy import deepcopy
import nmmo
import numpy as np
from neurips2022nmmo import CompetitionConfig
import multiprocessing as mp

item_name_to_index = {
    "Gold": 1,
    "Hat": 2,
    "Top": 3,
    "Bottom": 4,
    "Sword": 5,
    "Bow": 6,
    "Wand": 7,
    "Rod": 8,
    "Gloves": 9,
    "Pickaxe": 10,
    "Chisel": 11,
    "Arcane": 12,
    "Scrap": 13,
    "Shaving": 14,
    "Shard": 15,
    "Ration": 16,
    "Poultice": 17,
}
item_value_to_name = {
    1:"Gold",
    2:"Hat",
    3:"Top",
    4:"Bottom",
    5:"Sword",
    6:"Bow",
    7:"Wand",
    8:"Rod",
    9:"Gloves",
    10:"Pickaxe",
    11:"Chisel",
    12:"Arcane",
    13:"Scrap",
    14:"Shaving",
    15:"Shard",
    16:"Ration",
    17:"Poultice",
}
# Get entities (including player_id) surrounding player_id
def get_sur_entities(player_id: int, player_pkt: dict, npc_pkt: dict):
    r = player_pkt[str(player_id)]["base"]["r"]
    c = player_pkt[str(player_id)]["base"]["c"]
    sur_entities = []
    pos_map = {}
    for other_playerid in player_pkt:
        info_other = player_pkt[other_playerid]
        other_playerid = int(other_playerid)
        if other_playerid == player_id:
            continue
        r_other = info_other["base"]["r"]
        c_other = info_other["base"]["c"]
        if abs(r-r_other) <= 7 and abs(c-c_other) <= 7:
            sur_entities.append(other_playerid)
            pos_key = f"{r_other}_{c_other}"
            pos_map[pos_key] = other_playerid
    
    for npc_id in npc_pkt:
        info_npc = npc_pkt[npc_id]
        npc_id = int(npc_id)
        r_npc = info_npc["base"]["r"]
        c_npc = info_npc["base"]["c"]
        if abs(r-r_npc )<= 7 and abs(c-c_npc) <= 7:
            sur_entities.append(npc_id)
            pos_key = f"{r_npc}_{c_npc}"
            pos_map[pos_key] = npc_id
    
    # Sort sur entities
    sorted_sur_entities =[]
    r_topleft = r-7
    c_topleft = c-7
    for row_idx in range(15):
        for col_idx in range(15):
            r_tile = r_topleft+row_idx
            c_tile = c_topleft+col_idx
            pos_key = f"{r_tile}_{c_tile}"
            if pos_key in pos_map:
                agent_id = pos_map[pos_key]
                sorted_sur_entities.append(agent_id)
    assert set(sorted_sur_entities) == set(sur_entities)
    sorted_sur_entities.insert(0, player_id)
    
    return sorted_sur_entities
def extract_obs_of_packet(step: int, 
                          initial_map: list[list],
                          cur_packet: dict, 
                          next_packet: dict=None,
                          last_packet: dict=None) -> dict[int, dict[str, dict]]:
    border_pkt = cur_packet["border"]
    size_pkt = cur_packet["size"]
    player_pkt = cur_packet["player"]
    npc_pkt = cur_packet["npc"]
    market_pkt = cur_packet["market"]
    pos_pkt = cur_packet["pos"]
    wilderness_pkt = cur_packet["wilderness"]

    # Update map
    cur_map = deepcopy(initial_map)
    next_resource = None
    if next_packet:
        next_resource = next_packet["resource"]
    for pos in cur_packet["resource"]:
        if next_resource and pos not in next_resource:
            continue
        r = pos[0]
        c = pos[1]
        cur_map[r][c] = cur_map[r][c]-1

    # Cal attacker
    beattacked_map = {}
    tmp_packet = cur_packet
    if tmp_packet:
        for pid in tmp_packet["player"]:
            if "attack" in tmp_packet["player"][pid]["history"]:
                attack_target = tmp_packet["player"][pid]["history"]["attack"]["target"]
                beattacked_map[int(attack_target)] = int(pid)
        for npc_id in tmp_packet["npc"]:
            if "attack" in tmp_packet["npc"][npc_id]["history"]:
                attack_target = tmp_packet["npc"][npc_id]["history"]["attack"]["target"]
                beattacked_map[int(attack_target)] = int(npc_id)

    # Remove dead agents in current packet
    dead_agents = set()
    if next_packet:
        next_pids = list(next_packet["player"].keys())
        cur_pids = list(player_pkt.keys())
        for pid in cur_pids:
            if pid not in next_pids:
                player_pkt.pop(pid, None)
                dead_agents.add(int(pid))
        next_npcs = list(next_packet["npc"].keys())
        cur_npcs = list(npc_pkt.keys())
        for npc_id in cur_npcs:
            if npc_id not in next_npcs:
                npc_pkt.pop(npc_id, None)
                dead_agents.add(int(npc_id))

    # Calculate NEnts on tiles
    nents_tile = np.zeros((len(cur_map), len(cur_map[0])))
    for player_id in player_pkt:
        r = player_pkt[player_id]["base"]["r"]
        c = player_pkt[player_id]["base"]["c"]
        nents_tile[r][c] += 1
    for npc_id in npc_pkt:
        r = npc_pkt[npc_id]["base"]["r"]
        c = npc_pkt[npc_id]["base"]["c"]
        nents_tile[r][c] += 1

    # Create obs
    obs = {
        int(player_id): {
            "Entity": {
                "Continuous": np.zeros((100,24), dtype=np.float32),
                # "Discrete": np.zeros((100,5), dtype=np.int32),
                "N": np.array([0], dtype=np.int32)
            },
            "Tile": {
                "Continuous": np.zeros((225,4), dtype=np.float32),
                # "Discrete": np.zeros((225,3), dtype=np.int32),
                "N": np.array([0], dtype=np.int32)
            },
            "Item": {
                "Continuous": np.zeros((170,16), dtype=np.float32),
                # "Discrete": np.zeros((170,3), dtype=np.int32),
                "N": np.array([0], dtype=np.int32)
            },
            "Market": {
                "Continuous": np.zeros((170,16), dtype=np.float32),
                # "Discrete": np.zeros((170,3), dtype=np.int32),
                "N": np.array([0], dtype=np.int32)
            },
            "step": step
        }
        for player_id in player_pkt
    }

    for player_id in player_pkt:         
        r = player_pkt[player_id]["base"]["r"]
        c = player_pkt[player_id]["base"]["c"]
        player_id = int(player_id)
        
        # Get entities (including player_id) surrounding player_id
        sur_entities = get_sur_entities(player_id, player_pkt, npc_pkt)

        # fill entity data
        obs[player_id]["Entity"]["N"][0] = len(sur_entities)
        obs_entity_arr = obs[player_id]["Entity"]["Continuous"]
        for idx, entId in enumerate(sur_entities):
            if entId>0:
                ent_info = player_pkt[str(entId)]
            else:
                ent_info = npc_pkt[str(entId)]

            obs_entity_arr[idx, 0] = 1  # mask
            obs_entity_arr[idx, 1] = entId  # ID
            if entId in beattacked_map:
                obs_entity_arr[idx, 2] = beattacked_map[entId]  # attackerID
            obs_entity_arr[idx, 3] = ent_info["base"]["level"]    # level
            obs_entity_arr[idx, 4] = ent_info["base"]["item_level"]   # item_level
            obs_entity_arr[idx, 5] = 0  # comm
            obs_entity_arr[idx, 6] = ent_info["base"]["population"] # population
            obs_entity_arr[idx, 7] = ent_info["base"]["r"]  # R
            obs_entity_arr[idx, 8] = ent_info["base"]["c"]  # C
            obs_entity_arr[idx, 9] = ent_info["history"]["damage"]  # damage
            obs_entity_arr[idx, 10] = ent_info["history"]["timeAlive"]  # timealive
            obs_entity_arr[idx, 11] = 0  # freeze (deprecated)
            # Extract gold from inventory
            if step==0 and entId>0:
                obs_entity_arr[idx, 12] = 0  # gold
            else:
                if entId>0 and last_packet:
                    ent_info_ = last_packet["player"][str(entId)]
                else:
                    ent_info_ = ent_info
                for item in ent_info_["inventory"]["items"]:
                    if item["item"] == "Gold":
                        obs_entity_arr[idx, 12] = item["quantity"]  # gold
                        break
            if entId>0:
                obs_entity_arr[idx, 13] = ent_info["resource"]["health"]["val"]  # health
                obs_entity_arr[idx, 14] = ent_info["resource"]["food"]["val"]   # food
                obs_entity_arr[idx, 15] = ent_info["resource"]["water"]["val"]  # water
                obs_entity_arr[idx, 16] = ent_info["skills"]["melee"]["level"]  # melee level
                obs_entity_arr[idx, 17] = ent_info["skills"]["range"]["level"]  # range level
                obs_entity_arr[idx, 18] = ent_info["skills"]["mage"]["level"]  # mage level
                obs_entity_arr[idx, 19] = ent_info["skills"]["fishing"]["level"]  # fishing level
                obs_entity_arr[idx, 20] = ent_info["skills"]["herbalism"]["level"]  # Herbalism level
                obs_entity_arr[idx, 21] = ent_info["skills"]["prospecting"]["level"]  # Prospecting level
                obs_entity_arr[idx, 22] = ent_info["skills"]["carving"]["level"]  # Carving level
                obs_entity_arr[idx, 23] = ent_info["skills"]["alchemy"]["level"]  # Alchemy level
            else:
                obs_entity_arr[idx, 13] = ent_info["resource"]["health"]["val"]  # health
                obs_entity_arr[idx, 14] = 100   # npc without food
                obs_entity_arr[idx, 15] = 100   # npc without water
                obs_entity_arr[idx, 16] = ent_info["skills"]["melee"]["level"]  # melee level
                obs_entity_arr[idx, 17] = ent_info["skills"]["range"]["level"]  # range level
                obs_entity_arr[idx, 18] = ent_info["skills"]["mage"]["level"]  # mage level
                # obs_entity_arr[idx, 19] = ent_info["skills"]["fishing"]["level"]  # fishing level
                # obs_entity_arr[idx, 20] = ent_info["skills"]["herbalism"]["level"]  # Herbalism level
                # obs_entity_arr[idx, 21] = ent_info["skills"]["prospecting"]["level"]  # Prospecting level
                # obs_entity_arr[idx, 22] = ent_info["skills"]["carving"]["level"]  # Carving level
                # obs_entity_arr[idx, 23] = ent_info["skills"]["alchemy"]["level"]  # Alchemy level

        # Fill tile data
        obs[player_id]["Tile"]["N"][0] = 15
        obs_tile_arr = obs[player_id]["Tile"]["Continuous"]
        r_top_left = r - 7
        c_top_left = c - 7
        for row in range(15):
            for col in range(15):
                r_tile = r_top_left + row
                c_tile = c_top_left + col
                index_tile = cur_map[r_tile][c_tile]
                NEnts_tile = nents_tile[r_tile][c_tile]

                idx_arr = row*15 + col
                obs_tile_arr[idx_arr, 0] = NEnts_tile
                obs_tile_arr[idx_arr, 1] = index_tile
                obs_tile_arr[idx_arr, 2] = r_tile
                obs_tile_arr[idx_arr, 3] = c_tile

        # Fill item data
        items = player_pkt[str(player_id)]["inventory"]["items"]
        equipment =  player_pkt[str(player_id)]["inventory"]["equipment"]
        obs_item_arr = obs[player_id]["Item"]["Continuous"]
        obs[player_id]["Item"]["N"][0] = len(items)
        for idx, item in enumerate(items):
            obs_item_arr[idx, 0] =  0 # ID
            obs_item_arr[idx, 1] = item_name_to_index[item["item"]] # index
            obs_item_arr[idx, 2] = item["level"] # level
            obs_item_arr[idx, 3] = item["capacity"] # capacity
            obs_item_arr[idx, 4] = item["quantity"] # quantity
            if item["item"] == "Gold":
                obs_item_arr[idx, 5] = 0 # tradable
            else:
                obs_item_arr[idx, 5] = 1 # tradable
            obs_item_arr[idx, 6] = item["melee_attack"] # meleeattack
            obs_item_arr[idx, 7] = item["range_attack"] # rangeattack
            obs_item_arr[idx, 8] = item["mage_attack"] # mageattack
            obs_item_arr[idx, 9] = item["melee_defense"] # meleedefense
            obs_item_arr[idx, 10] = item["range_defense"]   # rangedefense
            obs_item_arr[idx, 11] = item["mage_defense"]    # magedefense
            obs_item_arr[idx, 12] = item["health_restore"]    # health restore
            obs_item_arr[idx, 13] = item["resource_restore"]    # resource restore
            obs_item_arr[idx, 14] = item["price"]   # price
            for key in ["hat", "top", "bottom", "held", "ammunition"]:
                if key in equipment and equipment[key]["item"] == item["item"] and equipment[key]["level"] == item["level"]:
                    obs_item_arr[idx, 15] = 1   # equipped
                    break
        
        # Fill market data
        obs_market_arr = obs[player_id]["Market"]["Continuous"]
        obs[player_id]["Market"]["N"][0] = len(market_pkt)
        for idx, goods_name in enumerate(list(market_pkt.keys())):
            goods = market_pkt[goods_name]
            name, level = goods_name.split("_")
            level = int(level)
            price, supply = int(goods["price"]), int(goods["supply"])
            defense, melee_attack, range_attack, mage_attack, health_restore, resource_restore = 0, 0, 0, 0, 0, 0
            # Armor
            if name in ["Hat", "Top", "Bottom"]: 
                defense = CompetitionConfig.EQUIPMENT_ARMOR_BASE_DEFENSE + level * CompetitionConfig.EQUIPMENT_ARMOR_LEVEL_DEFENSE
            # Tool
            if name in ["Rod", "Gloves", "Pickaxe", "Chisel", "Arcane"]:
                defense = CompetitionConfig.EQUIPMENT_TOOL_BASE_DEFENSE + level * CompetitionConfig.EQUIPMENT_TOOL_LEVEL_DEFENSE
            # Weapon
            if name  == "Sword":
                melee_attack = CompetitionConfig.EQUIPMENT_WEAPON_BASE_DAMAGE + level * CompetitionConfig.EQUIPMENT_WEAPON_LEVEL_DAMAGE
            if name  == "Bow":
                range_attack = CompetitionConfig.EQUIPMENT_WEAPON_BASE_DAMAGE + level * CompetitionConfig.EQUIPMENT_WEAPON_LEVEL_DAMAGE
            if name  == "Wand":
                mage_attack = CompetitionConfig.EQUIPMENT_WEAPON_BASE_DAMAGE + level * CompetitionConfig.EQUIPMENT_WEAPON_LEVEL_DAMAGE
            # Ammunition
            if name  == "Scrap":
                melee_attack = CompetitionConfig.EQUIPMENT_AMMUNITION_BASE_DAMAGE + level * CompetitionConfig.EQUIPMENT_AMMUNITION_LEVEL_DAMAGE
            if name  == "Shaving":
                range_attack = CompetitionConfig.EQUIPMENT_AMMUNITION_BASE_DAMAGE + level * CompetitionConfig.EQUIPMENT_AMMUNITION_LEVEL_DAMAGE
            if name  == "Shard":
                mage_attack = CompetitionConfig.EQUIPMENT_AMMUNITION_BASE_DAMAGE + level * CompetitionConfig.EQUIPMENT_AMMUNITION_LEVEL_DAMAGE
            # Consumable
            if name == "Ration":
                resource_restore = CompetitionConfig.PROFESSION_CONSUMABLE_RESTORE(level)
            if name == "Poultice":
                health_restore = CompetitionConfig.PROFESSION_CONSUMABLE_RESTORE(level)

            obs_market_arr[idx, 0] = 0   # id
            obs_market_arr[idx, 1] = item_name_to_index[name]   # index
            obs_market_arr[idx, 2] = level   # level
            obs_market_arr[idx, 3] = 0   # capacity
            obs_market_arr[idx, 4] = 1   # quantity
            obs_market_arr[idx, 5] = 1   # tradable
            obs_market_arr[idx, 6] = melee_attack    # melee attack
            obs_market_arr[idx, 7] = range_attack # range attack
            obs_market_arr[idx, 8] = mage_attack  # mage attack
            obs_market_arr[idx, 9] =  defense  # melee defense
            obs_market_arr[idx, 10] = defense  # range defense
            obs_market_arr[idx, 11] = defense  # mage defense
            obs_market_arr[idx, 12] = health_restore  # health restore
            obs_market_arr[idx, 13] = resource_restore  # resource restore
            obs_market_arr[idx, 14] = price # price
            obs_market_arr[idx, 15] = 0  # equipped
    
    return obs
def extract_action_of_packet(obs: dict, packet: dict):
    player_pkt = packet["player"]
    
    # Create actions
    actions = {
        int(player_id): {} for player_id in player_pkt
    }
    
    # Extract actions
    attack_key = "attack"
    move_key = "Move"
    buy_key = "Buy"
    sell_key = "Sell"
    use_key = "Use"
    for player_id in player_pkt:
        obs_pid = obs[int(player_id)]
        player_actions = player_pkt[player_id]["history"]["actions"]
        
        if attack_key in player_pkt[player_id]["history"]:
            attack = player_pkt[player_id]["history"][attack_key]
            target = attack["target"]
            style = attack["style"]
            style_values = {
                #edges return [Melee, Range, Mage]
                "Melee": 0,
                "Range": 1,
                "Mage": 2
            }
            # get target idx from corresponding obs
            target_idx = None
            for idx in range(100):
                if obs_pid["Entity"]["Continuous"][idx, 0] == 0:
                    break
                entId = obs_pid["Entity"]["Continuous"][idx, 1]
                if target == entId:
                    target_idx = idx
                    break
            assert target_idx, f"invalid target_idx:{target_idx}"
            actions[int(player_id)][nmmo.io.action.Attack] = {
                nmmo.io.action.Style: style_values[style],
                nmmo.io.action.Target: target_idx
            }
        if move_key in player_actions:
            # edges() return [North, South, East, West]
            direction_value = {
                "North": 0,
                "South": 1,
                "East": 2,
                "West": 3,
            }
            direction_str = player_actions[move_key]["Direction"]
            direction = direction_value[direction_str]
            actions[int(player_id)][nmmo.io.action.Move] = {
                nmmo.io.action.Direction: direction
            }
        if use_key in player_actions:
            use_item = player_actions[use_key]["Item"]
            use_item_name = use_item["item"]
            use_item_level = use_item["level"]
            
            obs_item_arr = obs_pid["Item"]["Continuous"]
            num_items = obs_pid["Item"]["N"][0]
            for idx in range(num_items):
                cur_item_name =  item_value_to_name[obs_item_arr[idx, 1]]
                cur_item_level = obs_item_arr[idx, 2]
                if cur_item_name==use_item_name and cur_item_level==use_item_level:
                    actions[int(player_id)][nmmo.io.action.Use] = {
                        nmmo.io.action.Item: idx
                    }
                    break
        if sell_key in player_actions:
            sell_price_cls = player_actions[sell_key]["Price"]
            sell_price = int(sell_price_cls.split("_")[1])
            sell_item = player_actions[sell_key]["Item"]
            sell_item_name = sell_item["item"]
            sell_item_level = sell_item["level"]
            
            obs_item_arr = obs_pid["Item"]["Continuous"]
            num_items = obs_pid["Item"]["N"][0]
            for item_idx in range(num_items):
                cur_item_name =  item_value_to_name[obs_item_arr[item_idx, 1]]
                cur_item_level = obs_item_arr[item_idx, 2]
                if cur_item_name==sell_item_name and cur_item_level==sell_item_level:
                    actions[int(player_id)][nmmo.io.action.Sell] = {
                        nmmo.io.action.Item: item_idx,
                        nmmo.io.action.Price: sell_price
                    }
                    break
        if buy_key in player_actions:
            buy_item = player_actions[buy_key]["Item"]
            buy_item_name = buy_item["item"]
            buy_item_level = buy_item["level"]
           
            obs_market_arr = obs_pid["Market"]["Continuous"]
            num_items = obs_pid["Market"]["N"][0]
            for item_idx in range(num_items):
                cur_item_name = item_value_to_name[obs_market_arr[item_idx, 1]] # item value
                cur_item_level = obs_market_arr[item_idx, 2]   # level
                if cur_item_name==buy_item_name and cur_item_level==buy_item_level:
                    actions[int(player_id)][nmmo.io.action.Buy] = {
                        nmmo.io.action.Item: item_idx
                    }
                    break
    
    return actions

    lzma_path = 'replay1000/replay-0-0.lzma'
def parse_lzma(lzma_path):
    obs_traj = []
    actions_traj = []
    
    with open(lzma_path, "rb") as fp:
        data = fp.read()
        data = lzma.decompress(data, format=lzma.FORMAT_ALONE)
        data = json.loads(data.decode('utf-8'))
        initial_map = data["map"]
        packets = data["packets"]
        print(f"{lzma_path} loaded as json!")
        
        # the first packet is empty, remove it
        packets.pop(0)
        
        for step, packet in enumerate(packets):
            if step < len(packets)-1:   # without action in the last packet, ignore it
                last_packet = None
                if step>0:
                    last_packet = packets[step-1]
                next_packet = packets[step+1]
                obs = extract_obs_of_packet(step, 
                                            deepcopy(initial_map), 
                                            packet, 
                                            next_packet, 
                                            last_packet)
                action = extract_action_of_packet(obs, next_packet)
                obs_traj.append(obs)
                actions_traj.append(action)
        
        print(f"{lzma_path} parsed to ndarray!")
    
    return obs_traj, actions_traj
def parse_replay(filepath: str, save_path: str):
    dataset = []
    obs_traj, actions_traj = parse_lzma(filepath)
    for obs, action in zip(obs_traj, actions_traj):
        for pid in range(121, 129):
            if pid not in action:
                if pid not in obs:
                    continue
            o, a = obs[pid], action[pid]
            dataset.append([o, a])
    print(f"save dataset: {save_path}, samples num: {len(dataset)}")
    np.savez_compressed(save_path, data=dataset)


if __name__ == "__main__":
    replays_dir = './replay1000'
    npy_save_dir = './dataset'
    # Get replay file paths
    replay_files = os.listdir(replays_dir)
    replay_paths = [
        os.path.join(replays_dir, replay_file)
        for replay_file in replay_files
    ]
    os.makedirs(npy_save_dir, exist_ok=True)
    replay_basenames = [
        os.path.basename(replay_path)
        for replay_path in replay_paths
    ]
    npy_save_paths = [
        os.path.join(npy_save_dir,
                        f"{basename.split('.')[0]}.npz")
        for basename in replay_basenames
    ]
    for rp, yp in zip(replay_paths, npy_save_paths):
        parse_replay(rp, yp)

    