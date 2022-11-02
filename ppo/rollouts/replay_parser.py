import os
import random
import numpy as np
import multiprocessing as mp


class ReplayParser:
    @staticmethod
    def parse_replay_files(replays_dir: str, 
                           npy_save_dir: str,
                           num_workers: int = 4):
        
        # Get replay file paths
        replay_files = os.listdir(replays_dir)
        replay_paths = [
            os.path.join(replays_dir, replay_file)
            for replay_file in replay_files
        ]
        os.makedirs(npy_save_dir, exist_ok=True)
        print(f"replays dir: {replays_dir}")
        print(f"replays num: {len(replay_paths)}")
        print(f"npy save dir: {npy_save_dir}")
        
        # Parse replay files to obs and actions
        print(f"Parsing replays to samples...")
        replay_basenames = [
            os.path.basename(replay_path)
            for replay_path in replay_paths
        ]
        npy_save_paths = [
            os.path.join(npy_save_dir,
                         f"{basename.split('.')[0]}.npz")
            for basename in replay_basenames
        ]
        print(f"num of workers: {num_workers}")
    
    @staticmethod
    def shuffle_two_npz_files(file_path, another_file_path):
        samples = np.load(file_path, allow_pickle=True)["data"]
        another_samples = np.load(another_file_path, allow_pickle=True)["data"]
        merge_samples = np.append(samples, another_samples, axis=0)
        random.shuffle(merge_samples)
        
        # Overwrite origin npy files
        middle_loc = len(merge_samples)//2
        np.savez_compressed(file_path, data=merge_samples[:middle_loc])
        np.savez_compressed(another_file_path, data=merge_samples[middle_loc:])
        print(f"{file_path} and {another_file_path} shuffled!")

    @staticmethod
    def simple_shuffle_samples(npy_save_dir, num_workers: int = 4):
        # Get .npy file paths
        npy_files = os.listdir(npy_save_dir)
        print(f"npy files num: {len(npy_files)}")
        npy_file_paths = [
            os.path.join(npy_save_dir, npy_file)
            for npy_file in npy_files
        ]
        random.shuffle(npy_file_paths)
        
        # Shuffle two adjacent files
        if len(npy_file_paths)%2!=0:
            npy_file_paths.pop(-1)
        prior_file_paths = [
            file_path for idx, file_path
            in enumerate(npy_file_paths) if idx%2==0
        ]
        next_file_paths = [
            file_path for idx, file_path
            in enumerate(npy_file_paths) if idx%2==1
        ]
        with mp.Pool(processes = num_workers) as pool:
            pool.starmap(ReplayParser.shuffle_two_npz_files, 
                         list(zip(prior_file_paths, next_file_paths)))