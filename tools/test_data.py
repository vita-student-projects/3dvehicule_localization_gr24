import os
import sys
import torch
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dataset.mono_dataset import MonoDataset

from engine.mono_engine import MonoEngine
from utils.engine_utils import tprint, load_cfg, generate_random_seed, set_random_seed

def main():
    # Arguments

    dataset = MonoDataset(
            "/home/vince/datasets/nuscenes_kitti_exported",
            split="train",
            nuscenes=True)
    dataset.collect_gt_infos()
if __name__ == '__main__':
    main()