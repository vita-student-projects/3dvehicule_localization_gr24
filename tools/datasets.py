import os
import sys
import torch
import argparse
import json
import cv2
from tqdm.auto import tqdm

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dataset.kitti_raw_dataset import KITTIRawDataset


# Arguments
parser = argparse.ArgumentParser('MonoCon Datasets viewer for KITTI Raw Dataset')
parser.add_argument('--data_dir',
                    type=str,
                    help="Path where sequence images are saved")
parser.add_argument('--calib_file',
                    type=str,
                    help="Path to calibration file (.txt)")
args = parser.parse_args()



# Main

# (1) Build Dataset
dataset = KITTIRawDataset(args.data_dir, args.calib_file)

# show first image
cv2.imshow("Image", dataset[0]["img"])
 
# Wait for the user to press a key
cv2.waitKey(0)
 
# Close all windows
cv2.destroyAllWindows()