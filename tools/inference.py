import os
import sys
import torch
import argparse
import json

from tqdm.auto import tqdm

import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.tracking import TrackletManager, VeloLSTM, load_checkpoint
from utils.visualizer import Visualizer
from model import BaseMonoDetector
from utils.engine_utils import load_cfg, tprint, move_data_device
from dataset.kitti_raw_dataset import KITTIRawDataset


# Arguments
parser = argparse.ArgumentParser('MonoCon Tester for KITTI Raw Dataset')
parser.add_argument('--data_dir',
                    type=str,
                    help="Path where sequence images are saved")
parser.add_argument('--calib_file',
                    type=str,
                    help="Path to calibration file (.txt)")
parser.add_argument('--checkpoint_file', 
                    type=str,
                    help="Path of the checkpoint file (.pth)")
parser.add_argument('--gpu_id', type=int, default=0, help="Index of GPU to use for testing")
parser.add_argument('--fps', type=int, default=25, help="FPS of the result video")
parser.add_argument('--save_dir', 
                    type=str,
                    help="Path of the directory to save the inferenced video")
parser.add_argument('--config_file',
                    type=str,
                    help="Path of the config file (.yaml)")
parser.add_argument('--LSTM_checkpoint_file',
                    type=str,
                    help="Path of the LSTM checkpoint file (.pth)")
args = parser.parse_args()

# Load Config
cfg = load_cfg(args.config_file)
cfg.GPU_ID = args.gpu_id


# Main

# (1) Build Dataset
dataset = KITTIRawDataset(args.data_dir, args.calib_file)


# (2) Build Model
device = f'cuda:{args.gpu_id}'

detector = BaseMonoDetector(
            num_dla_layers=cfg.MODEL.BACKBONE.NUM_LAYERS,
            pretrained_backbone=cfg.MODEL.BACKBONE.IMAGENET_PRETRAINED,
            dense_heads=cfg.DETECTOR)
detector.load_checkpoint(args.checkpoint_file)
detector.to(device)
detector.eval()

tprint(f"Checkpoint '{args.checkpoint_file}' is loaded to model.")


# (3) Inference
vis_results = []
model = VeloLSTM(
        batch_size=1,
        feature_dim=64,
        hidden_size=128,
        num_layers=2,
        loc_dim=7,
        dropout=0.0).to("cuda")
model.eval()
load_checkpoint(
    model,
    args.LSTM_checkpoint_file,
    optimizer=None,
    is_test=True)
trackletmanager = TrackletManager(model=model)
json_result = {
    "project" :  "Group_24_3D_localization",
    "output": []
}


frame_number=1
with torch.no_grad():
    for data in tqdm(dataset, desc="Collecting Results..."):
        data = move_data_device(data, device)
        vis_result = detector.batch_eval(data, get_vis_format=True)
        frame = vis_result[0]['img_bbox']
        frame['boxes_3d'] = frame['boxes_3d'].numpy()
        frame['labels_3d'] = frame['labels_3d'].numpy()
        filtered_detections_frame = trackletmanager.removeDupplicateIou(frame)
        predicted_box = trackletmanager.predict_kalman()
        tracketls = trackletmanager.update_tracklets(predicted_box,filtered_detections_frame)
        # print("length",len(trackletmanager.tracklets))
        refined_frame = trackletmanager.extract_result_from_tracklets()

        vis_results.append({"img_bbox":refined_frame,"img_bbox2d":[]})

        # json_result.append(
        #     {
        #         'boxes_3d':vis_result[0]['img_bbox']['boxes_3d'].tolist(),
        #         'scores_3d':vis_result[0]['img_bbox']['scores_3d'].tolist(),
        #         'labels_3d':vis_result[0]['img_bbox']['labels_3d'].tolist(),
        #     }
        # )
        json_result["output"].append(
            {
                "frame":frame_number,
                "predictions":vis_result[0]['img_bbox']['boxes_3d'].tolist(),
                "labels_3d":vis_result[0]['img_bbox']['labels_3d'].tolist()
            }
        )
        frame_number+=1

# save result in json file
vis_results_json = json.dumps(json_result)
with open(args.save_dir+"/vis_results.json", "w") as f:
    f.write(vis_results_json)
# (4) Visualize
visualizer = Visualizer(dataset, vis_format=vis_results)

visualizer.export_as_video(args.save_dir, plot_items=['3d','bev'], fps=args.fps)