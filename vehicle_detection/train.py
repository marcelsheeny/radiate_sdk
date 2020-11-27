
import argparse
import matplotlib.pyplot as plt
import cv2
import json
import numpy as np
import os
import random
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

# radiate sdk
import sys
sys.path.insert(0, '..')
import radiate  # noqa


# init params
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", help="Model Name (Ex: faster_rcnn_R_50_FPN_3x)",
                    default='retinanet_R_50_FPN_3x',
                    type=str)

parser.add_argument("--root_folder", help="root folder with radiate dataset",
                    default='../data/radiate/',
                    type=str)

parser.add_argument("--max_iter", help="Maximum number of iterations",
                    default=90000,
                    type=int)

parser.add_argument("--resume", help="Whether to resume training or not",
                    default=False,
                    type=bool)

parser.add_argument("--dataset_mode", help="dataset mode ('good_weather', 'good_and_bad_weather')",
                    default='good_weather',
                    type=str)

# parse arguments
args = parser.parse_args()
model_name = args.model_name
root_dir = args.root_folder
resume = args.resume
dataset_mode = args.dataset_mode
max_iter = args.max_iter


def train(model_name, root_dir, dataset_mode, max_iter):

    # output folder to save models
    output_dir = os.path.join('train_results', model_name + '_' + dataset_mode)
    os.makedirs(output_dir, exist_ok=True)

    # get folders depending on dataset_mode
    folders_train = []
    for curr_dir in os.listdir(root_dir):
        with open(os.path.join(root_dir, curr_dir, 'meta.json')) as f:
            meta = json.load(f)
        if meta["set"] == "train_good_weather":
            folders_train.append(curr_dir)
        elif meta["set"] == "train_good_and_bad_weather" and dataset_mode == "good_and_bad_weather":
            folders_train.append(curr_dir)

    def gen_boundingbox(bbox, angle):

        points = radiate.gen_boundingbox_rot(bbox, angle)

        min_x = np.min(points[0, :])
        min_y = np.min(points[1, :])
        max_x = np.max(points[0, :])
        max_y = np.max(points[1, :])

        return min_x, min_y, max_x, max_y

    def get_radar_dicts(folders):
        dataset_dicts = []
        idd = 0
        folder_size = len(folders)
        for folder in folders:
            radar_folder = os.path.join(root_dir, folder, 'Navtech_Cartesian')
            annotation_path = os.path.join(root_dir,
                                           folder, 'annotations', 'annotations.json')
            with open(annotation_path, 'r') as f_annotation:
                annotation = json.load(f_annotation)

            radar_files = os.listdir(radar_folder)
            radar_files.sort()
            for frame_number in range(len(radar_files)):
                record = {}
                objs = []
                bb_created = False
                idd += 1
                filename = os.path.join(
                    radar_folder, radar_files[frame_number])

                if (not os.path.isfile(filename)):
                    print(filename)
                    continue
                record["file_name"] = filename
                record["image_id"] = idd
                record["height"] = 1152
                record["width"] = 1152

                for object in annotation:
                    if (object['bboxes'][frame_number]):
                        class_obj = object['class_name']
                        if (class_obj != 'pedestrian' and class_obj != 'group_of_pedestrians'):
                            bbox = object['bboxes'][frame_number]['position']
                            angle = object['bboxes'][frame_number]['rotation']
                            bb_created = True
                            if cfg.MODEL.PROPOSAL_GENERATOR.NAME == "RRPN":
                                cx = bbox[0] + bbox[2] / 2
                                cy = bbox[1] + bbox[3] / 2
                                wid = bbox[2]
                                hei = bbox[3]
                                obj = {
                                    "bbox": [cx, cy, wid, hei, angle],
                                    "bbox_mode": BoxMode.XYWHA_ABS,
                                    "category_id": 0,
                                    "iscrowd": 0
                                }
                            else:
                                xmin, ymin, xmax, ymax = gen_boundingbox(
                                    bbox, angle)
                                obj = {
                                    "bbox": [xmin, ymin, xmax, ymax],
                                    "bbox_mode": BoxMode.XYXY_ABS,
                                    "category_id": 0,
                                    "iscrowd": 0
                                }

                            objs.append(obj)
                if bb_created:
                    record["annotations"] = objs
                    dataset_dicts.append(record)
        return dataset_dicts

    dataset_train_name = dataset_mode + '_train'
    dataset_test_name = dataset_mode + '_test'

    DatasetCatalog.register(dataset_train_name,
                            lambda: get_radar_dicts(folders_train))
    MetadataCatalog.get(dataset_train_name).set(thing_classes=["vehicle"])
    MetadataCatalog.get(dataset_train_name).evaluator_type = "coco"

    metadata = MetadataCatalog.get(dataset_train_name)

    cfg_file = os.path.join('test', 'config', model_name + '.yaml')
    cfg = get_cfg()
    cfg.OUTPUT_DIR = output_dir
    cfg.merge_from_file(cfg_file)
    cfg.DATASETS.TRAIN = (dataset_train_name,)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.STEPS: (25000, 35000)
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.2
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128]]

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)

    trainer.resume_or_load(resume=resume)
    trainer.train()


if __name__ == "__main__":
    train(model_name, root_dir, dataset_mode, max_iter)
