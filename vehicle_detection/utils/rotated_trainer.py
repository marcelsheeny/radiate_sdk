import numpy as np
import copy
import torch

from detectron2.structures import BoxMode
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from detectron2.data import build_detection_train_loader
from detectron2.engine import DefaultTrainer


def transform_instance_annotations(annotation, transforms, image_size, *, keypoint_hflip_indices=None):
    if annotation["bbox_mode"] == BoxMode.XYWHA_ABS:
        annotation["bbox"] = transforms.apply_rotated_box(
            np.asarray([annotation["bbox"]]))[0]
    else:
        bbox = BoxMode.convert(
            annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
        # Note that bbox is 1d (per-instance bounding box)
        annotation["bbox"] = transforms.apply_box([bbox])[0]
        annotation["bbox_mode"] = BoxMode.XYXY_ABS

    return annotation


def mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    image, transforms = T.apply_transform_gens([T.Resize((1152, 1152))], image)
    dataset_dict["image"] = torch.as_tensor(
        image.transpose(2, 0, 1).astype("float32"))

    annos = [
        transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    instances = utils.annotations_to_instances_rotated(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict


class RotatedTrainer(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=mapper)
