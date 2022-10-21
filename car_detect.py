from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

from detectron2.config import get_cfg
from detectron2.projects.point_rend import add_pointrend_config
import detectron2.data.transforms as T
from detectron2.data import MetadataCatalog

from PIL import Image
import numpy as np
import os
import time

import matplotlib.pyplot as plt

# Car Detection class
class CarDetector():
    def __init__(self, config, checkpoint, threshold=0.6, model_device='cpu'):
        self.model = None
        self.config = config
        self.checkpoint = checkpoint
        self.threshold = threshold
        self.model_device = model_device
        self.initialise()

    def initialise(self):
        cfg = get_cfg()
        add_pointrend_config(cfg)
        cfg.merge_from_file(self.config)
        cfg.MODEL.WEIGHTS = self.checkpoint
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold
        cfg.MODEL.DEVICE = self.model_device
        self.model = DefaultPredictor(cfg)
    
    def predict(self, img):
        start_time = time.time()
        predictions = self.model(img)

        # Get detection boxes
        boxes = predictions['instances'].pred_boxes
        masks = predictions['instances'].pred_masks

        # If no boxes are detected, return original image and None for mask
        if len(boxes) == 0:
            return Image.fromarray(img), None
        else:
            # Crop Image
            box = list(boxes)[0].detach().cpu().numpy()
            crop_img, cropped_image = self.crop_object(img, box)
            
            # Crop segmentation mask
            mask = list(masks)[0].detach().cpu().numpy()
            cropped_mask = self.crop_object(mask, box)
            
            return crop_img, cropped_image

    def crop_object(self, image, box):
        image = Image.fromarray(image)
        crop_img =  (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        cropped_image = image.crop(
            (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            )
        return crop_img, cropped_image
# Create CarDetector object
car_detector = CarDetector(
    config='configs/pointrend_rcnn_R_50_FPN_3x_coco.yaml',
    checkpoint='models/car_detector_1.pth'
) 