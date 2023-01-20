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
from detectron2.utils.visualizer import ColorMode
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            # Crop Imag
            print(boxes)
        return 
# Create CarDetector object
car_detector = CarDetector(
    config='configs/pointrend_rcnn_R_50_FPN_3x_coco.yaml',
    checkpoint='models/car_detector_1.pth'
)

# Detect panels

class PanelDetector():
    def __init__(self, panel_config, panel_checkpoint, panel_threshold=0.88, model_device='cpu'):
        self.panel_model = None
        self.panel_classes = None
        self.panel_class_metadata = None
        self.panel_config = panel_config
        self.panel_checkpoint = panel_checkpoint
        self.panel_threshold = panel_threshold
        self.model_device = model_device
        self.initialise_panel()

    def initialise_panel(self):

        cfg = get_cfg()
        add_pointrend_config(cfg)
        cfg.merge_from_file(self.panel_config)
        cfg.MODEL.WEIGHTS = self.panel_checkpoint
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.panel_threshold
        cfg.MODEL.DEVICE = self.model_device
        self.panel_model = DefaultPredictor(cfg)
        logger.info('Panel Model Loaded')
    
    def predict(self, img):
        '''
        Function that takes in a Numpy Array and country code and returns an Masked Image or None
        '''
        t1 = time.time()
        predictions = self.panel_model(img)
        t2 = time.time()
        logger.info('Panel Detection Model Inference took: {}'.format(str(t2 - t1)))

        instances = predictions["instances"].to("cpu")

        # Get detection boxes
        boxes = instances.pred_boxes.tensor.cpu().numpy()
        pred_classes = instances.pred_classes.cpu().numpy()
        # If no boxes are detected, return original image and None for mask
        if len(boxes) == 0:
            return Image.fromarray(img), None
        else:
            # Crop Image
            return boxes, pred_classes

panel_detector = PanelDetector(
    panel_config='configs/pointrend_rcnn_X_101_32x8d_FPN_3x_coco.yaml',
    panel_checkpoint='/Users/maixueqiao/Downloads/Project/damage_localisation/models/panel_detector.pth',
    )