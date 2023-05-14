#coding:utf-8

import os
from loguru import logger
import gluoncv


model_types = {
    1: "ActionRegconition",
    2: "Classification",
    3: "Detection",
    4: "DepthPrediction",
    5: "Segmentation",
    6: "PostEstimation"
}

__all__ = ["Downloader"]

class Downloader:
    def __init__(self, model_name, model_type, save_path=None):
        super().__init__()
        self.model_name = model_name
        self.model_type = model_type
        self.save_path = "./models" if save_path is None else save_path

    def download(self):
        net = gluoncv.model_zoo.get_model(self.model_name, pretrained=True, root="./models/.caches")
        net.hybridize()
        model_save_path = os.path.join(self.save_path, model_types[self.model_type], self.model_name)
        # layout='HWC'，即NHCW，其中N是batch_size, C是通道数
        gluoncv.utils.export_block(model_save_path, net, preprocess=True, layout='HWC')
        logger.info(f"Downloaded model in {model_save_path}")