#coding:utf-8

import cv2
import numpy as np
import mxnet as mx

__all__ = ["read_image", "draw_detect_bbox"]

def draw_detect_bbox(image, preds, save_path):
    for pred in preds[0]:
        score = pred[1]
        if score < 0.05: continue
        label = int(pred[0])
        bbox = pred[2:6] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
        bbox = bbox.astype(np.int32)
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
        cv2.putText(image, str(label), (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imwrite(save_path, image)