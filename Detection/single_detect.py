#coding:utf-8

import os, sys
sys.path.append(os.getcwd())

import argparse
from loguru import logger
import matplotlib.pyplot as plt

from utils.model_utils import LoadModel
from gluoncv.data.transforms.presets.yolo import load_test
from gluoncv import utils


def main(args):
    # 加载模型
    model = LoadModel(args.model_sym, args.model_params).ModelInstance()
    logger.info(f"Finish load model {args.model_sym}")
    # 加载图片
    image, ori_image = load_test(args.image_path)
    logger.info(f"Finish load image, shape {image.shape}")
    # 推理
    ids, scores, bboxes = model(image)
    logger.info(f"Finish inference, scores {scores}")
    # 保存结果
    dst_path = os.path.join(args.save_dir, os.path.basename(args.image_path))
    ax = utils.viz.plot_bbox(ori_image, bboxes[0], scores[0], ids[0], class_names=model.classes)
    plt.savefig(dst_path)
    logger.info(f"Finish detect, result in {dst_path}")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_sym", "-m", type=str, required=True)
    parser.add_argument("--model_params", "-p", type=str, required=True)
    parser.add_argument("--image_path", "-i", type=str, required=True)
    parser.add_argument("--save_dir", "-s", type=str, required=True)
    args = parser.parse_args()
    
    main(args)