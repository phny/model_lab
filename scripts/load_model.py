#coding:utf-8

import os, sys
sys.path.append(os.getcwd())

import mxnet as mx
from loguru import logger
import argparse
from gluoncv.model_zoo import get_model

from utils.model_utils import LoadModel
from gluoncv.data.transforms.presets.yolo import load_test


def main(model_sym, model_params):
    # 从sym、params中加载模型
    loader = LoadModel(model_sym, model_params)
    model = loader.ModelInstance()

    # model = get_model("yolo3_darknet53_voc", pretrained=True)

    # 执行一次前向传播，确认模型可用
    # input_data = mx.nd.zeros((1, 224, 224, 3), ctx=mx.cpu())

    x, img = load_test("resources/dog_bike_car.jpg")
    # TODO 从本地加载的模型输入变成了NCHW, 线上的是NHWC，原因待排查
    x = x.transpose((0, 2, 3, 1))
    
    ids, scores, bboxes = model(x)
    logger.info(f"ids {ids}, scores {scores}, bboxes {bboxes}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_sym", "-s", type=str, required=True)
    parser.add_argument("--model_params", "-p", type=str, required=True)
    args = parser.parse_args()
    
    model_sym = args.model_sym
    model_params = args.model_params
    main(model_sym, model_params)


