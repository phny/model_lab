#coding:utf-8

import mxnet as mx
from loguru import logger
import argparse

from utils.load_model import LoadModel


def main(model_sym, model_params):
    # 从sym、params中加载模型
    loader = LoadModel(model_sym, model_params)
    model = loader.ModelInstance()
    # 执行一次前向传播，确认模型可用
    input_data = mx.nd.zeros((1, 224, 224, 3), ctx=mx.cpu())
    output_data = model(input_data)
    logger.info(output_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_sym", "-s", type=str, required=True)
    parser.add_argument("--model_params", "-p", type=str, required=True)
    args = parser.parse_args()
    
    model_sym = args.model_sym
    model_params = args.model_params
    main(model_sym, model_params)


