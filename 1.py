import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt
import mxnet.ndarray as nd
import warnings
from loguru import logger
from gluoncv.data.transforms.presets.yolo import load_test


ctx = mx.gpu() if mx.context.num_gpus() else mx.cpu()
with warnings.catch_warnings():
    # warnings.simplefilter("ignore")
    model = mx.gluon.nn.SymbolBlock.imports(
        "models/Detection/yolo3_darknet53_voc-symbol.json", 
        ['data'], 
        "models/Detection/yolo3_darknet53_voc-0000.params", 
        ctx=ctx)
    print(model.collect_params())

x, ori_img = load_test("resources/dog_bike_car.jpg")
x = x.transpose((0, 2, 3, 1))
out = model(x)

# logger.info(out)

