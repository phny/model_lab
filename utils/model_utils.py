#coding:utf-8

import mxnet as mx
from mxnet.gluon import SymbolBlock

__all__ = ["LoadModel"]

class LoadModel(object):
    def __init__(self, model_sym, model_params):
        super().__init__()
        self.sym = model_sym
        self.params = model_params

    def ModelInstance(self):
        # 加载模型结构
        sym = mx.sym.load(self.sym)
        # 使用SymbolBlock创建模型
        net = SymbolBlock(outputs=sym, inputs=mx.sym.var('data'))
        # 加载模型参数
        net.load_parameters(self.params, ctx=mx.cpu())
        # 执行前向计算
        input_data = mx.nd.zeros((1, 224, 224, 3), ctx=mx.cpu())
        _ = net(input_data)
        return net

    
