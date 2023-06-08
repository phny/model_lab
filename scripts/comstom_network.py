import argparse
import os
import torch
import torch.nn.functional as F

class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv_layer1 = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv_layer2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.pooling_layer = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc_layer1 = torch.nn.Linear(64*14*14, 128)
        self.fc_layer2 = torch.nn.Linear(128, 3)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv_layer1(x)))
        x = self.bn2(F.relu(self.conv_layer2(x)))
        x = self.pooling_layer(x)
        x = x.view(-1, x.numel())
        x = F.relu(self.fc_layer1(x))
        x = self.fc_layer2(x)
        return x
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_onnx', type=str, default='sample.onnx', help='onnx save path')
    args = parser.parse_args()
    model_name = args.output_onnx
    
    net = Classifier()
    x = torch.randn(1, 3, 28, 28)
    
    output = net(x)
    
    torch.onnx.export(model=net, 
                    args=x, 
                    f=model_name, 
                    input_names=["input"], 
                    output_names=["output"], 
                    opset_version=11)