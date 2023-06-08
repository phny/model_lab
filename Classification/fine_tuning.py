#coding:utf-8

import torchvision.models as models
from PIL import Image
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


alexnet = models.alexnet(pretrained=True)

im = Image.open('resources/dog.jpg')

cifar10_dataset = torchvision.datasets.CIFAR10(root='./datas',
                                       train=False,
                                       transform=transforms.ToTensor(),
                                       target_transform=None,
                                       download=True)
# 取32张图片的tensor
tensor_dataloader = DataLoader(dataset=cifar10_dataset,
                               batch_size=32)
data_iter = iter(tensor_dataloader)
img_tensor, label_tensor = data_iter.next()
print(img_tensor.shape)
grid_tensor = torchvision.utils.make_grid(img_tensor, nrow=16, padding=2)
grid_img = transforms.ToPILImage()(grid_tensor)
print(grid_img)

# 打印出网络的结构
print(alexnet)
