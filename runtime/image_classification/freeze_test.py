# import torch
# import torchvision
# import torchvision.transforms as transforms
# from torchvision import models
# import torch.nn as nn
# import torch.optim as optim
#
# # 步骤1: 加载并预处理CIFAR-10数据集
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])
#
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
#
# # 步骤2: 加载VGG16模型
# vgg16 = models.vgg16(pretrained=True)
#
# # 修改最后的全连接层，以匹配CIFAR-10的类别数
# vgg16.classifier[6] = nn.Linear(4096, 10)
#
# # 步骤3: 冻结后一半的层
# total_features = len(vgg16.features)
# features_to_freeze = total_features // 2
#
# for i, child in enumerate(vgg16.features.children()):
#     if i < features_to_freeze:
#         for param in child.parameters():
#             param.requires_grad = False
#
# # 步骤4: 设置优化器和损失函数
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(vgg16.parameters(), lr=0.001, momentum=0.9)
#
# # 步骤5: 训练模型
# total_steps = 400
# for step in range(total_steps):
#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         inputs, labels = data
#
#         optimizer.zero_grad()
#
#         outputs = vgg16(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#         if i % 2000 == 1999:  # 每2000个mini-batches打印一次
#             print('[%d, %5d] loss: %.3f' % (step + 1, i + 1, running_loss / 2000))
#             running_loss = 0.0
#
#     # 在200个step后解冻所有层
#     if step == 199:
#         for child in vgg16.features.children():
#             for param in child.parameters():
#                 param.requires_grad = True
#
# print('Finished Training')
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import torch.optim as optim

# 检查GPU是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize(224),  # VGG16要求输入至少为224x224
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
class vgg16(torch.nn.Module):
    def __init__(self):
        super(vgg16, self).__init__()
        self.layer2 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer3 = torch.nn.ReLU(inplace=True)
        self.layer4 = torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer5 = torch.nn.ReLU(inplace=True)
        self.layer6 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.layer7 = torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer8 = torch.nn.ReLU(inplace=True)
        self.layer9 = torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.layer10 = torch.nn.ReLU(inplace=True)
        self.layer11 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.layer12 = torch.nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer13 = torch.nn.ReLU(inplace=True)
        self.layer14 = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer15 = torch.nn.ReLU(inplace=True)
        self.layer16 = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer17 = torch.nn.ReLU(inplace=True)
        self.layer18 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.layer19 = torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer20 = torch.nn.ReLU(inplace=True)
        self.layer21 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer22 = torch.nn.ReLU(inplace=True)
        self.layer23 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer24 = torch.nn.ReLU(inplace=True)
        self.layer25 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.layer26 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer27 = torch.nn.ReLU(inplace=True)
        self.layer28 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer29 = torch.nn.ReLU(inplace=True)
        self.layer30 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer31 = torch.nn.ReLU(inplace=True)
        self.layer32 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.layer35 = torch.nn.Linear(in_features=25088, out_features=4096, bias=True)
        self.layer36 = torch.nn.ReLU(inplace=True)
        self.layer37 = torch.nn.Dropout(p=0.5)
        self.layer38 = torch.nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.layer39 = torch.nn.ReLU(inplace=True)
        self.layer40 = torch.nn.Dropout(p=0.5)
        self.layer41 = torch.nn.Linear(in_features=4096, out_features=10, bias=True)
        self._initialize_weights()
    def forward(self, input0):
        out0 = input0.clone()
        out2 = self.layer2(out0)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)
        out8 = self.layer8(out7)
        out9 = self.layer9(out8)

        out10 = self.layer10(out9)
        out11 = self.layer11(out10)
        out12 = self.layer12(out11)
        out13 = self.layer13(out12)
        out14 = self.layer14(out13)
        out15 = self.layer15(out14)
        out16 = self.layer16(out15)
        out17 = self.layer17(out16)
        out18 = self.layer18(out17)
        out19 = self.layer19(out18)
        out20 = self.layer20(out19)
        out21 = self.layer21(out20)
        out22 = self.layer22(out21)
        out23 = self.layer23(out22)
        out24 = self.layer24(out23)
        out25 = self.layer25(out24)
        out26 = self.layer26(out25)
        out27 = self.layer27(out26)
        out28 = self.layer28(out27)
        out29 = self.layer29(out28)
        out30 = self.layer30(out29)
        out31 = self.layer31(out30)
        out32 = self.layer32(out31)
        out33 = out32.size(0)
        out34 = out32.view(out33, -1)
        out35 = self.layer35(out34)
        out36 = self.layer36(out35)
        out37 = self.layer37(out36)
        out38 = self.layer38(out37)
        out39 = self.layer39(out38)
        out40 = self.layer40(out39)
        out41 = self.layer41(out40)
        return out41
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)

# 下载CIFAR-10数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 修改VGG16模型
# vgg16 = models.vgg16(pretrained=False)  # 不使用预训练权重
# vgg16.classifier[6] = nn.Linear(4096, 10)  # 修改最后一层为10个输出
vgg16=vgg16()
# 将模型发送到GPU
vgg16 = vgg16.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()

print(type(vgg16.parameters()))
print(vgg16.parameters())
optimizer = optim.SGD(vgg16.parameters(), lr=0.001, momentum=0.9)
print(vgg16)
# 训练模型
for epoch in range(10):  # 迭代10个epoch
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):

        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = vgg16(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 500 == 499:  # 每2000个mini-batches打印一次
            correct = 0
            total = 0

            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = vgg16(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

# 测试模型
