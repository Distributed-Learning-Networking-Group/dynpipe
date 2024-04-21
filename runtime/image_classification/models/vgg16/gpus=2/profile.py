
import torch

# import torch
#from ptflops import get_model_complexity_info
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
        self.layer41 = torch.nn.Linear(in_features=4096, out_features=1000, bias=True)
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


# criterion = torch.nn.CrossEntropyLoss()
# model_vgg = vgg16()
# model_name = 'vgg'
# flops, params = get_model_complexity_info(model_vgg, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
# print("%s |%s |%s" % (model_name, flops, params))


torch.cuda.set_device(3)
# 加载VGG16模型
model = vgg16()

# 创建输入张量
input_tensor = torch.randn(64, 3, 224, 224)

# 设置计时器
forward_timer = []
backward_timer = []

# 注册钩子函数，在每个层的前向和反向传播开始和结束时记录时间
def forward_hook(module, input, output):
    forward_timer.append(torch.cuda.Event(enable_timing=True))
    forward_timer[-1].record()

def backward_hook(module, grad_input, grad_output):
    backward_timer.append(torch.cuda.Event(enable_timing=True))
    backward_timer[-1].record()

# 注册钩子函数到每个层
# for module in model.modules():
#     #if isinstance(module, torch.Conv2d) or isinstance(module, torch.Linear):
#     module.register_forward_hook(forward_hook)
#     module.register_backward_hook(backward_hook)



# 将模型移动到GPU上
model = model.cuda()
input_tensor = input_tensor.cuda()

big_tensor = torch.randn(2000, 2000).cuda()

time_list = []

# 进行一些计算，以增加GPU占用率
for i in range(2000):
    print(i)
    result = torch.matmul(big_tensor, big_tensor)

# 前向传播

with torch.autograd.profiler.profile(record_shapes=True) as prof:
    with torch.autograd.profiler.record_function("forward"):
        output_tensor = model(input_tensor)

    with torch.autograd.profiler.record_function("backward"):
        target = torch.randint(0, 1000, (64,)).cuda()
        loss_fn = torch.nn.CrossEntropyLoss()
        # 计算损失
        loss = loss_fn(output_tensor, target)
        loss.backward()

print(prof.key_averages().table())
# with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=True, profile_memory=False) as prof:
#     output = model(input_tensor)
#     loss_fn = torch.nn.CrossEntropyLoss()
#     target = torch.randint(0, 1000, (64,)).cuda()
#
#     # 计算损失
#     loss = loss_fn(output, target)
#     loss.backward()
# print(prof.table())
# 打印每个层的前向和反向传播时间
# for i, module in enumerate(model.modules()):
#     #if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
#     forward_time = forward_timer[i + 1].elapsed_time(forward_timer[i])
#     backward_time = backward_timer[i + 1].elapsed_time(backward_timer[i])
#     print("%dLayer: " %(i))
#     print(f"Forward pass time: {forward_time} ms")
#     print(f"Backward pass time: {backward_time} ms")
#     print()





