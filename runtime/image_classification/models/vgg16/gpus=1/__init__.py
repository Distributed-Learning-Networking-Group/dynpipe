# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .vgg16 import VGG16Partitioned
import re
import torch


def arch():
    return "vgg16"


def model(criterion):
    return [
        (VGG16Partitioned(), ["input"], ["output"]),
        (criterion, ["output"], ["loss"])
    ]


def full_model():
    return VGG16Partitioned()


class Vgg16():
    def __init__(self, declares, calculations):
        self.declares = declares
        self.calculations = calculations

    def generate_layer_blocks(self):
        self.layers = {}
        for layer in self.declares.split('\n'):
            m = re.search(r'self.layer([0-9]+)', layer)
            # print(m)
            layer_id = int(m.group(1))
            self.layers[layer_id] = layer
        self.blocks = [[]]
        for line in self.calculations.split('\n'):
            self.blocks[-1].append(line)
            if 'out32' not in line:
                self.blocks.append([])
        # print(self.layers)
        # print(self.blocks)
        # print(len(self.layers))
        # print(len(self.blocks))
        # print("end")

    def generate_stage(self, start, end):
        inputs = []
        outputs = []
        declare = []
        calculation = []
        for i in range(start, end):
            for line in self.blocks[i]:
                calculation.append(line)
                m = re.search(r'self.layer([0-9]+)', line)
                if m is not None:
                    layer_id = int(m.group(1))
                    declare.append(self.layers[layer_id])
                out = re.findall(r'out\d+', line)
                for arg in out[1:]:
                    if arg not in outputs and arg not in inputs:
                        inputs.append(arg)
                if out[0] not in outputs:
                    outputs.append(out[0])
        declare.append("self._initialize_weights()")
        return declare, calculation, inputs, outputs


class Stage(torch.nn.Module):
    def __init__(self, inputs, outputs, declares, calcus, fraction):
        super(Stage, self).__init__()
        # print("in Stage")
        # print("in out fraction")
        # print("{} {} {}".format(inputs, outputs, fraction), flush = True)
        self.dummy = torch.ones(1, dtype=torch.float32, requires_grad=True)
        # print("declares")
        exec('\n'.join(declares))
        # print("back")
        back = int(fraction * len(calcus))
        # print(back)
        if back == len(calcus):
            no_cp_ = ["{} = args[{}]".format(name, i)
                      for i, name in enumerate(inputs)]
            no_cp_.append("cp_out = cp.checkpoint(self.cp_forward, {}, self.dummy)".format(
                ','.join(inputs)))

            cp_ = calcus
            cp_i = 0
            cp_return = []
            no_cp_return = []
            for output in outputs:
                if output not in inputs:
                    cp_return.append(output)
                    no_cp_return.append("cp_out[{}]".format(cp_i))
                    cp_i += 1
                else:
                    no_cp_return.append(output)

            cp_ = ["{} = args[{}]".format(name, i)
                   for i, name in enumerate(inputs)] + cp_
            cp_.append("self.cp_out = ({},)".format(', '.join(cp_return)))
            no_cp_.append("self.out = ({},)".format(', '.join(no_cp_return)))

            self.cp = '\n'.join(cp_)
            self.no_cp = '\n'.join(no_cp_)
        elif back == 0:
            self.cp = "assert 1 == 0"
            no_cp_ = calcus

            no_cp_ = ["{} = args[{}]".format(name, i)
                      for i, name in enumerate(inputs)] + no_cp_
            no_cp_.append("self.out = ({})".format(', '.join(outputs)))

            self.no_cp = '\n'.join(no_cp_)
        else:
            no_cp_ = calcus[:-back]
            cp_ = calcus[-back:]

            no_cp_ = ["{} = args[{}]".format(name, i)
                      for i, name in enumerate(inputs)] + no_cp_

            cp_inputs = []
            cp_outputs = []
            for line in cp_:
                out = re.findall(r'out\d+', line)
                for arg in out[1:]:
                    if arg not in cp_outputs and arg not in cp_inputs:
                        cp_inputs.append(arg)
                if out[0] not in cp_outputs:
                    cp_outputs.append(out[0])

            cp_i = 0
            cp_return = []
            no_cp_return = []
            for output in outputs:
                if output in cp_outputs:
                    cp_return.append(output)
                    no_cp_return.append("cp_out[{}]".format(cp_i))
                    cp_i += 1
                else:
                    no_cp_return.append(output)

            no_cp_.append("cp_out = cp.checkpoint(self.cp_forward, {})".format(
                ', '.join(cp_inputs)))
            no_cp_.append("self.out = ({},)".format(', '.join(no_cp_return)))
            cp_ = ["{} = args[{}]".format(name, i)
                   for i, name in enumerate(cp_inputs)] + cp_
            cp_.append("self.cp_out = ({},)".format(', '.join(cp_return)))

            self.cp = '\n'.join(cp_)
            self.no_cp = '\n'.join(no_cp_)
        # print(self.no_cp)

    def forward(self, *args):
        exec(self.no_cp)
        return self.out

    def cp_forward(self, *args):
        exec(self.cp)
        return self.cp_out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)


def replace(inputs):
    for i in range(len(inputs)):
        if inputs[i] == 'out0':
            inputs[i] = 'input0'
        # elif inputs[i] == 'out1':
        #     inputs[i] = 'input1'
        # elif inputs[i] == 'out2':
        #     inputs[i] = 'input2'
    return inputs


def model_vgg16(criterion, partition, recompute_ratio):
    _declares = get_declares()
    _calculations = get_caculations()
    module = Vgg16(_declares, _calculations)
    module.generate_layer_blocks()
    start = 0
    inputs = []
    outputs = [['out41']]
    all_outputs = []
    declares = []
    calculations = []
    for i in partition:
        stage = module.generate_stage(start, start + i)
        start += i
        declares.append(stage[0])
        calculations.append(stage[1])
        inputs.append(stage[2])
        all_outputs.append(stage[3])

    for i in range(len(partition) - 1, 0, -1):
        previous_output = []
        for name in inputs[i]:
            if name != 'out0':
                previous_output.append(name)
        for name in outputs[0]:
            if name not in all_outputs[i] and name not in previous_output:
                previous_output.append(name)
        outputs.insert(0, previous_output)
    # (Stage0(), ["input0"], ["out0"]),
    # (Stage1(), ["out0"], ["out1"]),
    # (criterion, ["out1"], ["loss"])
    print(
        "Stage(inputs[0], outputs[0], declares[0], calculations[0], recompute_ratio[0])")
    print(type(Stage(inputs[0], outputs[0], declares[0],
          calculations[0], recompute_ratio[0])))
    return [
        (
            Stage(inputs[0], outputs[0], declares[0], calculations[0], recompute_ratio[0]), replace(inputs[0]), outputs[0]),
        (criterion, outputs[0], ["loss"])
    ]


def get_declares():
    return '''self.layer2 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
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
self.layer41 = torch.nn.Linear(in_features=4096, out_features=10, bias=True)'''


def get_caculations():
    return '''out2 = self.layer2(out0)
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
out41 = self.layer41(out40)'''
