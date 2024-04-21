import torch
import re
try:
    from modeling import BertLayerNorm
    from modeling import BertAdd
    from modeling import LinearActivation
    from modeling import BertEmbeddings
    from modeling import BertSelfAttention
except ImportError:
    from bert.modeling import BertLayerNorm
    from bert.modeling import BertAdd
    from bert.modeling import LinearActivation
    from bert.modeling import BertEmbeddings
    from bert.modeling import BertSelfAttention


import torch.utils.checkpoint as cp


class Bert():
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
            if '+' in line:
                self.blocks.append([])
        # print(self.layers)
        print(len(self.blocks))
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
        declare.append("self.apply(self.init_bert_weights)")
        return declare, calculation, inputs, outputs


class Stage(torch.nn.Module):
    def __init__(self, inputs, outputs, declares, calcus, fraction):
        super(Stage, self).__init__()
        # print("in Stage")
        # print("in out fraction")
        # print("{} {} {}".format(inputs, outputs, fraction), flush = True)
        self.dummy = torch.ones(1, dtype=torch.float32, requires_grad=True)
        # print("declares")
        # print(declares)
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
        # print("cp")
        # print(self.cp)
        # print("no_cp")
        # print(self.no_cp)

    def forward(self, *args):
        exec(self.no_cp)
        return self.out

    def cp_forward(self, *args):
        exec(self.cp)
        return self.cp_out
    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, (torch.nn.Linear, torch.nn.Embedding)):
    #             # Slightly different from the TF version which uses truncated_normal for initialization
    #             # cf https://github.com/pytorch/pytorch/pull/5617
    #             m.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    #         elif isinstance(m, BertLayerNorm):
    #             m.bias.data.zero_()
    #             m.weight.data.fill_(1.0)
    #         if isinstance(m, torch.nn.Linear) and m.bias is not None:
    #             m.bias.data.zero_()

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
