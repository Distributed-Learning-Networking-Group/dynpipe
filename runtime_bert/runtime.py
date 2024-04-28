# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.



import collections
import itertools
import time
import torch
import torch.distributed as dist

import communication
import communication_tyf_1
import runtime_utilities
import pickle
IMAGE_CLASSIFICATION = "image_classification"
TRANSLATION = "translation"
SPEECH_TO_TEXT = "speech_to_text"
BERT = "bert"


class ModulesWithDependencies:
    def __init__(self, modules_with_dependencies):
        self._modules = []
        self._all_input_names = []
        self._all_output_names = []
        for (module, input_names, output_names) in modules_with_dependencies:
            self._modules.append(module)
            self._all_input_names.append(input_names)
            self._all_output_names.append(output_names)

    def modules(self):
        return self._modules

    def all_input_names(self):
        return self._all_input_names

    def all_output_names(self):
        return self._all_output_names

    def is_input_tensor(self, tensor_name):
        for module_input_names in self._all_input_names:
            if tensor_name in module_input_names:
                return True
        return False


class StageRuntime:
    def __init__(self, model, distributed_backend, fp16, loss_scale,
                 training_tensor_shapes, eval_tensor_shapes,
                 training_tensor_dtypes, inputs_module_destinations,
                 target_tensor_names, configuration_maps, master_addr,
                 rank, local_rank, num_ranks_in_server, verbose_freq,
                 model_type,event,event1,worker_num_sum,batch_size,batch_size_for_communication,stage_num,stage_nums,enable_recompute=False):
        # Metadata needed for forward and backward pass within this stage.
        self.count_for_target_test = 0
        self.count_for_target_test_s = 0
        self.worker_num_sum = worker_num_sum
        self.i_for_initial = torch.tensor([0])
        self.previous_status = torch.zeros(self.worker_num_sum, dtype=torch.float)
        self.real_time = 0
        self.backward_real_time = 0
        self.tensors = []
        self.gradients = {}
        self.distributed_backend = distributed_backend
        self.fp16 = fp16
        self.loss_scale = loss_scale
        self.training_tensor_shapes = training_tensor_shapes
        self.eval_tensor_shapes = eval_tensor_shapes
        self.training_tensor_dtypes = training_tensor_dtypes
        self.model_type = model_type
        self.target_tensor_names = target_tensor_names
        self.EVENT = event
        self.EVENT1=event1
        self.batch_size = batch_size
        self.batch_size_for_communication = batch_size_for_communication
        self.initialize(model, inputs_module_destinations, configuration_maps,
                        master_addr, rank, local_rank, num_ranks_in_server)

        self.verbose_freq = verbose_freq
        self.forward_only = False

        self.forward_stats = runtime_utilities.RuntimeStats(forward=True)
        self.backward_stats = runtime_utilities.RuntimeStats(forward=False)
        self.status = torch.zeros(self.worker_num_sum, dtype=torch.float)
        # Enable recomputation to prevent the need to save activations
        # computed from the forward pass for the backward pass.
        self.enable_recompute = enable_recompute
        self.EVENT = event
        self.EVENT1=event1
        self.time_forward=0
        self.time_backward=0
        self.layer_forward_list = [1.0] * 38
        self.layer_backward_list = [1.0] * 38
        self.layer_communication_list = [0.0] * 37
        self.correct=0
        # Disable recomputation for the last stage.
        if rank == num_ranks_in_server - 1:
            self.enable_recompute = False
        self.stage_num = stage_num
        self.stage_nums = torch.tensor(stage_nums)  #
        self.straggle_for_stage_cmp = torch.ones(self.worker_num_sum, dtype=torch.float)  #
        self.straggle_for_stage_cal = torch.ones(self.worker_num_sum, dtype=torch.float)
        self.initial_status_cmp = torch.zeros(self.worker_num_sum, dtype=torch.float)  #
        self.initial_status_cal = torch.ones(self.worker_num_sum, dtype=torch.float)  #

    def initialize1(self, model, inputs_module_destinations,
                   configuration_maps, master_addr, rank,
                   local_rank, num_ranks_in_server,training_tensor_shapes,training_tensor_dtypes,target_tensor_names):
        self.tensors = []
        self.gradients = {}
        self.training_tensor_shapes = training_tensor_shapes
        self.training_tensor_dtypes = training_tensor_dtypes
        self.target_tensor_names = target_tensor_names
        self.status = torch.zeros(self.worker_num_sum, dtype=torch.float)
        self.i_for_initial = torch.tensor([0])
        self.send_ranks = {}
        self.receive_ranks = {}
        self.rank = rank
        self.local_rank = local_rank
        self.stage = None
        self.tensor_tags = {}
        self.forward_minibatch_id = 0
        self.backward_minibatch_id = 0
        self.criterion_input_name = str(model[-1][1][0])
        tensor_tag = 1
        for (_, input_tensors, output_tensors) in model:
            for input_tensor in input_tensors:
                if input_tensor not in self.tensor_tags:
                    self.tensor_tags[input_tensor] = tensor_tag
                    tensor_tag += 1
            for output_tensor in output_tensors:
                if output_tensor not in self.tensor_tags:
                    self.tensor_tags[output_tensor] = tensor_tag
                    tensor_tag += 1
        for target_tensor_name in sorted(self.target_tensor_names):
            self.tensor_tags[target_tensor_name] = tensor_tag
            tensor_tag += 1
        self.tensor_tags["ack"] = tensor_tag
        tensor_tag += 1

        module_to_stage_map = configuration_maps['module_to_stage_map']
        stage_to_rank_map = configuration_maps['stage_to_rank_map']
        stage_to_depth_map = configuration_maps['stage_to_depth_map']

        if module_to_stage_map is None:
            # If IP addresses not specified, resort to all layers on
            # single machine.
            assert self.rank is None
            self.modules_with_dependencies = ModulesWithDependencies(model)
            self.is_criterion = True
            self.rank_in_stage = 0
            self.num_ranks = 1
            self.num_ranks_in_first_stage = 1
            self.num_ranks_in_previous_stage = 0
            self.num_ranks_in_next_stage = 0
            self.num_stages = 1
            self.num_ranks_in_stage = 1
            self.num_warmup_minibatches = 0
            self.comm_handler = None
        else:
            assert len(module_to_stage_map) == len(model)
            assert self.rank is not None

            stage_to_module_map = collections.defaultdict(list)
            for module in range(len(module_to_stage_map)):
                stage_to_module_map[module_to_stage_map[module]].append(module)

            rank_to_stage_map = {}
            for stage in stage_to_rank_map:
                for rank in stage_to_rank_map[stage]:
                    rank_to_stage_map[rank] = stage

            # Now, use this mapping to determine the modules contained in
            # each stage.
            assert 0 <= self.rank < len(rank_to_stage_map)
            self.num_ranks = len(rank_to_stage_map)
            self.num_stages = len(stage_to_module_map)
            self.stage = rank_to_stage_map[self.rank]
            self.rank_in_stage = stage_to_rank_map[self.stage].index(self.rank)
            self.num_ranks_in_stage = len(stage_to_rank_map[self.stage])
            self.num_ranks_in_first_stage = len(stage_to_rank_map[0])
            self.num_ranks_in_previous_stage = 0
            self.ranks_in_previous_stage = []
            if self.stage > 0:
                self.num_ranks_in_previous_stage = len(
                    stage_to_rank_map[self.stage - 1])
                self.ranks_in_previous_stage = stage_to_rank_map[self.stage - 1]
            self.num_ranks_in_next_stage = 0
            self.ranks_in_next_stage = []
            if self.stage < self.num_stages - 1:
                self.num_ranks_in_next_stage = len(
                    stage_to_rank_map[self.stage + 1])
                self.ranks_in_next_stage = stage_to_rank_map[self.stage + 1]
            modules = stage_to_module_map[self.stage]
            self.modules_with_dependencies = ModulesWithDependencies(
                [model[module] for module in modules])
            self.is_criterion = self.stage == (self.num_stages - 1)
            if stage_to_depth_map is not None:
                self.num_warmup_minibatches = stage_to_depth_map[
                    str(self.stage)]
            else:
                self.num_warmup_minibatches = self.num_ranks - 1
                for i in range(self.stage):
                    self.num_warmup_minibatches -= len(
                        stage_to_rank_map[i])
                self.num_warmup_minibatches = self.num_warmup_minibatches // \
                    self.num_ranks_in_stage

            # To determine where tensors should be sent and received, first
            # determine the "producing" and "consuming" module IDs of each
            # tensor. We then use the corresponding machine ranks to send
            # and receive tensors.
            # master_port = 12345
            # self.comm_handler.master_addr=master_addr
            # self.comm_handler.master_port=master_port
            # self.comm_handler.rank=self.rank
            # self.comm_handler.local_rank=self.local_rank
            # self.comm_handler.num_ranks_in_server=num_ranks_in_server
            # self.comm_handler.world_size=self.num_ranks
            # self.comm_handler.fp16=self.fp16

            for i in range(len(model)-1):
                for tensor_name in model[i][2]:
                    if tensor_name in model[i+1][1]:
                        if module_to_stage_map[i] == \
                            module_to_stage_map[i+1]:
                            continue
                        # For now, assume that each stage is served by only
                        # a single machine.
                        if module_to_stage_map[i+1] == self.stage:
                            self.receive_ranks[tensor_name] = \
                                stage_to_rank_map[module_to_stage_map[i]]
                        if module_to_stage_map[i] == self.stage:
                            self.send_ranks[tensor_name] = \
                                stage_to_rank_map[module_to_stage_map[i+1]]

            for model_inputs in inputs_module_destinations.keys():
                destination_stage = module_to_stage_map[
                    inputs_module_destinations[model_inputs]]
                if destination_stage > self.stage:
                    self.send_ranks[model_inputs] = \
                        self.ranks_in_next_stage

                if 0 < self.stage <= destination_stage:
                    self.receive_ranks[model_inputs] = \
                        self.ranks_in_previous_stage

                if destination_stage > 0:
                    if model_inputs not in self.tensor_tags:
                        self.tensor_tags[model_inputs] = tensor_tag
                        tensor_tag += 1

        modules = self.modules_with_dependencies.modules()
        for i in range(len(modules)):
            modules[i] = modules[i].cuda()
            if self.fp16:
                import apex.fp16_utils as fp16_utils
                modules[i] = fp16_utils.BN_convert_float(modules[i].half())

        # Initialize all groups in the same order on every worker.
        if stage_to_rank_map is not None:
            groups = []
            for stage in range(self.num_stages):
                ranks = stage_to_rank_map[stage]
                if len(ranks) > 1:
                    groups.append(dist.new_group(ranks=ranks))
                else:
                    groups.append(None)
            group = groups[self.stage]
        else:
            group = None

        # self.modules_with_dependencies contains a list of PyTorch
        # modules, along with a list of user-defined input and output
        # tensor names. We use our module_executor.ModuleExecutor
        # class to wrap these dependencies, and use run_forward and
        # run_backward methods downstream.
        num_parameters = 0
        for i in range(len(modules)):
            if group is not None:
                if ((i < (len(modules)-1) and self.is_criterion)
                    or not self.is_criterion):
                    num_parameters += \
                        sum(x.size()[0] * x.size()[1]
                            if len(x.size()) > 1 else x.size()[0]
                            for x in modules[i].parameters() if x.size())
                    modules[i] = torch.nn.parallel.DistributedDataParallel(
                        modules[i],
                        process_group=group,
                        device_ids=[local_rank],
                        output_device=local_rank)
        if self.num_ranks_in_stage > 1:
            module_size = 4. * num_parameters
            print("Replicating stage: ranks=%d, module_size=%.3f" % (
                self.num_ranks_in_stage, module_size))

        if self.fp16:
            self.master_parameters = []
            self.model_parameters = []
            for i in range(len(modules)):
                import apex.fp16_utils as fp16_utils
                module_parameters, module_master_parameters = \
                    fp16_utils.prep_param_lists(modules[i])
                self.master_parameters.extend(module_master_parameters)
                self.model_parameters.extend(module_parameters)
        else:
            self.master_parameters = list(self.parameters())
            self.model_parameters = None

        if self.comm_handler is not None:
            self.comm_handler.initialize(
                self.straggle,
                self.receive_ranks,
                self.send_ranks,
                stage_to_rank_map[self.stage],
                self.tensor_tags,
                self.target_tensor_names,
                self.training_tensor_dtypes,
                self.rank_in_stage,
                self.num_ranks_in_stage,
                self.ranks_in_previous_stage,
                self.ranks_in_next_stage,
                self.batch_size_for_communication)
    def initialize(self, model, inputs_module_destinations,
                   configuration_maps, master_addr, rank,
                   local_rank, num_ranks_in_server):

        self.send_ranks = {}
        self.receive_ranks = {}
        self.rank = rank
        self.local_rank = local_rank
        self.stage = None
        self.tensor_tags = {}
        self.forward_minibatch_id = 0
        self.backward_minibatch_id = 0
        self.criterion_input_name = str(model[-1][1][0])
        self.straggle = torch.ones(self.worker_num_sum, dtype=torch.float)
        tensor_tag = 1
        for (_, input_tensors, output_tensors) in model:
            for input_tensor in input_tensors:
                if input_tensor not in self.tensor_tags:
                    self.tensor_tags[input_tensor] = tensor_tag
                    tensor_tag += 1
            for output_tensor in output_tensors:
                if output_tensor not in self.tensor_tags:
                    self.tensor_tags[output_tensor] = tensor_tag
                    tensor_tag += 1
        for target_tensor_name in sorted(self.target_tensor_names):
            self.tensor_tags[target_tensor_name] = tensor_tag
            tensor_tag += 1
        self.tensor_tags["ack"] = tensor_tag
        tensor_tag += 1

        module_to_stage_map = configuration_maps['module_to_stage_map']
        stage_to_rank_map = configuration_maps['stage_to_rank_map']
        stage_to_depth_map = configuration_maps['stage_to_depth_map']

        if module_to_stage_map is None:
            # If IP addresses not specified, resort to all layers on
            # single machine.
            assert self.rank is None
            self.modules_with_dependencies = ModulesWithDependencies(model)
            self.is_criterion = True
            self.rank_in_stage = 0
            self.num_ranks = 1
            self.num_ranks_in_first_stage = 1
            self.num_ranks_in_previous_stage = 0
            self.num_ranks_in_next_stage = 0
            self.num_stages = 1
            self.num_ranks_in_stage = 1
            self.num_warmup_minibatches = 0
            self.comm_handler = None
        else:
            assert len(module_to_stage_map) == len(model)
            assert self.rank is not None

            stage_to_module_map = collections.defaultdict(list)
            for module in range(len(module_to_stage_map)):
                stage_to_module_map[module_to_stage_map[module]].append(module)

            rank_to_stage_map = {}
            for stage in stage_to_rank_map:
                for rank in stage_to_rank_map[stage]:
                    rank_to_stage_map[rank] = stage

            # Now, use this mapping to determine the modules contained in
            # each stage.
            assert 0 <= self.rank < len(rank_to_stage_map)
            self.num_ranks = len(rank_to_stage_map)
            self.num_stages = len(stage_to_module_map)
            self.stage = rank_to_stage_map[self.rank]
            self.rank_in_stage = stage_to_rank_map[self.stage].index(self.rank)
            self.num_ranks_in_stage = len(stage_to_rank_map[self.stage])
            self.num_ranks_in_first_stage = len(stage_to_rank_map[0])
            self.num_ranks_in_previous_stage = 0
            self.ranks_in_previous_stage = []
            if self.stage > 0:
                self.num_ranks_in_previous_stage = len(
                    stage_to_rank_map[self.stage - 1])
                self.ranks_in_previous_stage = stage_to_rank_map[self.stage - 1]
            self.num_ranks_in_next_stage = 0
            self.ranks_in_next_stage = []
            if self.stage < self.num_stages - 1:
                self.num_ranks_in_next_stage = len(
                    stage_to_rank_map[self.stage + 1])
                self.ranks_in_next_stage = stage_to_rank_map[self.stage + 1]
            modules = stage_to_module_map[self.stage]
            self.modules_with_dependencies = ModulesWithDependencies(
                [model[module] for module in modules])
            self.is_criterion = self.stage == (self.num_stages - 1)
            if stage_to_depth_map is not None:
                self.num_warmup_minibatches = stage_to_depth_map[
                    str(self.stage)]
            else:
                self.num_warmup_minibatches = self.num_ranks - 1
                for i in range(self.stage):
                    self.num_warmup_minibatches -= len(
                        stage_to_rank_map[i])
                self.num_warmup_minibatches = self.num_warmup_minibatches // \
                                              self.num_ranks_in_stage

            # To determine where tensors should be sent and received, first
            # determine the "producing" and "consuming" module IDs of each
            # tensor. We then use the corresponding machine ranks to send
            # and receive tensors.

            for i in range(len(model) - 1):
                for tensor_name in model[i][2]:
                    if tensor_name in model[i + 1][1]:
                        if module_to_stage_map[i] == \
                                module_to_stage_map[i + 1]:
                            continue
                        # For now, assume that each stage is served by only
                        # a single machine.
                        if module_to_stage_map[i + 1] == self.stage:
                            self.receive_ranks[tensor_name] = \
                                stage_to_rank_map[module_to_stage_map[i]]
                        if module_to_stage_map[i] == self.stage:
                            self.send_ranks[tensor_name] = \
                                stage_to_rank_map[module_to_stage_map[i + 1]]

            for model_inputs in inputs_module_destinations.keys():
                destination_stage = module_to_stage_map[
                    inputs_module_destinations[model_inputs]]
                if destination_stage > self.stage:
                    self.send_ranks[model_inputs] = \
                        self.ranks_in_next_stage

                if 0 < self.stage <= destination_stage:
                    self.receive_ranks[model_inputs] = \
                        self.ranks_in_previous_stage

                if destination_stage > 0:
                    if model_inputs not in self.tensor_tags:
                        self.tensor_tags[model_inputs] = tensor_tag
                        tensor_tag += 1

        modules = self.modules_with_dependencies.modules()
        for i in range(len(modules)):
            modules[i] = modules[i].cuda()
            if self.fp16:
                import apex.fp16_utils as fp16_utils
                modules[i] = fp16_utils.BN_convert_float(modules[i].half())
        master_port = 12342
        self.comm_handler = communication_tyf_1.CommunicationHandler(
            master_addr=master_addr,
            master_port=master_port,
            rank=self.rank,
            local_rank=self.local_rank,
            num_ranks_in_server=num_ranks_in_server,
            world_size=self.num_ranks,
            fp16=self.fp16,
            backend=self.distributed_backend
        )
        # Initialize all groups in the same order on every worker.
        if stage_to_rank_map is not None:
            groups = []
            for stage in range(self.num_stages):
                ranks = stage_to_rank_map[stage]
                if len(ranks) > 1:
                    groups.append(dist.new_group(ranks=ranks))
                else:
                    groups.append(None)
            group = groups[self.stage]
        else:
            group = None

        # self.modules_with_dependencies contains a list of PyTorch
        # modules, along with a list of user-defined input and output
        # tensor names. We use our module_executor.ModuleExecutor
        # class to wrap these dependencies, and use run_forward and
        # run_backward methods downstream.
        num_parameters = 0
        for i in range(len(modules)):
            if group is not None:
                if ((i < (len(modules) - 1) and self.is_criterion)
                        or not self.is_criterion):
                    num_parameters += \
                        sum(x.size()[0] * x.size()[1]
                            if len(x.size()) > 1 else x.size()[0]
                            for x in modules[i].parameters() if x.size())
                    modules[i] = torch.nn.parallel.DistributedDataParallel(
                        modules[i],
                        process_group=group,
                        device_ids=[local_rank],
                        output_device=local_rank)
        if self.num_ranks_in_stage > 1:
            module_size = 4. * num_parameters
            print("Replicating stage: ranks=%d, module_size=%.3f" % (
                self.num_ranks_in_stage, module_size))

        if self.fp16:
            self.master_parameters = []
            self.model_parameters = []
            for i in range(len(modules)):
                import apex.fp16_utils as fp16_utils
                module_parameters, module_master_parameters = \
                    fp16_utils.prep_param_lists(modules[i])
                self.master_parameters.extend(module_master_parameters)
                self.model_parameters.extend(module_parameters)
        else:
            self.master_parameters = list(self.parameters())
            self.model_parameters = None

        if self.comm_handler is not None:
            self.comm_handler.initialize(
                torch.ones(self.worker_num_sum, dtype=torch.float),
                self.receive_ranks,
                self.send_ranks,
                stage_to_rank_map[self.stage],
                self.tensor_tags,
                self.target_tensor_names,
                self.training_tensor_dtypes,
                self.rank_in_stage,
                self.num_ranks_in_stage,
                self.ranks_in_previous_stage,
                self.ranks_in_next_stage,
                self.batch_size_for_communication)

    @property
    def target(self):
        return self.tensors[-1]["target"]

    def modules(self):
        return self.modules_with_dependencies.modules()

    def parameters(self):
        parameter_iterators = []
        for module in self.modules_with_dependencies.modules():
            parameter_iterators.append(module.parameters())
        return itertools.chain(*parameter_iterators)

    def state_dict(self):
        state_dict = collections.OrderedDict()
        for i, module in enumerate(self.modules_with_dependencies.modules()):
            state_dict["module%d" % i] = module.state_dict()
        if self.fp16:
            state_dict["master_parameters"] = self.master_parameters
        return state_dict

    def load_state_dict(self, state_dict):
        for i, module in enumerate(self.modules_with_dependencies.modules()):
            module.load_state_dict(state_dict["module%d" % i],strict=False)
        if self.fp16:
            saved_master_parameters = state_dict["master_parameters"]
            for master_parameter, saved_master_parameter in zip(
                self.master_parameters, saved_master_parameters):
                master_parameter.data.copy_(saved_master_parameter.data)

    def cuda(self):
        modules = self.modules_with_dependencies.modules()
        for i in range(len(modules)):
            modules[i] = modules[i].cuda()

    def zero_grad(self):
        modules = self.modules_with_dependencies.modules()
        for i in range(len(modules)):
            modules[i].zero_grad()

    def train(self, num_iterations,epoch):
        self.tensors = []
        self.gradients = {}
        self.tensor_shapes = self.training_tensor_shapes
        self.forward_only = False

        self.forward_minibatch_id = 0
        self.backward_minibatch_id = 0

        if self.comm_handler is not None:
            self.comm_handler.set_tensor_shapes(self.tensor_shapes)
            if epoch == 0:
                self.comm_handler.start_helper_threads(
                num_iterations, forward_only=False)
        modules = self.modules_with_dependencies.modules()
        for i in range(len(modules)):
            modules[i].train()

    def eval(self, num_iterations,epoch):
        self.tensors = []
        self.gradients = {}
        self.tensor_shapes = self.eval_tensor_shapes
        self.tensor_shapes["ack"] = (1,)
        self.forward_only = True

        self.forward_minibatch_id = 0
        self.backward_minibatch_id = 0

        if self.comm_handler is not None:
            self.comm_handler.set_tensor_shapes(self.tensor_shapes)
            # if epoch==0:
            #     self.comm_handler.start_helper_threads(
            #         num_iterations, forward_only=True)

        modules = self.modules_with_dependencies.modules()
        for i in range(len(modules)):
            modules[i].eval()

    def set_loader(self, loader):
        if loader is not None:
            self.loader_iter = iter(loader)
        else:
            self.loader_iter = None

    def set_loss_scale(self, loss_scale):
        self.loss_scale = loss_scale

    def receive_tensors_forward(self,stopped=False):
        if self.forward_only and len(self.tensors) > 0:
            self.tensors.pop(0)
        self.tensors.append({})
        if self.loader_iter is not None:
            input = next(self.loader_iter)
            if self.model_type == TRANSLATION:
                (input, target) = input
                src, src_length = input
                tgt, tgt_length = target

                self.tensors[-1]["input0"] = src.cuda(non_blocking=True)
                self.tensors[-1]["input1"] = torch.LongTensor(src_length).cuda(
                    non_blocking=True)
                self.tensors[-1]["input2"] = tgt[:-1].cuda(non_blocking=True)
                self.tensors[-1]["target"] = tgt[1:].cuda().contiguous().view(-1)
                self.tensors[-1]["target_length"] = \
                    torch.tensor([int(sum(torch.LongTensor(tgt_length) - 1))],
                                 dtype=torch.int).cuda()
            elif self.model_type == BERT:
                input0, input1, input2, target, _ = input
                input2 = input2.unsqueeze(1).unsqueeze(2)
                input2 = input2.to(dtype=torch.float32) # fp16 compatibility
                input2 = (1.0 - input2) * -10000.0
                self.tensors[-1]["input0"] = input0.cuda(non_blocking=True)
                self.tensors[-1]["input1"] = input1.cuda(non_blocking=True)
                self.tensors[-1]["input2"] = input2.cuda(non_blocking=True)
                self.tensors[-1]["target"] = target.cuda(non_blocking=True)
                # self.tensors[-1]["target_"] = target_.cuda(non_blocking=True)

            elif self.model_type == IMAGE_CLASSIFICATION:
                (input, target) = input
                if self.fp16:
                    input = input.half()
                self.tensors[-1]["input0"] = input.cuda(non_blocking=True)
                self.tensors[-1]["target"] = target.cuda(non_blocking=True)
            elif self.model_type == SPEECH_TO_TEXT:
                input, target, input_percentages, target_sizes = input
                input_sizes = input_percentages.mul_(int(input.size(3))).int()
                self.tensors[-1]["input0"] = input.cuda(non_blocking=True)
                self.tensors[-1]["input1"] = input_sizes.cuda(non_blocking=True)
                self.tensors[-1]["target"] = target.cuda(non_blocking=True)
                self.tensors[-1]["target_length"] = target_sizes.cuda(
                    non_blocking=True)
        else:
            # Receive all required tensors from upstream machines.
            for input_name in self.receive_ranks:
                if input_name == "ack":
                    continue

                self.tensors[-1][input_name] = \
                    self.comm_handler.recv(
                        input_name,
                        forward_minibatch_id=self.forward_minibatch_id,
                        backward_minibatch_id=self.backward_minibatch_id,
                        backward=False,
                        stopped=stopped)

                self.forward_stats.stats['receive_tensors_size'] += \
                    (self.tensors[-1][input_name].element_size() *
                     self.tensors[-1][input_name].nelement())

            # Used to track where to receive forward from.
            # self.comm_handler.increment_messaging_index(
            #     sending=False)

    def send_tensors_forward(self):
        # Send all required tensors downstream.
        for output_name in self.send_ranks:
            if output_name == "ack":
                continue
            self.comm_handler.send(
                output_name,
                self.tensors[-1][output_name],
                forward_minibatch_id=self.forward_minibatch_id,
                backward_minibatch_id=self.backward_minibatch_id,
                backward=False)

            self.forward_stats.stats['send_tensors_size'] += \
                (self.tensors[-1][output_name].element_size() *
                 self.tensors[-1][output_name].nelement())

    def receive_tensors_backward(self,stopped=False):
        # Receive all required gradients from downstream
        # machines.
        for output_name in self.send_ranks:
             if output_name in self.target_tensor_names or "input" in output_name:
                continue
             self.gradients[output_name] = \
                self.comm_handler.recv(
                    output_name,
                    forward_minibatch_id=self.forward_minibatch_id,
                    backward_minibatch_id=self.backward_minibatch_id,
                    backward=True,
                    stopped=stopped)

             self.backward_stats.stats['receive_tensors_size'] += \
                 (self.gradients[output_name].element_size() *
                  self.gradients[output_name].nelement())

    def send_tensors_backward(self):
        # Send all required gradients upstream.
        for input_name in self.receive_ranks:
            if input_name in self.target_tensor_names or "input" in input_name:
                continue

            self.comm_handler.send(
                input_name,
                self.gradients[input_name],
                forward_minibatch_id=self.forward_minibatch_id,
                backward_minibatch_id=self.backward_minibatch_id,
                backward=True)

            self.backward_stats.stats['send_tensors_size'] += \
                (self.gradients[input_name].element_size() *
                 self.gradients[input_name].nelement())

        # if self.num_ranks_in_previous_stage > 0:
        #     # Used to track where to send tensors in the
        #     # backward pass.
        #     self.comm_handler.increment_messaging_index(
        #         sending=True)

    def run_forward(self,if_in_warm_up=False,stopped=False):
        """Run forward pass.
        """
        # Receive tensors from previous worker.
        self.receive_tensors_forward(stopped=stopped)
        tensors = self.tensors[-1]

        # Run forward pass.
        time1=time.time()
        self._run_forward(tensors)
        time2=time.time()
        self.time_forward=time2-time1
        # Send tensors forward.
        self.send_tensors_forward()
        if self.verbose_freq > 0 and self.forward_minibatch_id % self.verbose_freq == 0:
            self.forward_stats.print_stats()
        self.forward_stats.reset_stats()
        self.forward_minibatch_id += 1

    def _run_forward(self, tensors):
        # Perform forward pass through model (self.modules_with_dependencies already
        # has modules in topological order).
        modules = self.modules_with_dependencies.modules()
        all_input_names = self.modules_with_dependencies.all_input_names()
        all_output_names = self.modules_with_dependencies.all_output_names()
        for i, (module, input_names, output_names) in \
                enumerate(zip(modules, all_input_names, all_output_names)):
            if i == (len(modules) - 1) and self.is_criterion:
                # If layer is criterion (loss).
                if self.model_type == SPEECH_TO_TEXT:
                    output = tensors["output"].transpose(0, 1).float()
                    output_sizes = tensors["output_sizes"].cpu()
                    target = tensors["target"].cpu()
                    target_sizes = tensors["target_length"].cpu()
                    input0_size = tensors["input0_size"].cpu()
                    module_outputs = [module(output, target, output_sizes, target_sizes) / input0_size[0]]
                elif self.model_type == BERT:
                    output = [tensors[input_name] for input_name in input_names]
                    output.append(tensors["target"])
                    module_outputs = [module(*output)]
                    # module_outputs=[module_outputs]
                    # self.correct=correct
                else:
                    module_outputs = [module(tensors[input_name],
                                             tensors["target"])
                                      for input_name in input_names]
                    module_outputs = [sum(module_outputs)]
            else:
                # If layer is non-criterion.
                module_outputs = module(*[tensors[input_name]
                                          for input_name in input_names])
                if not isinstance(module_outputs, tuple):
                    module_outputs = (module_outputs,)
                module_outputs = list(module_outputs)

            for (output_name, module_output) in zip(output_names, module_outputs):
                tensors[output_name] = module_output

        self.output = tensors[input_names[0]]
        if self.is_criterion and self.model_type == TRANSLATION:
            loss_per_batch = tensors[output_names[0]] * tensors[self.criterion_input_name].size(1)
            loss_per_token = loss_per_batch / tensors["target_length"][0].item()
            self.loss = loss_per_token
        elif self.is_criterion:
            self.loss = tensors[output_names[0]]
        else:
            self.loss = 1
    def run_backward(self,if_in_initial=False,stopped=False):
        # Receive input gradients needed for backward pass.
        if if_in_initial:
            return
        self.receive_tensors_backward(stopped=stopped)
        # Backward pass through modules in reverse order.
        inputs = {}
        outputs = {}
        input_gradients = {}
        output_gradients = {}

        # Get input and output names spanning all modules in this stage.
        all_input_names_set = set()
        all_output_names_set = set()

        modules = self.modules_with_dependencies.modules()
        all_input_names = self.modules_with_dependencies.all_input_names()
        all_output_names = self.modules_with_dependencies.all_output_names()

        for (input_names, output_names) in zip(all_input_names, all_output_names):
            for input_name in input_names:
                all_input_names_set.add(input_name)
            for output_name in output_names:
                all_output_names_set.add(output_name)

        tensors = self.tensors.pop(0)
        # Set inputs, outputs, and output_gradients.
        # Only set outputs/output_gradients for tensors that are not inputs of
        # other modules in this stage.
        # Similarly, only set inputs for tensors that are not outputs of other
        # modules in this stage.
        for (module, input_names, output_names) in \
            zip(reversed(modules), reversed(all_input_names), reversed(all_output_names)):
            for output_name in output_names:
                if output_name not in all_input_names_set:
                    if output_name not in self.gradients:
                        output_gradients[output_name] = None
                    else:
                        output_gradients[output_name] = self.gradients[output_name]
                    if tensors[output_name].requires_grad:
                        outputs[output_name] = tensors[output_name]
            for input_name in input_names:
                if input_name not in all_output_names_set:
                    inputs[input_name] = tensors[input_name]

        # Hook to record input gradients.
        def hook_wrapper(input_name):
            def hook(input_gradient):
                input_gradients[input_name] = input_gradient
            return hook

        for input_name in inputs:
            if input_name != "input0" and input_name != "input1" and input_name != "input2" \
                    and inputs[input_name].requires_grad:
                inputs[input_name].register_hook(hook_wrapper(input_name))

        if "loss" in outputs:
            outputs["loss"] *= self.loss_scale

        # Perform backward pass.
        time1=time.time()
        torch.autograd.backward(tuple([outputs[output_name] for output_name in outputs]),
                                grad_tensors=tuple([output_gradients[output_name]
                                                    for output_name in outputs]))
        time2=time.time()
        self.time_backward=time2-time1
        # Input tensors don't need gradients.
        for input_name in inputs:
            if not inputs[input_name].requires_grad:
                self.gradients[input_name] = inputs[input_name]
                continue

            if input_name != "input0" and input_name != "input1" and input_name != "input2" and input_name != "input":
                self.gradients[input_name] = input_gradients[input_name]

        # Send output gradients.
        self.send_tensors_backward()
        if self.verbose_freq > 0 and self.backward_minibatch_id % self.verbose_freq == 0:
            self.backward_stats.print_stats()
        self.backward_stats.reset_stats()
        self.backward_minibatch_id += 1

    def num_tokens(self):
        return self.tensors[-1]["target_length"][0].item()

    def run_ack(self):
        # No need for ack if running on a single worker.
        if self.rank is None:
            return

        # Receive ack from next stage. Send ack to previous stage.
        if self.stage < (self.num_stages-1):
            self.comm_handler.recv(
                "ack",
                forward_minibatch_id=self.forward_minibatch_id,
                backward_minibatch_id=self.backward_minibatch_id,
                backward=True)
        if self.stage > 0:
            self.comm_handler.send(
                "ack",
                torch.zeros(self.tensor_shapes["ack"],
                            dtype=torch.int64).cuda(),
                forward_minibatch_id=self.forward_minibatch_id,
                backward_minibatch_id=self.backward_minibatch_id,
                backward=True)

            # Used to track where to receive forward from.
            # self.comm_handler.increment_messaging_index(sending=True)

        self.backward_minibatch_id += 1

    def wait(self):
        if self.comm_handler is not None:
            self.comm_handler.wait()

    def num_iterations(self, loader_size):
        """ Determines number of iterations for this stage

        TODO: don't currently support uneven configurations.
        """
        return loader_size
        # if self.stage == 0 or self.stage is None:
        #     return loader_size
        #
        # num_iterations = loader_size * self.num_ranks_in_first_stage
        # assert num_iterations % self.num_ranks_in_stage == 0
        # num_iterations = num_iterations // self.num_ranks_in_stage
        #
        # return num_iterations

    def get_adjusted_learning_rate(self, base_lr):
        return base_lr
        # if self.stage == 0:
        #     return base_lr
        #
        # adjusted_lr = float(base_lr) * float(self.num_ranks_in_stage) \
        #               / float(self.num_ranks_in_first_stage)
        #
        # return adjusted_lr
    def Send_Status(self,tag):
        if self.stage!=self.worker_num_sum-1:
            dist.send(tensor=self.status,dst=self.worker_num_sum-1,tag=tag)
        else:
            return
    def Rec_Status(self,tag):
        if self.stage==self.worker_num_sum-1:
            for i in range(self.worker_num_sum-1):
                for_rec = torch.zeros(self.worker_num_sum,dtype=torch.float)
                dist.recv(tensor=for_rec, src=i, tag=tag)
                self.status[i]=for_rec[i]

        else:
            return
    def Send_Stage_nums(self,tag):
        if self.stage==self.worker_num_sum-1:
            for i in range(self.worker_num_sum-1):
                dist.send(tensor=self.stage_nums, dst=i, tag=tag)
        else:
            return
    def Rec_Stage_nums(self,tag):
        if self.stage!=self.worker_num_sum-1:
            dist.recv(tensor=self.stage_nums, src=self.worker_num_sum-1, tag=tag)
        else:
            return

    def Send_initial(self,tag):
        if self.stage==self.worker_num_sum-1:
            for i in range(self.worker_num_sum-1):
                dist.send(tensor=self.i_for_initial, dst=i, tag=tag)
        else:
            return
    def Rec_initial(self,tag):
        if self.stage!=self.worker_num_sum-1:
            dist.recv(tensor=self.i_for_initial, src=self.worker_num_sum-1, tag=tag)
        else:
            return
    def Send_param(self,filename,dst):
            with open(filename, 'rb') as f:
                data = f.read()
            # 首先，发送数据的大小
            size = torch.tensor([len(data)], dtype=torch.long)
            dist.send(tensor=size, dst=dst)
            # 然后发送数据本身
            buffer = torch.ByteTensor(list(bytearray(data)))
            dist.send(tensor=buffer, dst=dst)
    def Rec_param(self,recv_filename,recv_rank):
        file_size_tensor = torch.tensor([0], dtype=torch.long)
        dist.recv(tensor=file_size_tensor, src=recv_rank)

        file_size = file_size_tensor.item()

        # Preparing tensor for receiving file content
        file_tensor = torch.ByteTensor(file_size)

        # Receiving the file content
        dist.recv(tensor=file_tensor, src=recv_rank)

        # Writing out the received file content
        with open(recv_filename, 'wb') as f:
            f.write(file_tensor.numpy().tobytes())
