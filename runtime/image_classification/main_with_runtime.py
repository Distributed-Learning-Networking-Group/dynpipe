# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data.distributed
import torch.utils.data
import torch.optim
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.nn as nn
from torch.autograd import Variable
import torch
import copy
import time
import sys
import shutil
import os
import json
import importlib
from collections import OrderedDict
import threading
import argparse
import numpy as np
import random
sys.path.append("..")  # nopep8
import runtime
import sgd
from nsys_profile import NsysProgram
from sklearn.ensemble import RandomForestRegressor
import json
import joblib
EVENT = threading.Event()
EVENT1 = threading.Event()

START_TIME = time.time()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(2)
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data_dir', type=str,
                    help='path to dataset')
parser.add_argument('--distributed_backend', type=str,
                    help='distributed backend to use (gloo|nccl)')
parser.add_argument('--module', '-m', required=True,
                    help='name of module that contains model and tensor_shapes definition')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--eval-batch-size', default=100, type=int,
                    help='eval mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_policy', default='step', type=str,
                    help='policy for controlling learning rate')
parser.add_argument('--lr_warmup', action='store_true',
                    help='Warmup learning rate first 5 epochs')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--fp16', action='store_true',
                    help='train model in fp16 precision')
parser.add_argument('--loss_scale', type=float, default=1,
                    help='static loss scale, positive power of 2 to improve fp16 convergence')
parser.add_argument('--master_addr', default=None, type=str,
                    help="IP address of master (machine with rank 0)")
parser.add_argument('--config_path', default=None, type=str,
                    help="Path of configuration file")
parser.add_argument('--no_input_pipelining', action='store_true',
                    help="No pipelining of inputs")
parser.add_argument('--rank', default=None, type=int,
                    help="Rank of worker")
parser.add_argument('--local_rank', default=0, type=int,
                    help="Local rank of worker")
parser.add_argument('--forward_only', action='store_true',
                    help="Run forward pass only")
parser.add_argument('--num_minibatches', default=None, type=int,
                    help="Number of minibatches to run")
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--checkpoint_dir', default='/home/nyt/pipedream/runtime/image_classification/checkpoint/', type=str, metavar='PATH',
                    help='path to directory to save checkpoints')
parser.add_argument('--checkpoint_dir_not_nfs', action='store_true',
                    help='checkpoint dir is not on a shared NFS server')
parser.add_argument('-s', '--synthetic_data', action='store_true',
                    help="Use synthetic data")
parser.add_argument('-v', '--verbose_frequency', default=0, type=int, metavar='N',
                    help="Log verbose information")
parser.add_argument('--num_ranks_in_server', default=1, type=int,
                    help="number of gpus per machine")
parser.add_argument('--partition', default=None, type=str,
                    help="Path of partition configuration file")
parser.add_argument('--worker_num_sum', default=1, type=int,
                    help="number of gpus")
parser.add_argument('--present_stage_id', default=0, type=int,
                    help="stage_id")
parser.add_argument('--use_dynamic', default=False, type=bool)
# Recompute tensors from forward pass, instead of saving them.
parser.add_argument('--recompute', action='store_true',
                    help='Recompute tensors in backward pass')
# Macrobatching reduces the number of weight versions to save,
# by not applying updates every minibatch.
parser.add_argument('--macrobatch', action='store_true',
                    help='Macrobatch updates to save memory')

best_prec1 = 0
batch_list_all = []
forward_list = []
backward_list = []
full_batch_time = []
list12 = []
list34 = []
list56 = []
list78 = []
list910 = []
list_rec = []
list_send = []
# Helper methods.


def is_first_stage():
    return args.stage is None or (args.stage == 0)


def is_last_stage():
    return args.stage is None or (args.stage == (args.num_stages-1))

# Synthetic Dataset class.


class SyntheticDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, input_size, length, num_classes=1000):
        self.tensor = Variable(torch.rand(*input_size)).type(torch.FloatTensor)
        self.target = torch.Tensor(1).random_(0, num_classes)[
            0].type(torch.LongTensor)
        self.length = length

    def __getitem__(self, index):
        return self.tensor, self.target

    def __len__(self):
        return self.length


args = parser.parse_args()

if_restart_mp = False
if_restart_dp = False
all_ranks = [0, 1, 2, 3, 4, 5, 6, 7]
mp_ranks = [0, 1, 2, 3, 4, 5, 6, 7]
dp_ranks = []
dp_nums = []
# 1111


def main():
    global args, best_prec1, if_restart_dp, if_restart_mp
    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss()

    # create s of the model
    module = importlib.import_module(args.module)
    partition = json.load(open(args.partition, 'r'))
    worker_num_sum = args.worker_num_sum

    args.arch = module.arch()
    # model = module.model(criterion)

    args.batch_size_for_communication = partition["batch_size_all"][0]
    args.batch_size = partition["batch_size"][args.present_stage_id]
    # model_vgg=module.model_vgg16(criterion,[3,35], [0,0,0,0])

    # model = module.model_vgg16(criterion, partition["partition"], partition["recompute_ratio"])
    setup_seed(2)
    if 'vgg16' in args.config_path:
        model_input = module.model_vgg16(
            criterion, partition["partition"], partition["recompute_ratio"])
    elif 'resnet50' in args.config_path:
        model_input = module.model_resnet50(
            criterion, partition["partition"], partition["recompute_ratio"])
    elif 'resnet101' in args.config_path:
        model_input = module.model_resnet101(
            criterion, partition["partition"], partition["recompute_ratio"])
    # print("model")
    # print(model)
    # print("model_vgg")
    # print(model_vgg)
    # determine shapes of all tensors in passed-in model
    if args.arch == 'inception_v3':
        input_size = [args.batch_size, 3, 299, 299]
    else:
        input_size = [args.batch_size, 3, 224, 224]
    training_tensor_shapes = {
        "input0": input_size, "target": [args.batch_size]}
    dtypes = {"input0": torch.int64, "target": torch.int64}
    inputs_module_destinations = {"input": 0}
    target_tensor_names = {"target"}

    training_tensor_shapes1 = {
        "input0": input_size, "target": [args.batch_size]}
    dtypes1 = {"input0": torch.float32, "target": torch.int64}
    inputs_module_destinations1 = {"input": 0}

    # for (stage, inputs, outputs) in model[:-1]:  # Skip last layer (loss).
    #
    #     input_tensors = []
    #
    #     for input in inputs:
    #         input_tensor = torch.zeros(tuple(training_tensor_shapes[input]),
    #                                    dtype=torch.float32)
    #         input_tensors.append(input_tensor)
    #     with torch.no_grad():
    #         output_tensors = stage(*tuple(input_tensors))
    #     if not type(output_tensors) is tuple:
    #         output_tensors = [output_tensors]
    #     for output, output_tensor in zip(outputs,
    #                                      list(output_tensors)):
    #         training_tensor_shapes[output] = list(output_tensor.size())
    #         dtypes[output] = output_tensor.dtype
    #
    # eval_tensor_shapes = {}
    # for key in training_tensor_shapes:
    #     eval_tensor_shapes[key] = tuple(
    #         [args.eval_batch_size] + training_tensor_shapes[key][1:])
    #     training_tensor_shapes[key] = tuple(
    #         training_tensor_shapes[key])

    # Skip last layer (loss).
    for module_id, (stage, inputs, outputs) in enumerate(model_input[:-1]):
        # if module_id==0:
        #     checkpoint_file_path = "%scheckpoint.%d.pth.tar" % (args.checkpoint_dir, 0)
        #     assert os.path.isfile(checkpoint_file_path)
        #     print("=> loading checkpoint '{}'".format(checkpoint_file_path))
        #     checkpoint = torch.load(checkpoint_file_path)
        #     stage.load_state_dict(checkpoint['state_dict'], strict=True)
        #     print("=> loaded checkpoint '{}' (epoch {})"
        #           .format(checkpoint_file_path, checkpoint['epoch']))
        # if module_id==1:
        #     checkpoint_file_path = "%scheckpoint.%d.pth.tar" % (args.checkpoint_dir, 1)
        #     assert os.path.isfile(checkpoint_file_path)
        #     print("=> loading checkpoint '{}'".format(checkpoint_file_path))
        #     checkpoint = torch.load(checkpoint_file_path)
        #     stage.load_state_dict(checkpoint['state_dict'], strict=True)
        #     print("=> loaded checkpoint '{}' (epoch {})"
        #           .format(checkpoint_file_path, checkpoint['epoch']))
        input_tensors = []
        for module_input in inputs:
            if module_input in inputs_module_destinations1:
                inputs_module_destinations1[module_input] = module_id

            input_tensor = torch.ones(tuple(training_tensor_shapes1[module_input]),
                                      dtype=dtypes1[module_input])
            input_tensors.append(input_tensor)
        with torch.no_grad():
            output_tensors = stage(*tuple(input_tensors))
        if not type(output_tensors) is tuple:
            output_tensors = [output_tensors]
        for output, output_tensor in zip(outputs,
                                         list(output_tensors)):
            training_tensor_shapes1[output] = list(output_tensor.size())
            dtypes1[output] = output_tensor.dtype

    eval_tensor_shapes = {}
    for key in training_tensor_shapes1:
        eval_tensor_shapes[key] = tuple(
            training_tensor_shapes1[key])
        training_tensor_shapes1[key] = tuple(
            training_tensor_shapes1[key])

    configuration_maps = {
        'module_to_stage_map': None,
        'stage_to_rank_map': None,
        'stage_to_depth_map': None
    }
    if args.config_path is not None:
        json_config_file = json.load(open(args.config_path, 'r'))
        configuration_maps['module_to_stage_map'] = json_config_file.get(
            "module_to_stage_map", None)
        configuration_maps['stage_to_rank_map'] = json_config_file.get(
            "stage_to_rank_map", None)
        configuration_maps['stage_to_rank_map'] = {
            int(k): v for (k, v) in configuration_maps['stage_to_rank_map'].items()}
        configuration_maps['stage_to_depth_map'] = json_config_file.get(
            "stage_to_depth_map", None)
    # print("shape")
    # print(training_tensor_shapes)
    # print(training_tensor_shapes1)
    # if args.present_stage_id==1:
    #     args.loss_scale=float(1/3)
    r = runtime.StageRuntime(
        model=model_input, distributed_backend=args.distributed_backend,
        fp16=args.fp16, loss_scale=args.loss_scale,
        training_tensor_shapes=training_tensor_shapes1,
        eval_tensor_shapes=eval_tensor_shapes,
        training_tensor_dtypes=dtypes1,
        inputs_module_destinations=inputs_module_destinations,
        target_tensor_names=target_tensor_names,
        configuration_maps=configuration_maps,
        master_addr=args.master_addr, rank=args.rank,
        local_rank=args.local_rank,
        num_ranks_in_server=args.num_ranks_in_server,
        verbose_freq=args.verbose_frequency,
        model_type=runtime.IMAGE_CLASSIFICATION,
        event=EVENT,
        event1=EVENT1,
        worker_num_sum=worker_num_sum,
        batch_size=args.batch_size,
        # 总共的batch_size大小，所有的stage该数值相同
        batch_size_for_communication=args.batch_size_for_communication,
        stage_num=len(mp_ranks),
        stage_nums=partition["partition"],
        enable_recompute=args.recompute
    )

    # stage needed to determine if current stage is the first stage
    # num_stages needed to determine if current stage is the last stage
    # num_ranks needed to determine number of warmup_minibatches in case of pipelining
    args.stage = r.stage
    args.num_stages = r.num_stages
    args.num_ranks = r.num_ranks

    # define optimizer
    if args.no_input_pipelining:
        num_versions = 1
    else:
        # number of versions is the total number of machines following the current
        # stage, shared amongst all replicas in this stage
        num_versions = r.num_warmup_minibatches + 1

    # if specified, resume from checkpoint
    if args.resume:
        checkpoint_file_path = "%s.%d.pth.tar" % (args.resume, r.stage)
        assert os.path.isfile(checkpoint_file_path)
        print("=> loading checkpoint '{}'".format(checkpoint_file_path))
        checkpoint = torch.load(checkpoint_file_path)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        r.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(checkpoint_file_path, checkpoint['epoch']))

    optimizer = sgd.SGDWithWeightStashing(r.modules(), r.master_parameters,
                                          r.model_parameters, args.loss_scale,
                                          num_versions=num_versions,
                                          lr=args.lr,
                                          momentum=args.momentum,
                                          weight_decay=args.weight_decay,
                                          verbose_freq=args.verbose_frequency,
                                          macrobatch=args.macrobatch)
    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.arch == 'inception_v3':
        if args.data_dir is None:
            train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                         transform=transforms.Compose([
                                                              transforms.Resize(
                                                                  (224, 224)),
                                                              transforms.ToTensor(),
                                                              normalize,
                                                         ]))
        else:
            traindir = os.path.join(args.data_dir, 'train')
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(299),
                    transforms.ToTensor(),
                    normalize,
                ])
            )
    else:
        import torchvision
        if args.data_dir is None:
            print('using cifar10 dataset')
            train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                         transform=transforms.Compose([
                                                              transforms.Resize(
                                                                  (224, 224)),
                                                              transforms.ToTensor(),
                                                              normalize,
                                                         ]))
        else:
            traindir = os.path.join(args.data_dir, 'train')
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))
    if args.data_dir is None:
        val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                   download=True, transform=transforms.Compose([
                                                        transforms.Resize(
                                                            (224, 224)),
                                                        transforms.ToTensor(),
                                                        normalize,
                                                   ]))
    else:
        valdir = os.path.join(args.data_dir, 'test')
        val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    distributed_sampler = False
    train_sampler = None
    val_sampler = None
    if configuration_maps['stage_to_rank_map'] is not None:
        num_ranks_in_first_stage = len(
            configuration_maps['stage_to_rank_map'][0])
        if num_ranks_in_first_stage > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, num_replicas=num_ranks_in_first_stage,
                rank=args.rank)
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset, num_replicas=num_ranks_in_first_stage,
                rank=args.rank)
            distributed_sampler = True

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(
            train_sampler is None),
        num_workers=4, pin_memory=True, sampler=train_sampler, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=4, pin_memory=True, sampler=val_sampler, drop_last=True)

    # if checkpoint is loaded, start by running validation
    if args.use_dynamic:
        print("in dynamic")

    if args.resume:
        assert args.start_epoch > 0
        validate(val_loader, r, args.start_epoch-1)
    # args.epochs=1

    for epoch in range(args.start_epoch, args.epochs):

        # print(f'if_restart_mp_befor_initialize1:{if_restart_mp}')
        if args.use_dynamic and if_restart_dp:
            r.initialize1(model_input, inputs_module_destinations, configuration_maps,
                          args.master_addr, args.rank, args.local_rank, args.num_ranks_in_server,
                          training_tensor_shapes1, dtypes1)
        # print(f'if_restart_mp_after_initialize1:{if_restart_mp}')
        if args.use_dynamic and if_restart_mp:

            time.sleep(30)
            result = {}
            gpuid = args.local_rank
            program = NsysProgram.from_json("program.json")
            program.args = [
                "profile",
                f"--gpu-metrics-device={gpuid}",
                "--gpu-metrics-frequency=10"
            ]
            program.output = f"out_{gpuid}"
            program.profile_for(5)
            result["profile"] = program.export()
            values = [float(v) for v in result["profile"].values()]
            r.stage_performance[r.stage] = torch.tensor(
                values, dtype=torch.float32)
            r.Send_Stage_performance(epoch)
            r.Rec_Stage_performance(epoch)
            if is_last_stage():
                r.stage_nums = torch.tensor(calculate_new_placement(r.layer_forward_list, r.layer_backward_list, r.layer_communication_list,
                                                                    r.straggle_for_stage_cal, r.stage_num, r.stage_nums, 2, r.stage_performance, dp_nums))
            r.Send_Stage_nums(epoch)
            r.Rec_Stage_nums(epoch)
            # print("partition", r.stage_nums)
            model_vgg = module.model_vgg16(
                criterion, r.stage_nums.numpy().tolist(), partition["recompute_ratio"])
            training_tensor_shapes1 = {
                "input0": input_size, "target": [args.batch_size]}
            dtypes1 = {"input0": torch.float32, "target": torch.int64}
            # Skip last layer (loss).
            for module_id, (stage, inputs, outputs) in enumerate(model_vgg[:-1]):
                input_tensors = []
                for module_input in inputs:
                    if module_input in inputs_module_destinations1:
                        inputs_module_destinations1[module_input] = module_id

                    input_tensor = torch.ones(tuple(training_tensor_shapes1[module_input]),
                                              dtype=dtypes1[module_input])
                    input_tensors.append(input_tensor)
                with torch.no_grad():
                    output_tensors = stage(*tuple(input_tensors))
                if not type(output_tensors) is tuple:
                    output_tensors = [output_tensors]
                for output, output_tensor in zip(outputs,
                                                 list(output_tensors)):
                    training_tensor_shapes1[output] = list(
                        output_tensor.size())
                    dtypes1[output] = output_tensor.dtype

            eval_tensor_shapes = {}
            for key in training_tensor_shapes1:
                eval_tensor_shapes[key] = tuple(
                    training_tensor_shapes1[key])
                training_tensor_shapes1[key] = tuple(
                    training_tensor_shapes1[key])
            r.initialize1(model_vgg, inputs_module_destinations, configuration_maps,
                          args.master_addr, args.rank, args.local_rank, args.num_ranks_in_server,
                          training_tensor_shapes1, dtypes1)
            torch.distributed.barrier()

            for i in range(args.worker_num_sum):
                checkpoint_file_path = "%scheckpoint.%d.pth.tar" % (
                    args.checkpoint_dir, i)
                assert os.path.isfile(checkpoint_file_path)
                print("=> loading checkpoint '{}'".format(checkpoint_file_path))
                checkpoint = torch.load(checkpoint_file_path)
                for j, module in enumerate(r.modules_with_dependencies.modules()):
                    if j > 0:
                        break
                    module.load_state_dict(
                        checkpoint['state_dict']["module0"], strict=False)
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(checkpoint_file_path, checkpoint['epoch']))

            optimizer = sgd.SGDWithWeightStashing(r.modules(), r.master_parameters,
                                                  r.model_parameters, args.loss_scale,
                                                  num_versions=num_versions,
                                                  lr=args.lr,
                                                  momentum=args.momentum,
                                                  weight_decay=args.weight_decay,
                                                  verbose_freq=args.verbose_frequency,
                                                  macrobatch=args.macrobatch)

        if distributed_sampler:
            train_sampler.set_epoch(epoch)

        # train or run forward pass only for one epoch
        if args.forward_only:
            validate(val_loader, r, epoch)
        else:
            n_num = epoch*10
            if_restart_dp = False
            r.restart_type[1] = False
            if_restart_mp = False
            r.restart_type[0] = False
            train(train_loader, r, optimizer, epoch, inputs_module_destinations, configuration_maps,
                  args.master_addr, args.rank, args.local_rank, args.num_ranks_in_server, training_tensor_shapes1,
                  dtypes1, target_tensor_names, n_num, model_input)

            # # evaluate on validation set
            if not if_restart_mp and not if_restart_dp:
                validate(val_loader, r, epoch)
            # remember best prec@1 and save checkpoint

            # if args.checkpoint_dir and should_save_checkpoint:
            #     save_checkpoint({
            #         'epoch': epoch + 1,
            #         'arch': args.arch,
            #         'state_dict': r.state_dict(),
            #         'best_prec1': best_prec1,
            #         'optimizer' : optimizer.state_dict(),
            #     }, args.checkpoint_dir, r.stage)


def train(train_loader, r, optimizer, epoch, inputs_module_destinations, configuration_maps,
          master_addr, rank, local_rank, num_ranks_in_server, training_tensor_shapes1,
          dtypes1, target_tensor_names, n_num, model1):
    global args, best_prec1, if_restart_dp, if_restart_mp
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    Stage_time = AverageMeter_batch()
    i_for_initial = 0
    batch_list = []
    time_for_recieve = 0
    time_for_send = 0
    flag = False
    flag_if_transfer = False
    flag_if_recv = False
    flag_if_save = False
    # switch to train mode
    n = r.num_iterations(loader_size=len(train_loader))
    # n = 1000

    if args.num_minibatches is not None:
        n = args.num_minibatches

    r.train(n, epoch)
    if not is_first_stage():
        train_loader = None
    r.set_loader(train_loader)

    end = time.time()
    epoch_start_time = time.time()

    if args.no_input_pipelining:
        num_warmup_minibatches = 0
    else:
        num_warmup_minibatches = r.num_warmup_minibatches
    print("warm%d" % (num_warmup_minibatches))
    if args.verbose_frequency > 0:
        print("Letting in %d warm-up minibatches" % num_warmup_minibatches)
        print("Running training for %d minibatches" % n)
    batch_begin_time = time.time()
    # start num_warmup_minibatches forward passes
    for i in range(num_warmup_minibatches):
        r.run_forward()
    pre_real = 0.0
    pre_back = 0.0
    for i in range(n - num_warmup_minibatches):
        # perform forward pass
        # if i==100-num_warmup_minibatches and (r.stage==1 or r.stage==2):
        if i == i_for_initial+10-num_warmup_minibatches and i_for_initial > 0:
            print("begin")
            # EVENT.set()
            r.run_forward(stopped=True)
            # perform backward pass
            if args.fp16:
                r.zero_grad()
            else:
                optimizer.zero_grad()
            optimizer.load_old_params()
            if num_warmup_minibatches == 0:
                r.run_backward(stopped=True)
            else:
                r.run_backward()
            optimizer.load_new_params()
            optimizer.step()
            for i in range(num_warmup_minibatches):
                if i == num_warmup_minibatches-1:
                    optimizer.zero_grad()
                    optimizer.load_old_params()
                    r.run_backward(stopped=True)
                    optimizer.load_new_params()
                    optimizer.step()
                else:
                    optimizer.zero_grad()
                    optimizer.load_old_params()
                    r.run_backward()
                    optimizer.load_new_params()
                    optimizer.step()
            r.wait()
            EVENT.clear()
            EVENT1.clear()
            print("end")

            return
        pre_real += r.real_time
        pre_back += r.backward_real_time
        if i == n-num_warmup_minibatches-1:
            r.run_forward(stopped=True)
        else:
            r.run_forward()
        # adjust_learning_rate(optimizer, epoch, args.epochs,
        #                      r, args.lr_policy, i, n)
        if is_last_stage():
            # measure accuracy and record loss
            output, target, loss = r.output, r.target, r.loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), output.size(0))
            top1.update(prec1[0], output.size(0))
            top5.update(prec5[0], output.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            epoch_time = (end - epoch_start_time) / 3600.0
            full_epoch_time = (epoch_time / float(i+1)) * float(n)
            if i % args.print_freq == 0:
                full_batch_time.append(batch_time.val)
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Epoch time [hr]: {epoch_time:.3f} ({full_epoch_time:.3f})\t'
                      'Memory: {memory:.3f} ({cached_memory:.3f})\t'
                      'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1: {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5: {top5.val:.3f} ({top5.avg:.3f})\t'
                      'Running Time: {time:.3f}'.format(
                          epoch, i, n, batch_time=batch_time,
                          epoch_time=epoch_time, full_epoch_time=full_epoch_time,
                          loss=losses, top1=top1, top5=top5,
                          memory=(float(torch.cuda.memory_allocated()) / 10**9),
                          cached_memory=(float(torch.cuda.memory_cached()) / 10**9), time=time.time()-START_TIME))
                import sys
                sys.stdout.flush()
        else:

            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\tMemory: {memory:.3f} ({cached_memory:.3f})'.format(
                    epoch, i, n, memory=(
                        float(torch.cuda.memory_allocated()) / 10**9),
                    cached_memory=(float(torch.cuda.memory_cached()) / 10**9)))
                import sys
                sys.stdout.flush()
        # perform backward pass
        if args.fp16:
            r.zero_grad()
        else:
            optimizer.zero_grad()
        optimizer.load_old_params()
        if i == n-num_warmup_minibatches-1 and num_warmup_minibatches == 0:
            r.run_backward(stopped=True)
        else:
            r.run_backward()

        time_for_recieve += r.time12
        time_for_send += r.time34

        optimizer.load_new_params()
        optimizer.step()
        if args.use_dynamic:
            if i % 50 == 0:
                r.Send_initial(i)
                r.Rec_initial(i)
                r.Send_restart_type(i)
                r.Rec_restart_type(i)
                r.Send_profiles(i)
                r.Rec_profiles(i)
                i_for_initial = int(r.i_for_initial[0])
                if_restart_mp = bool(r.restart_type[0])
                if_restart_dp = bool(r.restart_type[1])
                print("i_for_initial", i_for_initial)
                print("restart_dtype", if_restart_mp, if_restart_dp)
                print("profiles", r.profiles)
                if i_for_initial and not flag_if_save:
                    flag_if_save = True
                    save_checkpoint({
                        'epoch': epoch,
                        'arch': args.arch,
                        'state_dict': r.state_dict(),
                        'best_prec1': best_prec1,
                    }, args.checkpoint_dir, r.stage)
                # recv parameters to gather
                if i_for_initial and not flag_if_recv and is_last_stage():
                    flag_if_recv = True
                    for i in range(len(all_ranks)-1):
                        recv_filename = "checkpoint/checkpoint.%d.pth.tar" % (
                            all_ranks[i])
                        r.Rec_param(recv_filename, all_ranks[i])
                    print("finish gather in last stage")
                # send parameters to decided stages
                if i_for_initial and not flag_if_transfer and is_last_stage():
                    flag_if_transfer = True
                    for i in range(len(all_ranks)-1):
                        for j in range(len(all_ranks)):
                            filename = "checkpoint/checkpoint.%d.pth.tar" % (j)
                            r.Send_param(filename, all_ranks[i])
                    print("finish send in last stage")

                if i_for_initial and not flag_if_transfer and not is_last_stage():
                    flag_if_transfer = True
                    filename = "checkpoint/checkpoint.%d.pth.tar" % (args.rank)
                    r.Send_param(filename, all_ranks[-1])
                    print("finish send self param")
                if i_for_initial and not flag_if_recv and not is_last_stage():
                    flag_if_recv = True
                    for i in range(len(all_ranks)):
                        recv_filename = "checkpoint/checkpoint.%d.pth.tar" % (
                            all_ranks[i])
                        r.Rec_param(recv_filename, all_ranks[-1])
                    print("finish recv all pram")
                r.status[r.stage] = pre_back+pre_real
                if i_for_initial == 0:
                    r.Send_Status(i)
                    r.Rec_Status(i)
                if i == 100:
                    print("finish initialize cmp status")
                    r.initial_status_cmp = r.status.clone()
                print(r.status)
                if is_last_stage():
                    if i > 100:
                        r.straggle_for_stage_cmp = r.status / r.initial_status_cmp
                        r.profiles = r.straggle_for_stage_cmp
                        for profile_index in range(len(r.profiles)):
                            if 0.5 < r.profiles[profile_index] < 2:
                                r.profiles[profile_index] = 1
                        print("straggle_cmp", r.straggle_for_stage_cmp)
                if is_last_stage() and i > 100 and not flag:
                    print("in dynamic last stage")
                    list_index = []

                    def if_exist_straggle(straggle_list):
                        Flag = False
                        for i in range(len(straggle_list)):
                            if straggle_list[i] >= 2 or straggle_list[i] <= 0.5:
                                list_index.append(i)
                                Flag = True
                        return Flag
                    print("list", list_index)
                    if if_exist_straggle(r.straggle_for_stage_cmp.numpy().tolist()):
                        for j in range(len(list_index)):
                            if list_index[j] in dp_ranks:
                                if_restart_dp = True
                            if list_index[j] in mp_ranks:
                                if_restart_mp = True
                        flag = True
                        r.restart_type = torch.tensor(
                            [if_restart_mp, if_restart_dp])
                        r.i_for_initial[0] = torch.tensor([i+60])
                        print("restart")
                pre_real = 0
                pre_back = 0
                time_for_recieve = 0
                time_for_send = 0

    # finish remaining backward passes
    for i in range(num_warmup_minibatches):
        if i == num_warmup_minibatches - 1:
            optimizer.zero_grad()
            optimizer.load_old_params()
            r.run_backward(stopped=True)
            optimizer.load_new_params()
            optimizer.step()
        else:
            optimizer.zero_grad()
            optimizer.load_old_params()
            r.run_backward()
            optimizer.load_new_params()
            optimizer.step()

    # wait for all helper threads to complete
    r.wait()

    print("Epoch %d: %.3f seconds" % (epoch, time.time() - epoch_start_time))
    print("Epoch start time: %.3f, epoch end time: %.3f" %
          (epoch_start_time, time.time()))


def validate(val_loader, r, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    n = r.num_iterations(loader_size=len(val_loader))
    # n = 50
    # if args.num_minibatches is not None:
    # n = args.num_minibatches

    r.eval(n, epoch)
    if not is_first_stage():
        val_loader = None
    r.set_loader(val_loader)

    end = time.time()
    epoch_start_time = time.time()

    if args.no_input_pipelining:
        num_warmup_minibatches = 0
    else:
        num_warmup_minibatches = r.num_warmup_minibatches

    if args.verbose_frequency > 0:
        print("Letting in %d warm-up minibatches" % num_warmup_minibatches)
        print("Running validation for %d minibatches" % n)

    with torch.no_grad():
        # for i in range(num_warmup_minibatches):
        #     r.run_forward()

        for i in range(n):
            # perform forward pass
            if i == n-1:
                r.run_forward(stopped=True)
            else:
                r.run_forward()
            # r.run_ack()

            if is_last_stage():
                output, target, loss = r.output, r.target, r.loss
                # print(r.target)
                # measure accuracy and record loss
                prec1, prec5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), output.size(0))
                top1.update(prec1[0], output.size(0))
                top5.update(prec5[0], output.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    print('Test: [{0}][{1}/{2}]\t'
                          'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Memory: {memory:.3f} ({cached_memory:.3f})\t'
                          'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1: {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5: {top5.val:.3f} ({top5.avg:.3f})'.format(
                              epoch, i, n, batch_time=batch_time, loss=losses,
                              top1=top1, top5=top5,
                              memory=(
                                  float(torch.cuda.memory_allocated()) / 10**9),
                              cached_memory=(float(torch.cuda.memory_cached()) / 10**9)))
                    import sys
                    sys.stdout.flush()

        if is_last_stage():
            print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))

        # for i in range(num_warmup_minibatches):
        #      r.run_ack()

        # wait for all helper threads to complete
        r.wait()

        print('Epoch %d: %.3f seconds' %
              (epoch, time.time() - epoch_start_time))
        print("Epoch start time: %.3f, epoch end time: %.3f" %
              (epoch_start_time, time.time()))

    return top1.avg


def save_checkpoint(state, checkpoint_dir, stage):
    assert os.path.isdir(checkpoint_dir)
    checkpoint_file_path = os.path.join(
        checkpoint_dir, "checkpoint.%d.pth.tar" % stage)
    torch.save(state, checkpoint_file_path)
    print("Saved checkpoint to %s" % checkpoint_file_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AverageMeter_batch(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=10):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, total_epochs, r, lr_policy, step, epoch_length):
    """ Adjusts learning rate based on stage, epoch, and policy.

    Gets learning rate for stage from runtime and adjusts based on policy.

    Supported LR policies:
         - step
         - polynomial decay
         - exponential decay
    """
    stage_base_lr = r.get_adjusted_learning_rate(base_lr=args.lr)

    if args.lr_warmup and epoch < 5:
        lr = stage_base_lr * \
            float(1 + step + epoch*epoch_length)/(5.*epoch_length)

    else:
        if lr_policy == "step":
            lr = stage_base_lr * (0.4 ** (epoch // 10))
        elif lr_policy == "polynomial":
            power = 2.0
            lr = stage_base_lr * \
                ((1.0 - (float(epoch) / float(total_epochs))) ** power)
        elif lr_policy == "exponential_decay":
            decay_rate = 0.97
            lr = stage_base_lr * \
                (decay_rate ** (float(epoch) / float(total_epochs)))
        else:
            raise NotImplementedError

    if step % 100 == 0:
        print("Epoch: %d Step %d \tLearning rate: %f" % (epoch, step, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def runtime_control(layers, stages, num_layer, present_stage_id, start_id, communicaiton, straggle):
    # layers(id,forward_time+backward_time) list
    # stages(id,compute_time) list
    # num_layer(id,num) list
    # start_id 所有stage的起始层id list
    # communication list
    # straggle list
    record = [[0 for _ in range(num_layer[present_stage_id]+1)]
              for _ in range(num_layer[present_stage_id]+1)]

    def compute_pipline_time(stage_list, communication_list):
        if len(stage_list) == 1:
            return stage_list[0]
        for i in range(len(stage_list)-1):
            tmp1 = stage_list[i]
            tmp2 = stage_list[i+1]
            tmp_communication = communication_list[i]
            tmp_time = 0
            if tmp1 == max(tmp1, tmp2):
                if (tmp1)/2 >= (tmp2)/2+tmp_communication:
                    tmp_time = tmp1
                else:
                    tmp_time = tmp_communication-(tmp1-tmp2)/2+tmp1
            if tmp2 == max(tmp1, tmp2):
                if (tmp2)/2 >= (tmp1)/2+tmp_communication:
                    tmp_time = tmp2
                else:
                    tmp_time = tmp_communication-(tmp2-tmp1)/2+tmp2
            stage_list[i+1] = tmp_time
        return stage_list[-1]

    def find_min_index_2d(arr):
        min_value = float('inf')
        min_index = (0, 0)
        for i in range(len(arr)):
            for j in range(len(arr[i])):
                if arr[i][j] < min_value:
                    min_value = arr[i][j]
                    min_index = (i, j)
        return min_index

    def compute_time_1(begin_layer_id, end_layer_id):
        stages_ = copy.copy(stages)
        list_ = []
        communication_list = []
        for i in range(end_layer_id - begin_layer_id):
            stages_[present_stage_id - 1] += layers[start_id[present_stage_id] +
                                                    i]*straggle[present_stage_id-1]
        for i in range(0, present_stage_id):
            list_.append(stages_[i])
            communication_list.append(communicaiton[i])
        max_time = compute_pipline_time(list_, communication_list)
        return max_time

    def compute_time_2(begin_layer_id, end_layer_id):
        stages_ = copy.copy(stages)
        list_ = []
        communication_list = []
        for i in range(end_layer_id - begin_layer_id):
            stages_[present_stage_id + 1] += layers[start_id[present_stage_id] +
                                                    begin_layer_id + i]*straggle[present_stage_id+1]
        for i in range(present_stage_id + 1, 4):
            # change 4 to stage_num_sum
            list_.append(stages_[i])
            communication_list.append(communicaiton[i])
        max_time = compute_pipline_time(list_, communication_list)
        return max_time

    def compute_time_3(begin_layer_id, end_layer_id):
        stages_ = copy.copy(stages)
        stages_[present_stage_id] = 0
        for i in range(end_layer_id - begin_layer_id):
            stages_[present_stage_id] += layers[start_id[present_stage_id] +
                                                begin_layer_id + i]*straggle[present_stage_id]
        return stages_[present_stage_id]
    for i in range(num_layer[present_stage_id]+1):
        for j in range(num_layer[present_stage_id]+1):
            if i >= j:
                record[i][j] = float('inf')
            else:
                if present_stage_id-1 >= 0:
                    time1 = compute_time_1(0, i)  # pre part
                else:
                    if i != 0:
                        record[i][j] = float('inf')
                        continue
                    time1 = 0
                time2 = compute_time_3(i, j)  # medium part f+b
                if present_stage_id+1 <= 3:
                    # change 3 to last stage id
                    time3 = compute_time_2(
                        j, num_layer[present_stage_id])  # last part
                else:
                    if j != num_layer[present_stage_id]:
                        record[i][j] = float('inf')
                        continue
                    time3 = 0
                if time1 == 0:
                    record[i][j] = compute_pipline_time(
                        [time2, time3], [communicaiton[j-1]])
                elif time3 == 0:
                    record[i][j] = compute_pipline_time(
                        [time1, time2], [communicaiton[start_id[present_stage_id]+i-1]])
                else:
                    tmp = compute_pipline_time(
                        [time1, time2], [communicaiton[start_id[present_stage_id]+i-1]])
                    record[i][j] = compute_pipline_time(
                        [tmp, time3], [communicaiton[start_id[present_stage_id]+j-1]])
    min_index = find_min_index_2d(record)
    print(record[min_index[0]][min_index[1]])
    return min_index


def calculate_new_placement(layer_forward_list, layer_backward_list, layer_communication_list, straggle_for_stage, stage_num, stage_nums, top_k, stage_performance, dp_nums):
    def main(stage_num, forward_cost_list, backward_cost_list, comm_cost_list, max_micro_batch_num, cur_micro_batch_num, timestamp):

        # # Total stage number
        # stage_num = 2
        # # Forward cost and backward cost list of each stage
        # forward_cost_list = [1, 1]
        # backward_cost_list = [3, 3]
        # # Communication cost list of each stage. e.g. The first element is the communication cost between stage 1 and stage 2.
        # comm_cost_list = [1]
        #
        # max_micro_batch_num = 99
        # cur_micro_batch_num = 0
        # timestamp = 0

        def get_comm_cost(stage1: float, stage2: float) -> float:
            """计算两个相邻 stage 的通信时间 (stage 顺序不做限制)。

            Args:
                stage1 (int): 前一个 stage 的编号。
                stage2 (int): 后一个 stage 的编号。

            Returns:
                int: _description_
            """
            assert abs(stage1 - stage2) == 1, "The stages must be adjacent!"
            return comm_cost_list[min(stage1, stage2)]

        class Stage:
            def __init__(self, idx: int, forward_cost: float, backward_cost: float) -> None:
                self.idx = idx
                self.forward_cost = forward_cost
                self.backward_cost = backward_cost

                self.warmup_mb_num = -1  # not used yet

                self.f_idx = -1  # index of next forward micro_batch
                self.b_idx = -1  # index of next backward micro_batch

            def __str__(self) -> str:
                return f"stage_idx: {self.idx}\tforward_cost: {self.forward_cost}\tbackward_cost: {self.backward_cost}\tf_idx: {self.f_idx}\tb_idx: {self.b_idx}."

            def get_next_f_idx(self) -> int:
                assert self.b_idx >= 0, "Previous backward index is not initialized!"
                return self.b_idx + (stage_num - self.idx)

            def get_next_b_idx(self) -> int:
                assert self.f_idx >= 0, "Previous forward index is not initialized!"
                return self.f_idx - (stage_num - 1 - self.idx)

            def get_prev_b_idx(self) -> int:
                assert self.f_idx >= 0, "Current forward index is not initialized!"
                return self.f_idx - (stage_num - self.idx)

            def get_prev_f_idx(self) -> int:
                assert self.b_idx >= 0, "Current backward index is not initialized!"
                return self.b_idx + (stage_num - 1 - self.idx)

        def generate_stages():
            stages = [Stage(i, forward_cost_list[i], backward_cost_list[i])
                      for i in range(stage_num)]
            return stages

        stages = generate_stages()

        class MicroBatch:
            def __init__(self, stage_num: int) -> None:
                self.stage_num = stage_num
                self.forward_ts = -1
                self.backward_ts = -1

                self.forward_cost = stages[self.stage_num].forward_cost
                self.backward_cost = stages[self.stage_num].backward_cost

            def __str__(self) -> str:
                return f"stage: {self.stage_num}\tforward_ts: {self.forward_ts}\tbackward_ts: {self.backward_ts}."

        def generate_batches():
            batches = [
                [
                    MicroBatch(stage)
                    for _ in range(max_micro_batch_num)
                ]
                for stage in range(stage_num)
            ]
            return batches

        def get_warmup_micro_batch_num(stage: int) -> int:
            return stage_num - stage - 1

        batches = generate_batches()

        def show_batches(show_diff: bool = True):
            last_forward_ts, last_backward_ts = -1, -1
            for stage_idx in range(stage_num):
                print(f"===== stage {stage_idx} =====")
                print(type(batches[stage_idx]), len(batches[stage_idx]))
                for idx, stage in enumerate(batches[stage_idx]):
                    if idx == len(batches[stage_idx])-stage_num:
                        return
                    if show_diff:
                        if idx % stage_num == 0:
                            if idx != 0:

                                print(
                                    f"mb_index: {idx}\tforward_diff: {stage.forward_ts - last_forward_ts}\tbackward_diff: {stage.backward_ts - last_backward_ts}"
                                )
                            last_forward_ts = stage.forward_ts
                            last_backward_ts = stage.backward_ts
                    else:
                        print(f"mb_index: {idx}", stage)

        def show_batches_time(show_diff: bool = True):
            last_forward_ts, last_backward_ts = -1, -1
            time_sum = 0
            count = 0
            for stage_idx in range(1):
                for idx, stage in enumerate(batches[stage_idx]):
                    if idx == len(batches[stage_idx])-stage_num:
                        return time_sum/(stage_num*count)
                    if show_diff:
                        if idx % stage_num == 0:
                            if idx != 0:
                                count += 1
                                time_sum += stage.forward_ts - last_forward_ts
                            last_forward_ts = stage.forward_ts
                    else:
                        print(f"mb_index: {idx}", stage)

        def show_stages():
            for stage in stages:
                print(stage)

        # * 1. Warmup stage
        for stage in range(stage_num):
            warmup_micro_batch_num = get_warmup_micro_batch_num(stage)
            stages[stage].warmup_mb_num = warmup_micro_batch_num
            if stage != stage_num - 1:
                if stage == 0:
                    batches[stage][0].forward_ts = timestamp

                for i in range(warmup_micro_batch_num):
                    if stage == 0:
                        if i != 0:
                            batches[stage][i].forward_ts = \
                                batches[stage][i - 1].forward_ts + \
                                batches[stage][i - 1].forward_cost
                    else:
                        batches[stage][i].forward_ts = max(
                            batches[stage][i - 1].forward_ts +
                            batches[stage][i - 1].forward_cost,
                            batches[stage - 1][i].forward_ts + batches[stage - 1][i].forward_cost +
                            get_comm_cost(stage - 1, stage)
                        )
                # update f_idx
                stages[stage].f_idx = warmup_micro_batch_num
            else:
                # update f_idx
                stages[stage].f_idx = 0

        # show_batches()
        # show_stages()

        # * 2. Running 1F1B
        is_f_mode = True
        exit_flag = False
        while cur_micro_batch_num <= max_micro_batch_num and (not exit_flag):
            # * Forward pass
            if is_f_mode:
                for stage_idx in range(stage_num):
                    stage = stages[stage_idx]

                    prev_b_idx = stage.get_prev_b_idx()
                    prev_backward_finish_ts = -1 if prev_b_idx < 0 else \
                        batches[stage_idx][prev_b_idx].backward_ts + \
                        batches[stage_idx][prev_b_idx].backward_cost

                    if stage_idx == 0:
                        batches[stage_idx][stage.f_idx].forward_ts = max(
                            batches[stage_idx][stage.f_idx - 1].forward_ts +
                            batches[stage_idx][stage.f_idx -
                                               1].forward_cost, prev_backward_finish_ts
                        )
                    elif stage_idx == stage_num - 1:
                        batches[stage_idx][stage.f_idx].forward_ts = max(
                            batches[stage_idx - 1][stage.f_idx].forward_ts +
                            batches[stage_idx - 1][stage.f_idx].forward_cost +
                            get_comm_cost(stage_idx - 1, stage_idx),
                            prev_backward_finish_ts
                        )
                    else:
                        batches[stage_idx][stage.f_idx].forward_ts = max(
                            prev_backward_finish_ts,
                            max(
                                batches[stage_idx][stage.f_idx - 1].forward_ts +
                                batches[stage_idx][stage.f_idx -
                                                   1].forward_cost,
                                batches[stage_idx - 1][stage.f_idx].forward_ts +
                                batches[stage_idx - 1][stage.f_idx].forward_cost +
                                get_comm_cost(stage_idx - 1, stage_idx)
                            )
                        )
                    # update cur_micro_batch_num
                    cur_micro_batch_num = max(cur_micro_batch_num, stage.f_idx)
                    # update b_idx
                    next_b_idx = stage.get_next_b_idx()
                    # print(f"next_b_idx: {next_b_idx}")
                    stage.b_idx = next_b_idx
                    if stage.b_idx >= max_micro_batch_num:
                        exit_flag = True
                        break

            # * Backward pass
            else:
                for stage_idx in range(stage_num - 1, -1, -1):
                    stage = stages[stage_idx]

                    prev_f_idx = stage.get_prev_f_idx()
                    prev_forward_finish_ts = -1 if prev_f_idx < 0 else \
                        batches[stage_idx][prev_f_idx].forward_ts + \
                        batches[stage_idx][prev_f_idx].forward_cost

                    if stage_idx == 0:
                        batches[stage_idx][stage.b_idx].backward_ts = max(
                            prev_forward_finish_ts,
                            batches[stage_idx + 1][stage.b_idx].backward_ts + batches[stage_idx +
                                                                                      1][stage.b_idx].backward_cost + get_comm_cost(stage_idx, stage_idx + 1)
                        )
                    elif stage_idx == stage_num - 1:
                        batches[stage_idx][stage.b_idx].backward_ts = prev_forward_finish_ts
                    else:
                        batches[stage_idx][stage.b_idx].backward_ts = max(
                            prev_forward_finish_ts,
                            batches[stage_idx + 1][stage.b_idx].backward_ts + batches[stage_idx +
                                                                                      1][stage.b_idx].backward_cost + get_comm_cost(stage_idx, stage_idx + 1)
                        )
                    # update cur_micro_batch_num
                    cur_micro_batch_num = max(cur_micro_batch_num, stage.b_idx)
                    # update f_idx
                    next_f_idx = stage.get_next_f_idx()
                    # print(f"prev_f_idx: {stage.f_idx}, next_f_idx: {next_f_idx}")
                    stage.f_idx = next_f_idx
                    if stage.f_idx >= max_micro_batch_num:
                        exit_flag = True
                        break

            # change mode
            is_f_mode = not is_f_mode

        a = show_batches_time()
        return a
    # print(main(3,[1,9,1],[1,3,3],[0,0],99,0,0))
    import heapq
    import random
    random.seed(0)
    # layer_forward_list=[random.uniform(5, 10) for _ in range(40)]
    # layer_backward_list=[random.uniform(5, 10) for _ in range(40)]
    # layer_communication_list=[random.uniform(1, 3) for _ in range(39)]

    layer_communication_list_ = layer_communication_list.copy()
    layer_communication_list_.insert(0, layer_communication_list[0])
    layer_process_time = [
        x + y+z for x, y, z in zip(layer_forward_list, layer_backward_list, layer_communication_list_)]

    for i in range(len(layer_process_time)):
        if i == 0 or i == len(layer_process_time)-1:
            continue
        layer_process_time[i] += layer_communication_list_[i+1]
    # straggle_for_stage=[1,1,1,1] #
    # stage_num=4
    # stage_nums=[6,14,12,8] #
    layer_forward_list_new = []
    layer_backward_list_new = []
    layer_communication_list_new = []
    # top_k=int(min(stage_nums)/10)
    # top_k=6  #super

    def top_k_max_indices(lst, start, end, k):
        # 使用heapq模块的nlargest函数找到前k个最大值的索引
        max_indices = heapq.nlargest(k, range(start, end), key=lst.__getitem__)
        max_indices.sort()
        return max_indices

    def calculate_stage_end_indices(stage_sizes):
        # 计算每个阶段的结束index
        end_indices = [sum(stage_sizes[:i+1]) for i in range(len(stage_sizes))]
        return end_indices

    stage_index = calculate_stage_end_indices(stage_nums)
    stage_index.insert(0, 0)
    print("stage_index", stage_index)
    max_indexes = []

    for i in range(1, len(stage_index)):
        max_index = top_k_max_indices(
            layer_process_time, stage_index[i-1], stage_index[i], top_k)
        max_index = [x+1 for x in max_index]
        max_indexes += max_index
    max_indexes.insert(0, 0)
    max_indexes.append(len(layer_forward_list))

    for i in range(1, len(max_indexes)-1):
        layer_forward_list_new.append(
            sum(layer_forward_list[max_indexes[i-1]:max_indexes[i]]))
        layer_backward_list_new.append(
            sum(layer_backward_list[max_indexes[i-1]:max_indexes[i]]))

    for i in range(1, len(max_indexes)-2):
        layer_communication_list_new.append(
            layer_communication_list[max_indexes[i]-1])

    if (max_indexes[-2]-1) != len(layer_forward_list)-1:
        layer_forward_list_new[-1] += sum(
            layer_forward_list[max_indexes[-2]:len(layer_forward_list)])
        layer_backward_list_new[-1] += sum(
            layer_backward_list[max_indexes[-2]:len(layer_forward_list)])
    # print(max_indexes)
    # print(layer_forward_list_new)
    # print(layer_backward_list_new)
    # print(layer_communication_list_new)
    import numpy as np
    import time
    time_begin = time.time()
    from itertools import combinations
    record = []
    items = list(range(1, len(layer_forward_list_new)))
    print(
        f'a input items:{items} \nstage_num:{stage_num-1}\n layer_forward_list_new{layer_forward_list_new}')

    a = list(combinations(items, stage_num - 1))
    import joblib
    import numpy as np
    rf_regressor = joblib.load('rf.pkl')

    def calculate_straggle(stage_performence, stage_information):
        straggles = torch.ones(len(stage_performence), dtype=float)
        for i in range(len(stage_performence)):
            if stage_performence[i][0] == 0:
                pass
            else:
                straggles[i] = torch.from_numpy(
                    rf_regressor.predict(np.array([list(stage_performence[i]) + list(stage_information[i])])))
        return straggles

    def get_stage_info(stage_nums):
        with open('data.npy', 'rb') as f:
            saved_stage_info_list = np.load(f)
        # insert all zeros vector at the beginning to enable cumsum calculation
        saved_stage_info_list = np.insert(
            saved_stage_info_list, 0, np.zeros(saved_stage_info_list[0].shape), axis=0)
        saved_stage_info_list = np.array(saved_stage_info_list, dtype=int)

        # * return message
        query_stage_info_list = []

        cumsum_stage_nums = np.cumsum(stage_nums)
        for idx in range(len(cumsum_stage_nums) - 1):
            if idx == 0:
                query_stage_info_list.append(
                    saved_stage_info_list[cumsum_stage_nums[idx]][0]
                )
            else:
                query_stage_info_list.append(
                    saved_stage_info_list[cumsum_stage_nums[idx]][0] -
                    saved_stage_info_list[cumsum_stage_nums[idx - 1]][0]
                )
        query_stage_info_list.append(
            saved_stage_info_list[-1][0] -
            saved_stage_info_list[cumsum_stage_nums[-2]][0]
        )
        return query_stage_info_list

    for i in a:
        new_stage_nums = []
        # print(i)
        for i_ in range(len(i) + 1):
            if i_ == 0:
                new_stage_nums.append(max_indexes[i[0]])
            elif i_ == len(i):
                new_stage_nums.append(
                    len(layer_forward_list) - max_indexes[i[i_ - 1]])
            else:
                new_stage_nums.append(
                    max_indexes[i[i_]] - max_indexes[i[i_ - 1]])
        # print("in repartition alg", new_stage_nums)
        stage_information = get_stage_info(new_stage_nums)
        straggle_for_stage_ = calculate_straggle(
            stage_performance, stage_information)
        layer_communication_list_new_ = []
        present_stage_forward = []
        present_stage_backward = []
        for j in range(len(i) + 1):
            if j == 0:
                present_stage_forward.append(
                    straggle_for_stage_[j]*sum(layer_forward_list_new[0:i[j]]))
            elif j == len(i):
                present_stage_forward.append(straggle_for_stage_[
                                             j]*sum(layer_forward_list_new[i[j - 1]:len(layer_backward_list_new)]))
            else:
                present_stage_forward.append(
                    straggle_for_stage_[j]*sum(layer_forward_list_new[i[j - 1]:i[j]]))
        for j in range(len(i) + 1):
            if j == 0:
                present_stage_backward.append(
                    straggle_for_stage_[j]*sum(layer_backward_list_new[0:i[j]]))
            elif j == len(i):
                present_stage_backward.append(straggle_for_stage_[
                                              j]*sum(layer_backward_list_new[i[j - 1]:len(layer_backward_list_new)]))
            else:
                present_stage_backward.append(
                    straggle_for_stage_[j]*sum(layer_backward_list_new[i[j - 1]:i[j]]))
        for j in range(len(i)):
            layer_communication_list_new_.append(
                layer_communication_list_new[i[j] - 1])
        record.append(main(stage_num, present_stage_forward,
                      present_stage_backward, layer_communication_list_new_, 99, 0, 0))

    # flat_index_of_min = np.argmin(record)
    # 将扁平化索引转换为多维索引
    # min_index = np.unravel_index(flat_index_of_min, record.shape)

    min_value_index = record.index(min(record))
    min_index = a[min_value_index]

    new_stage_nums = []
    # new_stage_nums.append(max_indexes[min_index[0]])
    # for i in range(1,stage_num-1):
    #     new_stage_nums.append(max_indexes[min_index[i]]-max_indexes[min_index[i-1]])
    # new_stage_nums.append(len(layer_forward_list)-max_indexes[min_index[stage_num-2]])
    # new_stage_nums=[max_indexes[min_index[0]],max_indexes[min_index[1]]-max_indexesmin_index[0]],
    #                 max_indexes[min_index[2]]-max_indexes[min_index[1]],len(layer_forward_list)-max_indexes[min_index[2]]]

    # new_stage_nums = [max_indexes[min_index[0]], len(
    #     layer_forward_list)-max_indexes[min_index[0]]]

    for i in range(len(min_index) + 1):
        if i == 0:
            new_stage_nums.append(max_indexes[min_index[0]])
        elif i == len(min_index):
            new_stage_nums.append(
                len(layer_forward_list) - max_indexes[min_index[i - 1]])
        else:
            new_stage_nums.append(
                max_indexes[min_index[i]] - max_indexes[min_index[i - 1]])

    print("rearange", new_stage_nums)
    new_stage_nums = dp_nums+new_stage_nums
    return new_stage_nums


# def calculate_new_placement(layer_forward_list,layer_backward_list,layer_communication_list,straggle_for_stage,stage_num,stage_nums,top_k):
#     def main(stage_num,forward_cost_list,backward_cost_list,comm_cost_list,max_micro_batch_num,cur_micro_batch_num,timestamp):
#
#         # # Total stage number
#         # stage_num = 2
#         # # Forward cost and backward cost list of each stage
#         # forward_cost_list = [1, 1]
#         # backward_cost_list = [3, 3]
#         # # Communication cost list of each stage. e.g. The first element is the communication cost between stage 1 and stage 2.
#         # comm_cost_list = [1]
#         #
#         # max_micro_batch_num = 99
#         # cur_micro_batch_num = 0
#         # timestamp = 0
#
#         def get_comm_cost(stage1: float, stage2: float) -> float:
#             """计算两个相邻 stage 的通信时间 (stage 顺序不做限制)。
#
#             Args:
#                 stage1 (int): 前一个 stage 的编号。
#                 stage2 (int): 后一个 stage 的编号。
#
#             Returns:
#                 int: _description_
#             """
#             assert abs(stage1 - stage2) == 1, "The stages must be adjacent!"
#             return comm_cost_list[min(stage1, stage2)]
#
#         class Stage:
#             def __init__(self, idx: int, forward_cost: float, backward_cost: float) -> None:
#                 self.idx = idx
#                 self.forward_cost = forward_cost
#                 self.backward_cost = backward_cost
#
#                 self.warmup_mb_num = -1  # not used yet
#
#                 self.f_idx = -1  # index of next forward micro_batch
#                 self.b_idx = -1  # index of next backward micro_batch
#
#             def __str__(self) -> str:
#                 return f"stage_idx: {self.idx}\tforward_cost: {self.forward_cost}\tbackward_cost: {self.backward_cost}\tf_idx: {self.f_idx}\tb_idx: {self.b_idx}."
#
#             def get_next_f_idx(self) -> int:
#                 assert self.b_idx >= 0, "Previous backward index is not initialized!"
#                 return self.b_idx + (stage_num - self.idx)
#
#             def get_next_b_idx(self) -> int:
#                 assert self.f_idx >= 0, "Previous forward index is not initialized!"
#                 return self.f_idx - (stage_num - 1 - self.idx)
#
#             def get_prev_b_idx(self) -> int:
#                 assert self.f_idx >= 0, "Current forward index is not initialized!"
#                 return self.f_idx - (stage_num - self.idx)
#
#             def get_prev_f_idx(self) -> int:
#                 assert self.b_idx >= 0, "Current backward index is not initialized!"
#                 return self.b_idx + (stage_num - 1 - self.idx)
#
#         def generate_stages():
#             stages = [Stage(i, forward_cost_list[i], backward_cost_list[i]) for i in range(stage_num)]
#             return stages
#
#         stages = generate_stages()
#
#         class MicroBatch:
#             def __init__(self, stage_num: int) -> None:
#                 self.stage_num = stage_num
#                 self.forward_ts = -1
#                 self.backward_ts = -1
#
#                 self.forward_cost = stages[self.stage_num].forward_cost
#                 self.backward_cost = stages[self.stage_num].backward_cost
#
#             def __str__(self) -> str:
#                 return f"stage: {self.stage_num}\tforward_ts: {self.forward_ts}\tbackward_ts: {self.backward_ts}."
#
#         def generate_batches():
#             batches = [
#                 [
#                     MicroBatch(stage)
#                     for _ in range(max_micro_batch_num)
#                 ] \
#                 for stage in range(stage_num)
#             ]
#             return batches
#
#         def get_warmup_micro_batch_num(stage: int) -> int:
#             return stage_num - stage - 1
#
#         batches = generate_batches()
#
#         def show_batches(show_diff: bool = True):
#             last_forward_ts, last_backward_ts = -1, -1
#             for stage_idx in range(stage_num):
#                 print(f"===== stage {stage_idx} =====")
#                 print(type(batches[stage_idx]),len(batches[stage_idx]))
#                 for idx, stage in enumerate(batches[stage_idx]):
#                     if idx==len(batches[stage_idx])-stage_num:
#                         return
#                     if show_diff:
#                         if idx % stage_num == 0:
#                             if idx != 0:
#
#                                 print(
#                                     f"mb_index: {idx}\tforward_diff: {stage.forward_ts - last_forward_ts}\tbackward_diff: {stage.backward_ts - last_backward_ts}"
#                                 )
#                             last_forward_ts = stage.forward_ts
#                             last_backward_ts = stage.backward_ts
#                     else:
#                         print(f"mb_index: {idx}", stage)
#
#         def show_batches_time(show_diff: bool = True):
#             last_forward_ts, last_backward_ts = -1, -1
#             time_sum=0
#             count=0
#             for stage_idx in range(1):
#                 for idx, stage in enumerate(batches[stage_idx]):
#                     if idx==len(batches[stage_idx])-stage_num:
#                         return time_sum/(stage_num*count)
#                     if show_diff:
#                         if idx % stage_num == 0:
#                             if idx != 0:
#                                 count+=1
#                                 time_sum+=stage.forward_ts - last_forward_ts
#                             last_forward_ts = stage.forward_ts
#                     else:
#                         print(f"mb_index: {idx}", stage)
#
#         def show_stages():
#             for stage in stages:
#                 print(stage)
#
#         # * 1. Warmup stage
#         for stage in range(stage_num):
#             warmup_micro_batch_num = get_warmup_micro_batch_num(stage)
#             stages[stage].warmup_mb_num = warmup_micro_batch_num
#             if stage != stage_num - 1:
#                 if stage == 0:
#                     batches[stage][0].forward_ts = timestamp
#
#                 for i in range(warmup_micro_batch_num):
#                     if stage == 0:
#                         if i != 0:
#                             batches[stage][i].forward_ts = \
#                                 batches[stage][i - 1].forward_ts + batches[stage][i - 1].forward_cost
#                     else:
#                         batches[stage][i].forward_ts = max(
#                             batches[stage][i - 1].forward_ts + batches[stage][i - 1].forward_cost,
#                             batches[stage - 1][i].forward_ts + batches[stage - 1][i].forward_cost +
#                             get_comm_cost(stage - 1, stage)
#                         )
#                 # update f_idx
#                 stages[stage].f_idx = warmup_micro_batch_num
#             else:
#                 # update f_idx
#                 stages[stage].f_idx = 0
#
#         # show_batches()
#         # show_stages()
#
#         # * 2. Running 1F1B
#         is_f_mode = True
#         exit_flag = False
#         while cur_micro_batch_num <= max_micro_batch_num and (not exit_flag):
#             # * Forward pass
#             if is_f_mode:
#                 for stage_idx in range(stage_num):
#                     stage = stages[stage_idx]
#
#                     prev_b_idx = stage.get_prev_b_idx()
#                     prev_backward_finish_ts = -1 if prev_b_idx < 0 else \
#                         batches[stage_idx][prev_b_idx].backward_ts + batches[stage_idx][prev_b_idx].backward_cost
#
#                     if stage_idx == 0:
#                         batches[stage_idx][stage.f_idx].forward_ts = max(
#                             batches[stage_idx][stage.f_idx - 1].forward_ts +
#                             batches[stage_idx][stage.f_idx - 1].forward_cost, prev_backward_finish_ts
#                         )
#                     elif stage_idx == stage_num - 1:
#                         batches[stage_idx][stage.f_idx].forward_ts = max(
#                             batches[stage_idx - 1][stage.f_idx].forward_ts +
#                             batches[stage_idx - 1][stage.f_idx].forward_cost + get_comm_cost(stage_idx - 1, stage_idx),
#                             prev_backward_finish_ts
#                         )
#                     else:
#                         batches[stage_idx][stage.f_idx].forward_ts = max(
#                             prev_backward_finish_ts,
#                             max(
#                                 batches[stage_idx][stage.f_idx - 1].forward_ts +
#                                 batches[stage_idx][stage.f_idx - 1].forward_cost,
#                                 batches[stage_idx - 1][stage.f_idx].forward_ts +
#                                 batches[stage_idx - 1][stage.f_idx].forward_cost + get_comm_cost(stage_idx - 1, stage_idx)
#                             )
#                         )
#                     # update cur_micro_batch_num
#                     cur_micro_batch_num = max(cur_micro_batch_num, stage.f_idx)
#                     # update b_idx
#                     next_b_idx = stage.get_next_b_idx()
#                     # print(f"next_b_idx: {next_b_idx}")
#                     stage.b_idx = next_b_idx
#                     if stage.b_idx >= max_micro_batch_num:
#                         exit_flag = True
#                         break
#
#             # * Backward pass
#             else:
#                 for stage_idx in range(stage_num - 1, -1, -1):
#                     stage = stages[stage_idx]
#
#                     prev_f_idx = stage.get_prev_f_idx()
#                     prev_forward_finish_ts = -1 if prev_f_idx < 0 else \
#                         batches[stage_idx][prev_f_idx].forward_ts + batches[stage_idx][prev_f_idx].forward_cost
#
#                     if stage_idx == 0:
#                         batches[stage_idx][stage.b_idx].backward_ts = max(
#                             prev_forward_finish_ts,\
#                             batches[stage_idx + 1][stage.b_idx].backward_ts + batches[stage_idx + 1][stage.b_idx].backward_cost + get_comm_cost(stage_idx, stage_idx + 1)
#                         )
#                     elif stage_idx == stage_num - 1:
#                         batches[stage_idx][stage.b_idx].backward_ts = prev_forward_finish_ts
#                     else:
#                         batches[stage_idx][stage.b_idx].backward_ts = max(
#                             prev_forward_finish_ts,\
#                             batches[stage_idx + 1][stage.b_idx].backward_ts + batches[stage_idx + 1][stage.b_idx].backward_cost + get_comm_cost(stage_idx, stage_idx + 1)
#                         )
#                     # update cur_micro_batch_num
#                     cur_micro_batch_num = max(cur_micro_batch_num, stage.b_idx)
#                     # update f_idx
#                     next_f_idx = stage.get_next_f_idx()
#                     # print(f"prev_f_idx: {stage.f_idx}, next_f_idx: {next_f_idx}")
#                     stage.f_idx = next_f_idx
#                     if stage.f_idx >= max_micro_batch_num:
#                         exit_flag = True
#                         break
#
#             # change mode
#             is_f_mode = not is_f_mode
#
#         a=show_batches_time()
#         return a
#     # print(main(3,[1,9,1],[1,3,3],[0,0],99,0,0))
#     import heapq
#     import random
#     random.seed(0)
#     # layer_forward_list=[random.uniform(5, 10) for _ in range(40)]
#     # layer_backward_list=[random.uniform(5, 10) for _ in range(40)]
#     # layer_communication_list=[random.uniform(1, 3) for _ in range(39)]
#
#
#     layer_communication_list_=layer_communication_list.copy()
#     layer_communication_list_.insert(0,layer_communication_list[0])
#     layer_process_time=[x + y+z for x, y,z in zip(layer_forward_list, layer_backward_list,layer_communication_list_)]
#
#     for i in range(len(layer_process_time)):
#         if i==0 or i==len(layer_process_time)-1:
#             continue
#         layer_process_time[i]+=layer_communication_list_[i+1]
#     # straggle_for_stage=[1,1,1,1] #
#     # stage_num=4
#     # stage_nums=[6,14,12,8] #
#     layer_forward_list_new=[]
#     layer_backward_list_new=[]
#     layer_communication_list_new=[]
#     # top_k=int(min(stage_nums)/10)
#     # top_k=6  #super
#
#     def top_k_max_indices(lst, start,end,k):
#         # 使用heapq模块的nlargest函数找到前k个最大值的索引
#         max_indices = heapq.nlargest(k, range(start,end), key=lst.__getitem__)
#         max_indices.sort()
#         return max_indices
#     def calculate_stage_end_indices(stage_sizes):
#         # 计算每个阶段的结束index
#         end_indices = [sum(stage_sizes[:i+1]) for i in range(len(stage_sizes))]
#         return end_indices
#     stage_index=calculate_stage_end_indices(stage_nums)
#     stage_index.insert(0,0)
#     print("stage_index",stage_index)
#     max_indexes=[]
#     for i in range(1,len(stage_index)):
#         max_index=top_k_max_indices(layer_process_time,stage_index[i-1],stage_index[i],top_k)
#         max_index=[x+1 for x in max_index]
#         max_indexes+=max_index
#     max_indexes.insert(0,0)
#     max_indexes.append(len(layer_forward_list))
#     for i in range(1,len(max_indexes)-1):
#         layer_forward_list_new.append(sum(layer_forward_list[max_indexes[i-1]:max_indexes[i]]))
#         layer_backward_list_new.append(sum(layer_backward_list[max_indexes[i-1]:max_indexes[i]]))
#     for i in range(1,len(max_indexes)-2):
#         layer_communication_list_new.append(layer_communication_list[max_indexes[i]-1])
#     if (max_indexes[-2]-1)!=len(layer_forward_list)-1:
#         layer_forward_list_new[-1]+=sum(layer_forward_list[max_indexes[-2]:len(layer_forward_list)])
#         layer_backward_list_new[-1]+=sum(layer_backward_list[max_indexes[-2]:len(layer_forward_list)])
#     # print(max_indexes)
#     # print(layer_forward_list_new)
#     # print(layer_backward_list_new)
#     # print(layer_communication_list_new)
#     import numpy as np
#     import time
#     time_begin=time.time()
#
#     #for test
#     # layer_forward_list_new=layer_forward_list
#     # layer_communication_list_new=layer_communication_list
#     # layer_backward_list_new=layer_backward_list
#
#     record=np.full((len(layer_forward_list_new),)*(stage_num-1),np.inf)
#     for i in range(1,len(layer_forward_list_new)):
#         for j in range(1,len(layer_forward_list_new)):
#             for k in range(1,len(layer_forward_list_new)):
#                 if i<j<k:
#                     print(i,j,k)
#                     present_stage_forward=[]
#                     present_stage_backward=[]
#                     present_stage_forward.append(straggle_for_stage[0]*sum(layer_forward_list_new[0:i]))
#                     present_stage_forward.append(straggle_for_stage[1]*sum(layer_forward_list_new[i: j]))
#                     present_stage_forward.append(straggle_for_stage[2]*sum(layer_forward_list_new[j:k]))
#                     present_stage_forward.append(straggle_for_stage[3]*sum(layer_forward_list_new[k:len(layer_forward_list_new)]))
#
#                     present_stage_backward.append(straggle_for_stage[0]*sum(layer_backward_list_new[0:i]))
#                     present_stage_backward.append(straggle_for_stage[1]*sum(layer_backward_list_new[i:j]))
#                     present_stage_backward.append(straggle_for_stage[2]*sum(layer_backward_list_new[j:k]))
#                     present_stage_backward.append(straggle_for_stage[3]*sum(layer_backward_list_new[k:len(layer_forward_list_new)]))
#                     record[i][j][k]=main(stage_num,present_stage_forward,present_stage_backward,layer_communication_list_new,99,0,0)
#                     # print(present_stage_forward)
#                     # print(present_stage_backward)
#                 else:
#                     continue
#
#     flat_index_of_min = np.argmin(record)
#     # 将扁平化索引转换为多维索引
#     min_index = np.unravel_index(flat_index_of_min, record.shape)
#     print("最小值的下标:", min_index)
#     print("final result",record[min_index[0]][min_index[1]][min_index[2]])
#     print("time",time.time()-time_begin)
#     new_stage_nums=[]
#     new_stage_nums.append(max_indexes[min_index[0]])
#     for i in range(1,stage_num-1):
#         new_stage_nums.append(max_indexes[min_index[i]]-max_indexes[min_index[i-1]])
#     new_stage_nums.append(len(layer_forward_list)-max_indexes[min_index[stage_num-2]])
#     new_stage_nums=[max_indexes[min_index[0]],max_indexes[min_index[1]]-max_indexes[min_index[0]],
#                     max_indexes[min_index[2]]-max_indexes[min_index[1]],len(layer_forward_list)-max_indexes[min_index[2]]]
#     print("rearange",new_stage_nums)
#     return new_stage_nums
if __name__ == '__main__':
    # if args.rank==1:
    #     from viztracer import VizTracer
    #     with VizTracer(output_file="result_pp.json") as tracer:
    #         main()
    # else:
    #     main()
    main()
