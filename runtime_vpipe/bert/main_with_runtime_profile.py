from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# ==================
import threading
import argparse
from collections import OrderedDict
import importlib
import json
import os
import shutil
import sys
import time
import numpy
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
# try:
#     from modeling import BertConfig, CrossEntropyWrapper
# except ImportError:
#     from bert.modeling import BertConfig, CrossEntropyWrapper
from modeling import BertConfig, CrossEntropyWrapper
from tqdm import tqdm, trange
from utils import is_main_process, format_step
import h5py
import numpy as np
import copy
sys.path.append("..")
import runtime
#import lamb
from schedulers import PolyWarmUpScheduler
import sgd
time_list_for=[]
time_list_bac=[]
time_list_batch=[]
time_list_last_f=[]
time_list_last_b=[]
layers=[]
num_layers=[]
EVENT=threading.Event()
EVENT1=threading.Event()
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
## Required parameters
parser.add_argument("--bert_config",
                    default="bert_config.json",
                    type=str,
                    help="The BERT model config")
parser.add_argument("--max_predictions_per_seq",
                    default=80,
                    type=int,
                    help="The maximum total of masked tokens in input sequence")

## Pipedream parameters
parser.add_argument('--data_dir', type=str,
                    help='path to dataset')
parser.add_argument('--distributed_backend', type=str,
                    help='distributed backend to use (gloo|nccl)')
parser.add_argument('--module', '-m', required=True,
                    help='name of module that contains model and tensor_shapes definition')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size (default: 8)')
parser.add_argument('--grad-clip', default=5.0, type=float,
                    help='enabled gradient clipping and sets maximum gradient norm value')
parser.add_argument('--eval-batch-size', default=8, type=int,
                    help='eval mini-batch size (default: 8)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_policy', default='step', type=str,
                    help='policy for controlling learning rate')
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
parser.add_argument('--partition', default=None, type=str,
                    help="Path of partition configuration file")
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
parser.add_argument('--checkpoint_dir', default='', type=str, metavar='PATH',
                    help='path to directory to save checkpoints')
parser.add_argument('--checkpoint_dir_not_nfs', action='store_true',
                    help='checkpoint dir is not on a shared NFS server')
parser.add_argument('-s', '--synthetic_data', action='store_true',
                    help="Use synthetic data")
parser.add_argument('-v', '--verbose_frequency', default=0, type=int, metavar='N',
                    help="Log verbose information")
parser.add_argument('--num_ranks_in_server', default=4, type=int,
                    help="number of gpus per machine")

parser.add_argument('--max-length-train', default=128, type=int,
                    help='maximum sequence length for training')
parser.add_argument('--min-length-train', default=0, type=int,
                    help='minimum sequence length for training')
parser.add_argument('--no-bucketing', action='store_true',
                    help='enables bucketing')

# Recompute tensors from forward pass, instead of saving them.
parser.add_argument('--recompute', action='store_true',
                    help='Recompute tensors in backward pass')
parser.add_argument('--sync_mode', type=str, choices=['asp', 'bsp'],
                    required=True, help='synchronization mode')
# Macrobatching reduces the number of weight versions to save,
# by not applying updates every minibatch.
parser.add_argument('--macrobatch', action='store_true',
                    help='Macrobatch updates to save memory')

BSP = 'bsp'
ASP = 'asp'

best_prec1 = 0


# Helper methods.
def is_first_stage():
    return args.stage is None or (args.stage == 0)


def is_last_stage():
    return args.stage is None or (args.stage == (args.num_stages - 1))


class pretraining_dataset(Dataset):

    def __init__(self, input_file, max_pred_length):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        f = h5py.File(input_file, "r")
        keys = ['input_ids', 'input_mask', 'segment_ids', 'masked_lm_positions', 'masked_lm_ids',
                'next_sentence_labels']
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        f.close()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs[0])

    def __getitem__(self, index):
        [input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, next_sentence_labels] = [
            torch.from_numpy(input[index].astype(np.int64)) if indice < 5 else torch.from_numpy(
                np.asarray(input[index].astype(np.int64))) for indice, input in enumerate(self.inputs)]

        masked_lm_labels = torch.ones(input_ids.shape, dtype=torch.long) * -1
        index = self.max_pred_length
        # store number of  masked tokens in index
        padded_mask_indices = (masked_lm_positions == 0).nonzero(as_tuple=False)
        if len(padded_mask_indices) != 0:
            index = padded_mask_indices[0].item()
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]

        return [input_ids, segment_ids, input_mask,
                masked_lm_labels, next_sentence_labels]


def main():
    global args, best_prec1
    args = parser.parse_args()
    # torch.cuda.set_device(args.local_rank)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.local_rank}"
    print(args.local_rank)
    print("in main begin")
    config = BertConfig.from_json_file(args.bert_config)

    config_1=config


    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    criterion = CrossEntropyWrapper(config.vocab_size)

    # create stages of the model

    # def get_new_json(filepath, key, value):
    #     key_ = key.split(".")
    #     key_length = len(key_)
    #     with open(filepath, 'rb') as f:
    #         json_data = json.load(f)
    #         i = 0
    #         a = json_data
    #         while i < key_length:
    #             if i + 1 == key_length:
    #                 a[key_[i]] = value
    #                 i = i + 1
    #             else:
    #                 a = a[key_[i]]
    #                 i = i + 1
    #     f.close()
    #     return json_data
    #
    # def rewrite_json_file(filepath, json_data):
    #     with open(filepath, 'w') as f:
    #         json.dump(json_data, f)
    #     f.close()
    #
    # key = "signal"
    # value = int(1)
    # json_path = "vgpus=4/vpipe_signal.json"
    #
    # m_json_data = get_new_json(json_path, key, value)
    # rewrite_json_file(json_path, m_json_data)
    #
    # signal=json.load(open("vgpus=4/vpipe_signal.json",'r'))
    # if signal["signal"]==1:
    #     print("signal")
    #     print(signal["signal"])
    partition = json.load(open(args.partition, 'r'))
    # print("partition")
    # print(args.partition)
    module = importlib.import_module(args.module)

    args.arch = "bert"
    model = module.model(criterion, partition["partition"], partition["recompute_ratio"])
    # print("model length")
    # print(len(model))
    # print(model)
    # model1=module.model(criterion,[12, 14, 12, 11], partition["recompute_ratio"])

    input_size = [args.batch_size, args.max_length_train]
    training_tensor_shapes = {"input0": input_size, "input1": input_size,
                              "input2": [args.batch_size, 1, 1, args.max_length_train],
                              "target": input_size}
    dtypes = {"input0": torch.int64, "input1": torch.int64,
              "input2": torch.float32, "target": torch.int64}
    inputs_module_destinations = {"input0": 0, "input1": 0, "input2": 0}
    target_tensor_names = {"target"}

    training_tensor_shapes1 = {"input0": input_size, "input1": input_size,
                              "input2": [args.batch_size, 1, 1, args.max_length_train],
                              "target": input_size}
    dtypes1 = {"input0": torch.int64, "input1": torch.int64,
              "input2": torch.float32, "target": torch.int64}
    inputs_module_destinations1 = {"input0": 0, "input1": 0, "input2": 0}
    target_tensor_names1 = {"target"}




    for module_id, (stage, inputs, outputs) in enumerate(model[:-1]):  # Skip last layer (loss).
        input_tensors = []
        for module_input in inputs:
            if module_input in inputs_module_destinations:
                inputs_module_destinations[module_input] = module_id

            input_tensor = torch.ones(tuple(training_tensor_shapes[module_input]),
                                      dtype=dtypes[module_input]).cuda()
            input_tensors.append(input_tensor)
        #print(stage)
        stage.cuda()


        # PyTorch should not maintain metadata for a backward pass on
        # synthetic inputs. Without the following line, the runtime is
        # as much as 1.5x slower in a full DP configuration.
        with torch.no_grad():
            output_tensors = stage(*tuple(input_tensors))
        if not type(output_tensors) is tuple:
            output_tensors = [output_tensors]
        for output, output_tensor in zip(outputs,
                                         list(output_tensors)):
            training_tensor_shapes[output] = list(output_tensor.size())
            dtypes[output] = output_tensor.dtype

    # for module_id, (stage, inputs, outputs) in enumerate(model1[:-1]):  # Skip last layer (loss).
    #     input_tensors = []
    #     for module_input in inputs:
    #         if module_input in inputs_module_destinations1:
    #             inputs_module_destinations1[module_input] = module_id
    #
    #         input_tensor = torch.ones(tuple(training_tensor_shapes1[module_input]),
    #                                   dtype=dtypes1[module_input]).cuda()
    #         input_tensors.append(input_tensor)
    #     #print(stage)
    #     stage.cuda()
    #
    #     # PyTorch should not maintain metadata for a backward pass on
    #     # synthetic inputs. Without the following line, the runtime is
    #     # as much as 1.5x slower in a full DP configuration.
    #     #with torch.no_grad():
    #     output_tensors = stage(*tuple(input_tensors))
    #     if not type(output_tensors) is tuple:
    #         output_tensors = [output_tensors]
    #     for output, output_tensor in zip(outputs,
    #                                      list(output_tensors)):
    #         training_tensor_shapes1[output] = list(output_tensor.size())
    #         dtypes1[output] = output_tensor.dtype

    eval_tensor_shapes = {}
    for key in training_tensor_shapes:
        eval_tensor_shapes[key] = tuple(
            training_tensor_shapes[key])
        training_tensor_shapes[key] = tuple(
            training_tensor_shapes[key])

    # for key in training_tensor_shapes1:
    #     eval_tensor_shapes[key] = tuple(
    #         training_tensor_shapes1[key])
    #     training_tensor_shapes1[key] = tuple(
    #         training_tensor_shapes1[key])

    configuration_maps = {
        'module_to_stage_map': None,
        'stage_to_rank_map': None,
        'stage_to_depth_map': None
    }
    if args.config_path is not None:
        json_config_file = json.load(open(args.config_path, 'r'))
        configuration_maps['module_to_stage_map'] = json_config_file.get("module_to_stage_map", None)
        configuration_maps['stage_to_rank_map'] = json_config_file.get("stage_to_rank_map", None)
        configuration_maps['stage_to_rank_map'] = {
            int(k): v for (k, v) in configuration_maps['stage_to_rank_map'].items()}
        configuration_maps['stage_to_depth_map'] = json_config_file.get("stage_to_depth_map", None)

    r = runtime.StageRuntime(
        model=model, distributed_backend=args.distributed_backend,
        fp16=args.fp16, loss_scale=args.loss_scale,
        training_tensor_shapes=training_tensor_shapes,
        eval_tensor_shapes=eval_tensor_shapes,
        training_tensor_dtypes=dtypes,
        inputs_module_destinations=inputs_module_destinations,
        target_tensor_names=target_tensor_names,
        configuration_maps=configuration_maps,
        master_addr=args.master_addr,
        rank=args.rank, local_rank=args.local_rank,
        num_ranks_in_server=args.num_ranks_in_server,
        verbose_freq=args.verbose_frequency,
        model_type=runtime.BERT,
        event=EVENT,
        event1=EVENT1,
        worker_num_sum=args.worker_num_sum,
        batch_size=args.batch_size,
        batch_size_for_communication=args.batch_size_for_communication,  # 总共的batch_size大小，所有的stage该数值相同
        stage_num=4,
        stage_nums=partition["partition"],
        enable_recompute=args.recompute)
    print("finish stage initial")

    # stage needed to determine if current stage is the first stage
    # num_stages needed to determine if current stage is the last stage
    # num_ranks needed to determine number of warmup_minibatches in case of pipelining
    args.stage = r.stage
    args.num_stages = r.num_stages
    args.num_ranks = r.num_ranks
    if not is_first_stage():
        args.synthetic_data = True

    # number of versions is the total number of machines following the current
    # stage, shared amongst all replicas in this stage
    num_versions = r.num_warmup_minibatches + 1

    # if specified, resume from checkpoint
    #if args.resume:

    # optimizer = sgd.SGDWithWeightStashing(
    #     modules=r.modules(), master_parameters=r.master_parameters,
    #     model_parameters=r.model_parameters, loss_scale=args.loss_scale,
    #     num_versions=num_versions, lr=args.lr, momentum=args.momentum,
    #     weight_decay=args.weight_decay, verbose_freq=args.verbose_frequency)


    # TODO: make this configurable by args
    optimizer = sgd.SGDWithWeightStashing(
        modules=r.modules(), master_parameters=r.master_parameters,
        model_parameters=r.model_parameters, loss_scale=args.loss_scale,
        num_versions=num_versions, lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay, verbose_freq=args.verbose_frequency)

    # optimizer = lamb.LambWithWeightStashing(
    #     modules=r.modules(), master_parameters=r.master_parameters,
    #     model_parameters=r.model_parameters, loss_scale=args.loss_scale,
    #     num_versions=num_versions, lr=args.lr, betas=(0.9, 0.999),
    #     weight_decay=args.weight_decay, verbose_freq=args.verbose_frequency,
    #     macrobatch=args.macrobatch)
    lr_scheduler = PolyWarmUpScheduler(optimizer,
                                       warmup=0.2,
                                       total_steps=36320)



    cudnn.benchmark = True

    distributed_sampler = False
    if configuration_maps['stage_to_rank_map'] is not None:
        num_ranks_in_first_stage = len(configuration_maps['stage_to_rank_map'][0])
        if num_ranks_in_first_stage > 1:
            distributed_sampler = True



    # if checkpoint is loaded, start by running validation

    test_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if
                   os.path.isfile(os.path.join(args.data_dir, f)) and 'test' in f]
    test_file = test_files[0]
    test_data = pretraining_dataset(test_file, args.max_predictions_per_seq)
    val_loader = DataLoader(test_data, shuffle=True,
                              batch_size=args.batch_size, num_workers=4,
                              pin_memory=True)

    if args.resume:
        assert args.start_epoch > 0
        validate(val_loader, r, args.start_epoch - 1)

    train_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if
                   os.path.isfile(os.path.join(args.data_dir, f)) and 'training' in f]
    data_file = train_files[0]
    train_data = pretraining_dataset(data_file, args.max_predictions_per_seq)

    train_loader = DataLoader(train_data, shuffle=True,
                              batch_size=args.batch_size, num_workers=4,
                              pin_memory=True)


    print("train begin")
    for epoch in range(2, 48):
        model1 = module.model(criterion, [epoch-1,1,48-epoch,1], partition["recompute_ratio"])
        print([epoch-1,1,48-epoch,1])
        for module_id, (stage, inputs, outputs) in enumerate(model1[:-1]):  # Skip last layer (loss).
            input_tensors = []
            for module_input in inputs:
                if module_input in inputs_module_destinations1:
                    inputs_module_destinations1[module_input] = module_id

                input_tensor = torch.ones(tuple(training_tensor_shapes1[module_input]),
                                          dtype=dtypes1[module_input]).cuda()
                input_tensors.append(input_tensor)
            # print(stage)
            stage.cuda()

            # PyTorch should not maintain metadata for a backward pass on
            # synthetic inputs. Without the following line, the runtime is
            # as much as 1.5x slower in a full DP configuration.
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

        r.initialize1(model1, inputs_module_destinations1, configuration_maps,
                      args.master_addr, args.rank, args.local_rank, args.num_ranks_in_server,
                      training_tensor_shapes1,
                      dtypes1, target_tensor_names1)

        if distributed_sampler:
             train_loader.sampler.set_epoch(epoch)
        # train or run forward pass only for one epoch
        # if args.forward_only:
        if args.resume:
            validate(val_loader, r, epoch)
        else:
            train(train_loader, r, optimizer, epoch, lr_scheduler)
            # evaluate on validation set
            #prec1 = validate(val_loader, r, epoch)
            prec1 = 0
            if r.stage != r.num_stages: prec1 = 0

            # remember best prec@1 and save checkpoint
            best_prec1 = max(prec1, best_prec1)

            should_save_checkpoint = args.checkpoint_dir_not_nfs or r.rank_in_stage == 0
            if args.checkpoint_dir and should_save_checkpoint:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': r.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer': optimizer.state_dict()
                }, args.checkpoint_dir, r.stage, epoch)

def train(train_loader, r, optimizer, epoch, lr_scheduler):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    train_loss=[]
    # switch to train mode
    n = r.num_iterations(loader_size=len(train_loader))
    print("n")
    print(n)
    # print(args.num_minibatches)
    if args.num_minibatches is not None:
        n = min(n, args.num_minibatches)

    if not is_first_stage(): train_loader = None
    r.set_loader(train_loader)

    num_warmup_minibatches = r.num_warmup_minibatches

    # print("warm")
    # print(num_warmup_minibatches)
    # current step
    s = 0
    warmup_steps = 0

    epoch_start_time = 0
    batch_start_time = 0
    time_for=0
    time_bac=0
    time_bac_begin=time.time()
    time_last_f=0
    time_last_b=0
    def pipelining(steps, print_freq,time_for,time_bac,time_bac_begin,time_last_f,time_last_b,weight_stash=False):
        nonlocal s, epoch_start_time, batch_start_time
        # start num_warmup_minibatches forward passes
        for i in range(num_warmup_minibatches):
            r.run_forward()

        s += num_warmup_minibatches
        # steps=1000
        for i in range(steps - num_warmup_minibatches):
            if i == 1009 - num_warmup_minibatches and epoch == 0:
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
                # torch.distributed.barrier()
                r.wait()
                EVENT.clear()
                EVENT1.clear()
                print("end")

                return




            s += 1
            # perform forward pass
            time1=time.time()
            r.run_forward()
            time_last_f+=time.time()-time1
            if i>=60:
                time_for+=r.time_forward
            if is_last_stage():
                # measure accuracy and record loss
                output, target, loss = r.output, r.target, r.loss.item()
                losses.update(loss)

                if s == warmup_steps:
                    epoch_start_time = time.time()
                    batch_start_time = time.time()

                if s % print_freq == 0 and s > warmup_steps:
                    #r.Send_Status(i)
                    # if i%print_freq==0:
                    #     r.Refresh_Status(i)
                    # measure elapsed time
                    batch_time.update((time.time() - batch_start_time) / print_freq)
                    batch_start_time = time.time()
                    epoch_time = (time.time() - epoch_start_time) / 3600.0
                    full_epoch_time = (epoch_time / float(s - warmup_steps)) * float(n)
                    train_loss.append(loss)
                    print('Stage: [{0}] Epoch: [{1}][{2}/{3}]\t'
                          'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Epoch time [hr]: {epoch_time:.3f} ({full_epoch_time:.3f})\t'
                          'Memory: {memory:.3f} ({cached_memory:.3f})\t'
                          'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                        args.stage, epoch, s, n, batch_time=batch_time,
                        epoch_time=epoch_time, full_epoch_time=full_epoch_time,
                        loss=losses,  # top1=top1, top5=top5,
                        memory=(float(torch.cuda.memory_allocated()) / 10 ** 9),
                        cached_memory=(float(torch.cuda.memory_cached()) / 10 ** 9)))
                    import sys;
                    sys.stdout.flush()
            else:
                if s % print_freq == 0 and s > warmup_steps:
                    # if i%print_freq==0:
                    #     r.Send_Status_local(i)
                    # if i==r.rank*10+50:
                    #     r.status[r.rank]=1
                    #r.Rec_Status(i)
                    print('Stage: [{0}] Epoch: [{1}][{2}/{3}]\tMemory: {memory:.3f} ({cached_memory:.3f})'.format(
                        args.stage, epoch, s, n, memory=(float(torch.cuda.memory_allocated()) / 10 ** 9),
                        cached_memory=(float(torch.cuda.memory_cached()) / 10 ** 9)))
                    import sys;
                    sys.stdout.flush()

            # perform backward pass
            if not weight_stash:
                time2=time.time()
                r.run_backward()
                time_last_b+=time.time()-time2
            else:
                optimizer.zero_grad()
                optimizer.load_old_params()
                r.run_backward()
                optimizer.load_new_params()
                lr_scheduler.step()
                optimizer.step()
            if i>=60:
                time_bac += r.time_backward
            if i%1000==0:
                time_list_for.append(time_for)
                time_list_bac.append(time_bac)

                time_batch=time.time()-time_bac_begin
                time_bac_begin=time.time()
                time_list_batch.append(time_batch)
                time_for=0
                time_bac=0
                time_last_f=0
                time_last_b=0
            if i%1000==0 and i>0:
                def save_list_to_txt(data, filename):
                    numpy.savetxt(filename, data)
                if r.stage==0:
                    save_list_to_txt(time_list_for,"data_0_for")
                    save_list_to_txt(time_list_bac,"data_0_bac")
                if r.stage==1:
                    save_list_to_txt(time_list_for,"data_1_for")
                    save_list_to_txt(time_list_bac,"data_1_bac")
                if r.stage==2:
                    save_list_to_txt(time_list_for,"data_2_for")
                    save_list_to_txt(time_list_bac,"data_2_bac")
                if r.stage == 3:
                    save_list_to_txt(time_list_for, "data_3_for")
                    save_list_to_txt(time_list_bac, "data_3_bac")
        # finish remaining backward passes
        for i in range(num_warmup_minibatches):
            if not weight_stash:
                r.run_backward()
            else:
                optimizer.zero_grad()
                optimizer.load_old_params()
                r.run_backward()
                optimizer.load_new_params()
                lr_scheduler.step()
                optimizer.step()

        if not weight_stash:
            lr_scheduler.step()
            optimizer.base_optimizer.step()
            optimizer.zero_grad()

    if args.sync_mode == BSP:
        accumulation_steps = 32
        n -= (n % accumulation_steps)
        r.train(n)
        r.set_loss_scale(4 / accumulation_steps)
        print_freq = (args.print_freq // accumulation_steps) * accumulation_steps
        warmup_steps = 5 * print_freq
        for t in range(n // accumulation_steps):
            pipelining(accumulation_steps, print_freq)
    else:
        r.train(n)
        warmup_steps = 5 * args.print_freq
        pipelining(n, args.print_freq,time_for,time_bac,time_bac_begin,time_last_f,time_last_b,weight_stash=False)
        print("end in pipeline")
        # print("train_los")
        # print(train_loss)
        # if is_last_stage():
        #     with open("./trainloss_2.txt", 'w') as train_los:
        #         train_los.write(str(train_loss))

    # wait for all helper threads to complete
    r.wait()
    print("end in wait")
    # if is_last_stage():
    #     with open("./trainloss.txt", 'w') as train_los:
    #         train_los.write(str(train_loss))
    print("Epoch %d: %.3f seconds" % (epoch, time.time() - epoch_start_time))
    print("Epoch start time: %.3f, epoch end time: %.3f" % (epoch_start_time, time.time()))

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def validate(val_loader, r, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    n = r.num_iterations(loader_size=len(val_loader))
    if args.num_minibatches is not None:
        n = min(n, args.num_minibatches)
    r.eval(n)
    if not is_first_stage(): val_loader = None
    r.set_loader(val_loader)

    end = time.time()
    epoch_start_time = time.time()

    num_warmup_minibatches = r.num_warmup_minibatches

    with torch.no_grad():
        for i in range(num_warmup_minibatches):
            r.run_forward()

        for i in range(n - num_warmup_minibatches):
            # perform forward pass
            r.run_forward()
            r.run_ack()

            if is_last_stage():
                output, target, loss = r.output, r.target, r.loss.item()
                # print("output")
                # print(output)
                # print(output.shape)
                # print("target")
                # print(target)

                # measure accuracy and record loss
                # prec1, prec5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss)
                # top1.update(prec1[0], output.size(0))
                # top5.update(prec5[0], output.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    print('Test: [{0}][{1}/{2}]\t'
                          'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Memory: {memory:.3f} ({cached_memory:.3f})\t'
                          'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                        epoch, i, n, batch_time=batch_time, loss=losses,
                        memory=(float(torch.cuda.memory_allocated()) / 10 ** 9),
                        cached_memory=(float(torch.cuda.memory_cached()) / 10 ** 9)))
                    import sys;
                    sys.stdout.flush()

        if is_last_stage():
            print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))

        for i in range(num_warmup_minibatches):
            r.run_ack()

        # wait for all helper threads to complete
        r.wait()

        print('Epoch %d: %.3f seconds' % (epoch, time.time() - epoch_start_time))
        print("Epoch start time: %.3f, epoch end time: %.3f" % (epoch_start_time, time.time()))

    return top1.avg


# TODO: Verify that checkpointing works correctly for GNMT
def save_checkpoint(state, checkpoint_dir, stage, epoch):
    assert os.path.isdir(checkpoint_dir)
    checkpoint_file_path = os.path.join(checkpoint_dir, "checkpoint.%d.pth.tar.epoch.%d" % (stage, epoch))
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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        print("target")
        print(target.size())
        _, pred = output.topk(maxk, 1, True, True)
        #print(pred.shape)
        #print(pred)
        print("pred")
        print(pred.size())
        pred=pred.view(pred.shape[0]*pred.shape[1], pred.shape[2])
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
# def find_solution(layers):
#     for i in range(layers):
#         for j in range(layers):

def runtime_control(layers,stages,num_layer,present_stage_id,start_id):
    #layers(id,stage_id,forward_time,backward_time,communication)
    #stages(id,compute_time)
    #num_layer(id,num)
    #start_id 当前stage起始层的id
    record=[[0 for _ in range(num_layer[present_stage_id]+1)] for _ in range(num_layer[present_stage_id]+1)]
    def find_min_index_2d(arr):
        min_value = float('inf')
        min_index = (0, 0)
        for i in range(len(arr)):
            for j in range(len(arr[i])):
                if arr[i][j] < min_value:
                    min_value = arr[i][j]
                    min_index = (i, j)
        return min_index
    def compute_time_1(begin_layer_id,end_layer_id):
        stages_=copy.copy(stages)
        list_ = []
        for i in range(end_layer_id - begin_layer_id):
            stages_[present_stage_id - 1] += layers[start_id + i]
        for i in range(0, present_stage_id):
            list_.append(stages_[i])
        max_time = max(list_)
        return max_time

    def compute_time_2(begin_layer_id, end_layer_id):
        stages_=copy.copy(stages)
        list_ = []
        for i in range(end_layer_id - begin_layer_id):
            stages_[present_stage_id + 1] += layers[start_id + begin_layer_id + i]
        for i in range(present_stage_id + 1, 4):
            # change 4 to stage_num_sum
            list_.append(stages_[i])
        max_time = max(list_)
        return max_time

    def compute_time_3(begin_layer_id, end_layer_id):
        stages_ = copy.copy(stages)
        stages_[present_stage_id] = 0
        for i in range(end_layer_id - begin_layer_id):
            stages_[present_stage_id] += layers[start_id + begin_layer_id + i]
        return stages_[present_stage_id]
    for i in range(num_layer[present_stage_id]+1):
        for j in range(num_layer[present_stage_id]+1):
            if i>=j:
                record[i][j]=float('inf')
            else:
                time1=compute_time_1(0,i)#pre part
                time2=compute_time_3(i,j)#medium part f+b
                time3=compute_time_2(j,num_layer[present_stage_id])#last part
                record[i][j]=max(time1,time2,time3)
    min_index=find_min_index_2d(record)
    return min_index

if __name__ == '__main__':
    main()