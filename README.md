# DynPipe

If you have any questions about DynPipe, please contact d202381426@hust.edu.cn for quick response.

## Overview

Repo architecture

**runtime**: contains our initial system

**runtime_bert**: contains system for bert

**profile**: contains code for prediction model

**pic**: contains scripts for figures generating and our evaluation results 
## Setups

### ImageClassification

1.Create base pytorch 

2.Setup Environment

```bash
pip install -r requirements.txt
```

3. Download and preprocess the dataset.

Vgg16, Resnet50 pre-training uses the following datasets:
-   CIFAR10 Simply use the built-in download method of torchvision
-   Mini-ImageNet https://huggingface.co/datasets/GATE-engine/mini_imagenet

### BERT

1. Setup Enviroment 

Note that you should modify the docker base image version to the Nvidia pytorch docker release 20.01. 

This may help you avoid an issue caused by the PyTorch variable version checking.

Docker file refer to : https://github.com/NVIDIA/DeepLearningExamples/blob/24b8c9c7fdfd1fa5b80d5c342f96dd922feffd24/PyTorch/LanguageModeling/BERT/Dockerfile


2. Download and preprocess the dataset.

BERT pre-training uses the following datasets:
-   BookCorpus

To download, verify, extract the datasets, and create the shards in `.hdf5` format, see:  

https://github.com/NVIDIA/DeepLearningExamples/blob/24b8c9c7fdfd1fa5b80d5c342f96dd922feffd24/PyTorch/LanguageModeling/BERT/Dockerfile

## Reproducing Experiments

The evaluation scripts can extract the results from output and generate the figures in the paper, Here list the core evaluation cases:

```bash
# Fig.9
cd dynpipe/pic/acc_loss_pic/
python acc_loss.py
# Fig.10
cd dynpipe/pic/throughtput_bar/
python fig_10_throughtout_bar.py
# Fig.11
cd dynpipe/pic/GPU_use/
python gpu_use.py
# Fig.12 Since the total number of iterations is inconsistent, we need to concat the pictures.
cd dynpipe/pic/dyn/
python cs_p2p.py
# Fig.13
#iteration time 
cd dynpipe/pic/iteration_time
python iteration_time.py
#dyn_acc
cd dynpipe/pic/dyn_acc
python dyn_acc.py
```
Evaluation results are stored in pic/pdf folder,
formatted as PDF files.

Here list the core reproducing steps:
- generate partition plans
```bash
# generate plans through profile data
cd dynpipe/scripts/
python calculate_layering.py
python json_config_generation.py
```
- generate prediction models
```bash
cd dynpipe/profile/
python main.py # generate datasets
python process_json.py # generate predict model
```
- begin training
```bash
#pipedream
cd dynpipe/runtime/image_classification/
python main_with_runtime_pipedream.py --module models.vgg16.gpus=8 --rank [PRESENT_RANK_ID] --local_rank [PRESENT_GPU_ID] --master_addr [MASTER_ADDRESS] --config_path models/vgg16/gpus=8/hybrid_conf.json --partition models/vgg16/gpus=8/vgg16_8.json --present_stage_id [PRESENT_STAGE_ID] --worker_num_sum 8 --num_minibatches 420 --distributed_backend gloo --data_dir [DATA_ADDRESS]
#DynPipe-Re
python main_with_runtime_dynpipe_re.py --module models.vgg16.gpus=8 --rank [PRESENT_RANK_ID] --local_rank [PRESENT_GPU_ID] --master_addr [MASTER_ADDRESS] --config_path models/vgg16/gpus=8/hybrid_conf.json --partition models/vgg16/gpus=8/vgg16_8.json --present_stage_id [PRESENT_STAGE_ID] --worker_num_sum 8 --num_minibatches 420 --distributed_backend gloo --data_dir [DATA_ADDRESS]
#Simple
python main_with_runtime_simple.py --module models.vgg16.gpus=8 --rank [PRESENT_RANK_ID] --local_rank [PRESENT_GPU_ID] --master_addr [MASTER_ADDRESS] --config_path models/vgg16/gpus=8/hybrid_conf.json --partition models/vgg16/gpus=8/vgg16_8.json --present_stage_id [PRESENT_STAGE_ID] --worker_num_sum 8 --num_minibatches 420 --distributed_backend gloo --data_dir [DATA_ADDRESS]
#DynPipe
python main_with_runtime.py --module models.vgg16.gpus=8 --rank [PRESENT_RANK_ID] --local_rank [PRESENT_GPU_ID] --master_addr [MASTER_ADDRESS] --config_path models/vgg16/gpus=8/hybrid_conf.json --partition models/vgg16/gpus=8/vgg16_8.json --present_stage_id [PRESENT_STAGE_ID] --worker_num_sum 8 --num_minibatches 420 --distributed_backend gloo --data_dir [DATA_ADDRESS]
 ```
- submit tasks for gpu interference
```bash
#the same as processes in creating datasets for prediction model 
```
