# A Study of Filter Pruning on LeNet5

This project contains a pytorch implementation of:
- building a LeNet5 model; 
- training the LeNet5 model with MNISTl;  
- pruning the LeNet5 model utilizing the L1-norm based Filter Pruning method in [Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710)
- fine-tuning the pruned model;
- retraining the pruned model with randomly initialized weights.

The implementation is partially borrowed from [here](https://github.com/Eric-mingjie/rethinking-network-pruning/tree/master/cifar/l1-norm-pruning), which contains re-implementation of all CIFAR experiments of the following paper
Pruning Filters for Efficient ConvNets (ICLR 2017).
## Dependencies
torch v2.0.0, torchvision v0.15.1

## Baseline 

```shell
python baseline.py
```

## Prune
The `ratio` argument specifies which pruning ratio to use: `0.65` (gives actual pruning ratio ~57%); `0.8` (gives actual pruning ratio ~67%); `0.9` (gives actual pruning ratio ~77%).
```shell
python lenet5prune.py --ratio 0.65 --save [DIRECTORY TO STORE RESULT]
python lenet5prune.py --ratio 0.8 --save [DIRECTORY TO STORE RESULT]
python lenet5prune.py --ratio 0.9 --save [DIRECTORY TO STORE RESULT]
```

## Fine-tune

```shell
python finetune.py --refine [PATH TO THE PRUNED MODEL] --save [DIRECTORY TO STORE RESULT]
```

## Scratch-train

```shell
python scratch_train.py --scratch [PATH TO THE PRUNED MODEL] --save [DIRECTORY TO STORE RESULT]
```
