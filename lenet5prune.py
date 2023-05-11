import argparse
import numpy as np
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

from model import LeNet5

# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming LeNet5 prune')
parser.add_argument('--dataset', type=str, default='mnist',
                    help='training dataset (default: mnist)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--model', default='./logs/baselines/checkpoint.pth.tar', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--save', default='./logs/pruned/ratio_7', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')
parser.add_argument('--ratio', type=float, default=0.65, metavar='CR',
                    help='compression ratio (default: 0.65 -> ~57%)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.save):
    os.makedirs(args.save)

model = LeNet5()

if args.cuda:
    model.cuda()

if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.model, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

print('Pre-processing Successful!')


# simple test model after Pre-processing prune (simple set BN scales to zeros)
def test(model):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'mnist':
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data.mnist', train=False, transform=transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
    else:
        raise ValueError("No valid dataset is given.")
    model.eval()
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return float(100. * correct / len(test_loader.dataset))


acc = test(model)

# Adjust the pruning rate to a suitable value for LeNet5, e.g., 0.5.
prune_prob = args.ratio

layer_id = 0
cfg = []
cfg_mask = []
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        out_channels = m.weight.data.shape[0]
        weight_copy = m.weight.data.abs().clone().cpu().numpy()
        L1_norm = np.sum(weight_copy, axis=(1, 2, 3))
        min_keep = 1
        num_keep = max(int(out_channels * (1 - prune_prob)), min_keep)
        # num_keep = int(out_channels * (1 - prune_prob))
        arg_max = np.argsort(L1_norm)
        arg_max_rev = arg_max[::-1][:num_keep]
        mask = torch.zeros(out_channels)
        mask[arg_max_rev.tolist()] = 1
        cfg_mask.append(mask)
        cfg.append(num_keep)
        layer_id += 1

        mask = mask.to(m.weight.device).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        m.weight.data.mul_(mask)
        m.bias.data.mul_(mask.squeeze())

new_model = LeNet5(cfg=cfg)
if args.cuda:
    new_model.cuda()

conv_counter = 0
for (name0, m0), (name1, m1) in zip(model.named_children(), new_model.named_children()):
    if isinstance(m0, nn.Conv2d):
        mask = cfg_mask[conv_counter]
        idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
        if idx.size == 1:
            idx = np.resize(idx, (1,))
        w = m0.weight.data[idx.tolist(), :, :, :].clone()
        m1.weight.data = w.clone()
        conv_counter += 1
    elif isinstance(m0, nn.Linear):
        m1.weight.data = m0.weight.data.clone()
        m1.bias.data = m0.bias.data.clone()

torch.save({
    'cfg': cfg,
    'state_dict': new_model.state_dict()
}, os.path.join(args.save, 'pruned.pth.tar'))

original_num_parameters = sum([param.nelement() for param in model.parameters()])
num_parameters = sum([param.nelement() for param in new_model.parameters()])
print(new_model)
model = new_model
acc = test(model)

actual_compression_ratio = 1 - (num_parameters / original_num_parameters)

print("number of parameters: " + str(num_parameters))
with open(os.path.join(args.save, "prune.txt"), "w") as fp:
    fp.write("Number of parameters: \n" + str(num_parameters) + "\n")
    fp.write("Test accuracy: \n" + str(acc) + "\n")
    fp.write("Prune ratio: \n" + str(actual_compression_ratio) + "\n")
