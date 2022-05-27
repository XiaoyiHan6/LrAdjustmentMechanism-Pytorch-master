from torch.utils.data import DataLoader
from torchvision.models import AlexNet
from torchvision import transforms
import torchvision
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='CV Training')
    parser.add_mutually_exclusive_group()
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='CIFAR10')
    parser.add_argument('--dataset_root', type=str, default="data", help='Dataset root directory path')
    parser.add_argument('--img_size', type=int, default=227, help='image size')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--checkpoint', type=str, default="checkpoint", help='Save .pth fold')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--tensorboard', type=str, default="tensorboard", help='loss visualization')
    parser.add_argument('--epoch', type=int, default=20, help="Number of epochs to train")
    return parser.parse_args()


args = parse_args()


def adjust_learning_rate(optim, size=2, gamma=0.1, i=0):
    if (i + 1) % size == 0:
        pow = (i + 1) // size
        lr = args.lr * np.power(gamma, pow)
        for param_group in optim.param_groups:
            param_group['lr'] = lr


# 1.Create SummaryWriter
writer = SummaryWriter(args.tensorboard)

# 2.Ready dataset
if args.dataset == 'CIFAR10':
    train_dataset = torchvision.datasets.cifar.CIFAR10(root=args.dataset_root, train=True, transform=transforms.Compose
    ([transforms.Resize(args.img_size), transforms.ToTensor()]), download=True)
else:
    raise ValueError("Dataset is not CIFAR10")

print('CUDA available: {}'.format(torch.cuda.is_available()))

# 3.Length
train_dataset_size = len(train_dataset)
print("the train dataset size is {}".format(train_dataset_size))

# 4.DataLoader
train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size)

# 5.Create model
model = AlexNet()

if torch.cuda.is_available():
    model = model.cuda()
    model = torch.nn.DataParallel(model).cuda()
else:
    model = torch.nn.DataParallel(model)

# 6.Create loss
cross_entropy_loss = nn.CrossEntropyLoss()

# 7.Optimizer
optim = torch.optim.SGD(model.parameters(), lr=args.lr)

# 8. Set some parameters to control loop
iter = 0
t0 = time.time()
# epoch
for i in range(args.epoch):
    t1 = time.time()
    print(" -----------------the {} number of training epoch --------------".format(i))
    model.train()
    for data in train_dataloader:
        imgs, targets = data
        if torch.cuda.is_available():
            cross_entropy_loss = cross_entropy_loss.cuda()
            imgs, targets = imgs.cuda(), targets.cuda()
        outputs = model(imgs)
        loss_train = cross_entropy_loss(outputs, targets)
        writer.add_scalar("train_loss", loss_train.item(), iter)
        optim.zero_grad()
        loss_train.backward()
        optim.step()
        iter = iter + 1
        if iter % 100 == 0:
            print(
                "Epoch: {} | Iteration: {} | lr: {} | loss: {} | np.mean(loss): {} "
                    .format(i, iter, optim.param_groups[0]['lr'], loss_train.item(),
                            np.mean(loss_train.item())))

    writer.add_scalar("lr", optim.param_groups[0]['lr'], i)
    adjust_learning_rate(optim, i=i)
    t2 = time.time()
    h = (t2 - t1) // 3600
    m = ((t2 - t1) % 3600) // 60
    s = ((t2 - t1) % 3600) % 60
    print("epoch {} is finished, and time is {}h{}m{}s".format(i, int(h), int(m), int(s)))

    if i % 1 == 0:
        print("Save state, iter: {} ".format(i))
        torch.save(model.state_dict(), "{}/AlexNet_{}.pth".format(args.checkpoint, i))

torch.save(model.state_dict(), "{}/AlexNet.pth".format(args.checkpoint))
t3 = time.time()
h_t = (t3 - t0) // 3600
m_t = ((t3 - t0) % 3600) // 60
s_t = ((t3 - t0) % 3600) // 60
print("The finished time is {}h{}m{}s".format(int(h_t), int(m_t), int(s_t)))
writer.close()
