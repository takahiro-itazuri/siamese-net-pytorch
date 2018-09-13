import os
import sys
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from contrastive import ContrastiveLoss
from network import SiameseNetwork
from util import create_pairs

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--margin', type=float, default=1.0, help='margin for contrastive loss')
  parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
  parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
  parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
  parser.add_argument('--batch_size', type=int, default=1000, help='batch size')
  parser.add_argument('--log_dir', required=True, help='log directory')
  parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loading')
  opt = parser.parse_args()
  opt.use_gpu = torch.cuda.is_available()

  if not os.path.exists(opt.log_dir):
    os.makedirs(opt.log_dir)

  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
  ])

  train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
  train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

  siamese_net = SiameseNetwork()
  if opt.use_gpu:
    siamese_net = siamese_net.cuda()
  
  criterion = ContrastiveLoss()
  optimizer = torch.optim.SGD(siamese_net.parameters(), lr=opt.lr, momentum=opt.momentum)

  history = {}
  history['loss'] = []

  for epoch in range(opt.num_epochs):
    num_itrs = len(train_loader)
    running_loss = 0
    for itr, (inputs, labels) in enumerate(train_loader):
      optimizer.zero_grad()

      x1, x2, t = create_pairs(inputs, labels)
      x1, x2, t = Variable(x1), Variable(x2), Variable(t)
      if opt.use_gpu:
        x1, x2, t = x1.cuda(), x2.cuda(), t.cuda()

      y1, y2 = siamese_net(x1, x2)
      loss = criterion(y1, y2, t)

      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      sys.stdout.write('\r\033[Kitr [{}/{}], loss: {:.4f}'.format(itr, num_itrs, loss.item()))
      sys.stdout.flush()

    history['loss'].append(running_loss / num_itrs)
    sys.stdout.write('\r\033[Kepoch [{}/{}], loss: {:.4f}'.format(epoch+1, opt.num_epochs, running_loss / num_itrs))
    sys.stdout.write('\n')

  torch.save(siamese_net.state_dict(), os.path.join(opt.log_dir, 'model.pth'))
  
  with open(os.path.join(opt.log_dir, 'history.pkl'), 'wb') as f:
    pickle.dump(history, f)

  plt.plot(history['loss'])
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.grid()
  plt.savefig(os.path.join(opt.log_dir, 'loss.png'))

if __name__=='__main__':
  main()