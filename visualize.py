import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from network import SiameseNetwork

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', required=True, help='model path')
  parser.add_argument('--sample_size', type=int, default=1000, help='sample size (default: 1000)')
  parser.add_argument('--log_dir', required=True, help='log directory')
  parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loading')
  opt = parser.parse_args()
  opt.use_gpu = torch.cuda.is_available()

  if not os.path.exists(opt.log_dir):
    os.makedirs(opt.log_dir)
  
  if not os.path.exists(opt.model):
    print(opt.model + ' does not exists.')
    sys.exit()

  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
  ])

  test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
  test_loader = DataLoader(test_dataset, batch_size=opt.sample_size, shuffle=False)

  siamese_net = SiameseNetwork()
  siamese_net.load_state_dict(torch.load(opt.model))
  if opt.use_gpu:
    siamese_net = siamese_net.cuda()

  for inputs, labels in test_loader:
    x = Variable(inputs)
    if opt.use_gpu:
      x = x.cuda()

    y = siamese_net.forward_once(x)

    y = y.data.cpu().numpy()
    labels = labels.numpy()

    plt.figure(figsize=(5,5))
    plt.scatter(y[:,0], y[:,1], marker='.', c=labels, cmap=plt.cm.jet)
    plt.colorbar()
    plt.grid()
    plt.savefig(os.path.join(opt.log_dir, 'visualization.png'))
    break
    

if __name__=='__main__':
  main()