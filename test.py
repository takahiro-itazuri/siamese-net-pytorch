import os
import argparse
import torch
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from network import SiameseNetwork
from contrastive import ContrastiveLoss
from util import create_pairs

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', required=True, help='model path')
  parser.add_argument('--batch_size', type=int, default=1000, help='batch size (default: 1000)')
  opt = parser.parse_args()
  opt.use_gpu = torch.cuda.is_available()

  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
  ])

  test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
  test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)

  siamese_net = SiameseNetwork()
  siamese_net.load_state_dict(torch.load(opt.model))
  if opt.use_gpu:
    siamese_net = siamese_net.cuda()

  criterion = ContrastiveLoss()

  running_loss = 0
  num_itrs = len(test_loader)
  for inputs, labels in test_loader:
    x1, x2, t = create_pairs(inputs, labels)
    x1, x2, t = Variable(x1), Variable(x2), Variable(t)
    if opt.use_gpu:
      x1, x2, t = x1.cuda(), x2.cuda(), t.cuda()
    
    y1, y2 = siamese_net(x1, x2)
    loss = criterion(y1, y2, t)

    running_loss += loss.item()
  
  print('loss: {:.4f}'.format(running_loss / num_itrs))

if __name__=='__main__':
  main()