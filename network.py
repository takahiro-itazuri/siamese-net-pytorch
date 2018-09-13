import torch
from torch import nn

class SiameseNetwork(nn.Module):
  def __init__(self):
    super(SiameseNetwork, self).__init__()
    self.cnn = nn.Sequential(
      nn.Conv2d(1, 64, kernel_size=3, padding=1),   # (  1, 28, 28) -> ( 64, 28, 28)
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2, stride=2),                    # ( 64, 28, 28) -> ( 64, 14. 14)
      nn.Conv2d(64, 128, kernel_size=3, padding=1), # ( 64, 14, 14) -> (128, 14, 14)
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2, stride=2)                     # (128, 14, 14) -> (128,  7,  7)
    )

    self.fc = nn.Sequential(
      nn.Linear(128 * 7 * 7, 4096),
      nn.ReLU(inplace=True),
      nn.Linear(4096, 2)
    )

  def forward_once(self, x):
    out = self.cnn(x)
    out = out.view(out.shape[0], -1)
    out = self.fc(out)
    return out

  def forward(self, in1, in2):
    out1 = self.forward_once(in1)
    out2 = self.forward_once(in2)
    return out1, out2