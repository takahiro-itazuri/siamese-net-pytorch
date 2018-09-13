import torch
from torch import nn

class ContrastiveLoss(nn.Module):
  """ Contrastive Loss Class
  
  Args:
    margin (float): value of margin
  """
  def __init__(self, margin=1.0):
    super(ContrastiveLoss, self).__init__()
    self.margin = margin

  def forward(self, x1, x2, y):
    dist = torch.sqrt(torch.sum(torch.pow(x1 - x2, 2), 1))
    loss = y * dist + (1 - y) * torch.clamp(self.margin - dist, min=0.0)
    loss = torch.sum(loss) * 0.5 / y.shape[0]
    return loss
