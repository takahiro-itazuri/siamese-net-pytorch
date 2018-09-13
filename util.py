import torch

def create_pairs(inputs, labels):
  x1 = inputs[::2]
  x2 = inputs[1::2]
  target = []

  for i in range(0, labels.shape[0], 2):
    if labels[i] == labels[i+1]:
      target.append(1.0)
    else:
      target.append(0.0)  
  target = torch.tensor(target)

  return x1, x2, target 
