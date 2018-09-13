# PyTorch Implementation of Siamese Network
This is PyTorch implementation of siamese Network.

Siamese network is a kind of similarity learning and a mapping function from raw input into compact representation. For detail, please see [[1]](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf).

This implementation refers [delijati/pytorch-siamese](https://github.com/delijati/pytorch-siamese) partially.

## Prerequisites
|python|3.6|
|pytorch|0.4.1|
|torchvision|0.2.1|
|matplotlib|2.2.2|


## Get Started
### Train siamese network
```bash
python train.py --log_dir logs
```

### Visualize
```bash
python visualize.py --log_dir logs --model logs/model.pth
```