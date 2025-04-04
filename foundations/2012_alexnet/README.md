# Implementing AlexNet from scratch

Here is the original article from 2012. [ImageNet Classification with Deep Convolutional
Neural Networks](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

# Resouces

1. https://www.digitalocean.com/community/tutorials/alexnet-pytorch
2. https://github.com/Lornatang/AlexNet-PyTorch
3. https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py
4. https://github.com/dansuh17/alexnet-pytorch
5. https://lightning.ai/jed/studios/alexnet-from-scratch-in-pytorch-lightning
6. https://github.com/talqadi7/alexnet-implementation
7. https://danielparicio.com/posts/understanding-alexnet/
8. https://danielparicio.com/posts/implementing-alexnet/
9. https://github.com/danipari/CNN-Training-From-Scratch/blob/master/training_alexnet.ipynb
10. https://www.pinecone.io/learn/series/image-search/imagenet/
11. https://medium.com/@karandeepdps/alexnet-vggnet-resnet-and-inception-11880a1ed3cd
12. https://image-net.org/index.php
13. https://github.com/vdumoulin/conv_arithmetic
14. https://maximliu-85602.medium.com/learn-cnn-and-pytorch-through-understanding-torch-nn-conv2d-class-54ad94bcc7d0
15. https://cs231n.github.io/convolutional-networks/

# Env setup

```bash
python3 -m venv alexnet-env
source alexnet-env/bin/activate
pip3 install notebook torch torchvision torchaudio scipy matplotlib tqdm tensorboard
```

# Data

We will be using `torchvision.datasets.ImageNet` module. See the documentation
[here](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageNet.html)

Download the data from [here](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php)

1. [ILSVRC2012_devkit_t12.tar.gz](https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz) (2.5MB)
2. [ILSVRC2012_img_train.tar](https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar) (138GB)
3. [ILSVRC2012_img_val.tar](https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar) (6.3GB)
