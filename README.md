### Re-implementation for MixMatch

- Uses WideResNet-28-2 as backbone network
- Uses STL-10 dataset

### Installing environment

- (Recommand) With Docker
Use official Pytorch docker image: `pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel`
Install EfficientNet, tqdm: `pip install efficientnet_pytorch tqdm`

- Without Docker
Install Pytorch, EfficientNet and tqdm: `pip install torch==1.8.0 torchvision==0.9.0 efficientnet_pytorch tqdm`

### Get trained model weight

- Install gdown: `pip install gdown`
- Download weight:
    - #### WideResNet-28-2
    - Model without EMA, linearly growing unsup weight (test acc 0.889): `gdown --id 16coQ8EO0tBvRMVsmW_hbpIyKGTaRJL8l`  
    - Model without EMA, exponentially growing unsup weight (test acc 0.89575) `gdown --id 1xD34AAt5MazlEUwmIbPfeWPwncvYBB9g`  
    - Model with EMA, exponentially growing unsup weight (test acc 0.909375) `gdown --id 1JBQBr5t7ii7dyWpTxB1uojdUTjEl9CCS`  

### Train on STL-10 dataset
- #### Apply mixmatch's batch computing algorithm to whole dataset  
    `python train_mixmatch.py`  
- Use `--use_disk` option to store computed data at your disk instead of memory (SSD recommended). Without this option, over 20GB of memory is required in the case of STL-10 dataset.  
- #### Apply mixmatch's batch computing algorithm to each mini-batch  
    `python train_mixmatch_perbatch.py`

### TODO:
- Add configuration file