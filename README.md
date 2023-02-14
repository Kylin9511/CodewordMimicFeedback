## Overview
This is a PyTorch implementation of BCRNet inference. The BCRNet network performance is improved with codeword mimic (CM) learning. The key results (BCRNet benchmark and BCRNet-CM) in paper [Better Lightweight Network for Free: Codeword Mimic Learning for Massive MIMO CSI feedback](https://arxiv.org/abs/2210.16544) can be reproduced.

## Requirements

The following requirements need to be installed.
- Python == 3.7
- [PyTorch == 1.10.0](https://pytorch.org/get-started/previous-versions/#v1100)

## Project Preparation

#### A. Data Preparation

The channel state information (CSI) matrix is generated from [COST2100](https://ieeexplore.ieee.org/document/6393523) model and setting can be found in our paper. On the other hand, Chao-Kai Wen provides a pre-processed COST2100 dataset, which we adopt in BCsiNet training and inference. You can download it from [Google Drive](https://drive.google.com/drive/folders/1_lAMLk_5k1Z8zJQlTr5NRnSD6ACaNRtj?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1Ggr6gnsXNwzD4ULbwqCmjA).

#### B. Checkpoints Downloading
The checkpoints of our proposed BCRNet can be downloaded from [Baidu Netdisk](https://pan.baidu.com/s/1Kb1fU-TObWZqfmk2EBCBBw) (passwd: spkc) or [Google Drive](https://drive.google.com/drive/folders/1euHV5rYDS-Bkxi6rTsTRZm0Yf3NYzlz8?usp=sharing)

#### C. Project Tree Arrangement

We recommend you to arrange the project tree as follows.

```
home
├── CodewordMimicFeedback  # The cloned current repository
│   ├── dataset
│   ├── models
│   ├── utils
│   ├── main.py
├── COST2100  # COST2100 dataset downloaded following section A
│   ├── DATA_Htestin.mat
│   ├── ...
├── Experiments
│   ├── checkpoints  # The checkpoints folder downloaded following section B
│   │     ├── in_cr4
│   │     ├── in_cr8
│   │     ├── ...
│   ├── run.sh  # The bash script
...
```

## Key Results Reproduction

The key results reported in Table I of the paper are presented as follows.

Compression Ratio | Methods | Scenario | NMSE | Params | Checkpoints Path
:--: | :-- | :--: | --: | :--: | :--
1/4  | BCRNet    | indoor   | -17.39dB | 33K | in_cr4/bcrnet.pth
1/4  | BCRNet-CM | indoor   | -19.25dB  | 33K | in_cr4/bcrnet-cm.pth
1/4  | BCRNet    | outdoor  | -8.90dB | 33K | out_cr4/bcrnet.pth
1/4  | BCRNet-CM | outdoor  | -10.00dB  | 33K | out_cr4/bcrnet-cm.pth
1/8  | BCRNet    | indoor   | -13.19dB | 17K | in_cr8/bcrnet.pth
1/8  | BCRNet-CM | indoor   | -13.90dB  | 17K | in_cr8/bcrnet-cm.pth
1/8  | BCRNet    | outdoor  | -6.31dB | 17K | out_cr8/bcrnet.pth
1/8  | BCRNet-CM | outdoor  | -6.73dB  | 17K | out_cr8/bcrnet-cm.pth
1/16 | BCRNet    | indoor   | -8.94dB  | 8K | in_cr16/bcrnet.pth
1/16 | BCRNet-CM | indoor   | -10.36dB  | 8K | in_cr16/bcrnet-cm.pth
1/16 | BCRNet    | outdoor  | -4.36dB | 8K | out_cr16/bcrnet.pth
1/16 | BCRNet-CM | outdoor  | -4.53dB  | 8K | out_cr16/bcrnet-cm.pth
1/32 | BCRNet    | indoor   | -7.87dB  | 4K | in_cr32/bcrnet.pth
1/32 | BCRNet-CM | indoor   | -8.20dB  | 4K | in_cr32/bcrnet-cm.pth
1/32 | BCRNet    | outdoor  | -2.91dB | 4K | out_cr32/bcrnet.pth
1/32 | BCRNet-CM | outdoor  | -2.98dB  | 4K | out_cr32/bcrnet-cm.pth

The key results reported in Table II of the paper are presented as follows. Note that the performance of the original CsiNet can be found in their papers [CsiNet](https://ieeexplore.ieee.org/document/8322184) and [CsiNet+](https://ieeexplore.ieee.org/document/8972904/).
Compression Ratio | Methods | Scenario | NMSE | Params | Checkpoints Path
:--: | :-- | :--: | --: | :--: | :--
1/4  | CsiNet-CM | indoor   | -25.60dB  | 33K | in_cr4/csinet-cm.pth
1/4  | CsiNet-CM | outdoor  | -10.09dB  | 33K | out_cr4/csinet-cm.pth
1/8  | CsiNet-CM | indoor   | -15.33dB  | 17K | in_cr8/csinet-cm.pth
1/8  | CsiNet-CM | outdoor  | -7.63dB  | 17K | out_cr8/csinet-cm.pth
1/16 | CsiNet-CM | indoor   | -10.12dB  | 8K | in_cr16/csinet-cm.pth
1/16 | CsiNet-CM | outdoor  | -5.02dB  | 8K | out_cr16/csinet-cm.pth
1/32 | CsiNet-CM | indoor   | -8.75dB  | 4K | in_cr32/csinet-cm.pth
1/32 | CsiNet-CM | outdoor  | -3.38dB  | 4K | out_cr32/csinet-cm.pth

In order to reproduce the aforementioned key results, you need to download the given dataset and checkpoints. Moreover, you should arrange your project tree as instructed. An example of `Experiments/run.sh` can be found as follows.

``` bash
python /home/CodewordMimicFeedback/main.py \
  --data-dir '/home/COST2100' \
  --scenario 'in' \
  --model 'bcrnet' \
  --pretrained './checkpoints/in_cr4/bcrnet.pth' \
  --batch-size 200 \
  --workers 0 \
  --reduction 4 \
  --cpu \
  2>&1 | tee log.out
```

## Acknowledgment

This repository is modified from the [BCsiNet open source code](https://github.com/Kylin9511/BCsiNet). Please refer to it for more information.

Thank Chao-Kai Wen and Shi Jin group again for providing the pre-processed COST2100 dataset, you can find their related work named CsiNet in [Github-Python_CsiNet](https://github.com/sydney222/Python_CsiNet).
