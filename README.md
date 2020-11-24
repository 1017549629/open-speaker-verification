
# Open Speaker Verification

## News

[11/24/2020] Important future updates will be focused on VoxSRC2020 Tech Reports. Also a large scale ID(1M+ Identities) training scripts will be updated later. More model benchmarks will be updated :)

[10/25/2020] I have released my baseline system ResNet34-AM-VoxCeleb2 based on **mmclassification**. Thanks to [mmclassification](https://github.com/open-mmlab/mmclassification), I can fully utilize all the functions and modules provided without paying a lot attention on the training process.

## Introduction

This is a repo intended to provide an open speaker verification tool. Currently this project only provides a training and extraction process. More fundamental functions like feature extraction, post processing, scoring backends and augmentation research will be updated later.
The project is based on [mmclassification codebase](https://github.com/open-mmlab/mmclassification).
Please refer to [mmclassification  readme](README.mmclassification.md) for installation and running scripts.
The code is tested with PyTorch 1.6.0 and CUDA 10.2. **NOTE**: The pretrained model is saved in PyTorch 1.6.0. So if you are using older versions, you may need to upgrade your PyTorch Version to 1.6.0+ to load our released model.

## Data preparation

All features adopted in our training frameworks are extracted from [Kaldi](https://github.com/kaldi-asr/kaldi), cepstral mean normalized(CMN) and no VAD implemented.

Preparation scripts will be released soon.

## Attribute

### dataset

- [x] [random speaker dataset](mmcls/datasets/speaker_dataset.py)
- [x] [balanced speaker dataset PK sampler](mmcls/datasets/pk_speaker_dataset.py)
- [x] [PK sampler for distributed dataloader](mmcls/datasets/samplers/distributed_sampler.py) **Important for metric learning in SV**
- [ ] dynamic speaker dataset

### pipeline

- [x] [SpecCutout](mmcls/datasets/pipelines/formating.py)

### backbone

- [x] [ResNet34](mmcls/models/backbones/resnet_cifar.py)
- [x] [SEResNet34](mmcls/models/backbones/seresnet_asv.py)
- [x] [SEResNet34-MSEA](configs/asv/backbone/asv_resnet_msea.py)
- [ ] RES-SE-TDNN

### Pooling Method

- [x] [STATS pooling](mmcls/models/necks/STP.py)
- [x] [SelfAttention pooling(channel-dependent in ECAPA_TDNN)](mmcls/models/necks/ecapa_ASP.py)
- [ ] MultiHeadAttention pooling

### Metric

- [x] [AmSoftmax](mmcls/models/heads/am_head.py)
- [x] [AAMSoftmax](mmcls/models/heads/aam_head.py)
- [x] [AP+Softmax](mmcls/models/heads/softmaxproto_head.py)

## Released Model Benchmark

**NOTE**: The test set is VOX1-O(cleaned) dataset and training set is VoxCeleb2-dev. Backend is cosine similarity scoring. The minDCF criterion is the same as  VoxSRC2020.

| Model | Backbone          | Metric | feature | batch size | config | raw EER on Vox1-O(cleaned) | raw DCF | raw EER on Vox1-H | checkpoint |
|:---------:|:-----------------:|:------------:|:------------:|:------------:|:------------:|--------------|:------------:|:------------:|:------------:|
| ResNet34-AM-VoxCeleb2 | ResNet34       | AMSoftmax, scale=30, margin=0.2 | 81 FBANK(including energy) | 128 | [conf](configs/asv/classifier/vox2_resnet34_b128x4.py) | 1.207 | 0.0738 | 2.44 | [ckpt](https://drive.google.com/file/d/1d5cJQsLNUrZ3-IIiBPI8l-7W7G_jqKV8/view?usp=sharing) |
| ResNet34-AM-VoxCeleb2-syncBN | ResNet34 | AMSoftmax, scale=30, margin=0.2 | 81 FBANK(including energy) | 128 | [conf](configs/asv/classifier/vox2_resnet34_b128x4_syncBN.py) | 1.196 | 0.0791 | - | [ckpt](https://drive.google.com/file/d/1rJ9tMGU4OVXQwF66e0Z-0scDc9MmipHo/view?usp=sharing) |
| SEResNet34-AM-VoxCeleb2 | SEResNet34 | AMSoftmax, scale=30, margin=0.2 | 81 FBANK(including energy) | 100 | [conf](configs/asv/classifier/vox2_seresnet34_b100x4.py) | 1.121 | 0.0771 | 2.43 | [ckpt](https://drive.google.com/file/d/1jPIDDNnv_IY5rR5CUA6bqSiw-dSAP_oM/view?usp=sharing) |
| SEResNet34-AM-VoxCeleb2-syncBN | SEResNet34 | AMSoftmax, scale=30, margin=0.2 | 81 FBANK(including energy) | 100 | [conf](configs/asv/classifier/vox2_seresnet34_b100x4_syncBN.py) | 1.175 | 0.0745 | - | realeased soon |
| SEResNet34-AM-VoxCeleb2(with checkpointed) | SEResNet34 | AMSoftmax, scale=30, margin=0.2 | 81 FBANK(including energy) | 128 | [conf](configs/asv/classifier/vox2_seresnet34_b128x4.py) | 1.07 | 0.0747 | 2.43 | - |