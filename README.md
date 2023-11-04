## RUPQ: Relative Update-Preserving Quantizer

This is the official pytorch implementation of the paper "RUPQ: Improving low-bit quantization by equalizing relative updates of quantization parameters", British Machine Vision Conference, 2023. This repository contains the reproduction of the results presented in the paper for image classification (ResNet-18 and MobileNet-v2), super-resolution (SRResNet and EDSR) and object detection (YOLO-v3). 

## Reference

If you find our work useful, please cite: TBD

## Quick Start

### Prerequisites

It is recommended to use python 3.7.13.

Run this command to install all the necessary libraries:

```
pip install -r rupq/requirements.txt
```

Add the following line to the end of your ~/.bashrc file:

```
export PYTHONPATH=/path/to/this/repository:$PYTHONPATH
```

### Dataset

Change the `rupq/dataloaders/datasets_info.txt` and enter the actual path to the required dataset.

 - The DIV2K dataset can be found [here](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar).
 - The Set14 dataset can be found [here](https://cv.snu.ac.kr/research/EDSR/benchmark.tar).
 - The COCO dataset can be downloaded by running [this](https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/data/get_coco_dataset.sh) script.

### Models

 - EDSR model implementation is taken from [here](https://github.com/sanghyun-son/EDSR-PyTorch).
 - YOLO-v3 implementation is taken from [here](https://github.com/eriklindernoren/PyTorch-YOLOv3).

### Training the full-precision (FP) model:

Run this command to start training FP model:

```
python rupq/main.py --config rupq/examples/float/classification/resnet18/fp.yaml --logdir fp_resnet18_logdir --gpu=0,1
```

### Quantizing the model:

To start the w2a2 quantization of ResNet-18, run this command:

```
python rupq/main.py --config rupq/examples/quantized/classification/resnet18/w2a2_rupq.yaml --logdir w2a2_resnet18_logdir --load fp_resnet18_logdir/version_0/model.ckpt --gpu=0,1
```

Run the tensorboard to check the additional information about quantization process:

```
tensorboard --logdir w2a2_resnet18_logdir
```

For quantizing custom model on custom dataset, the following things are required:

    1. Provide the code for dataloader in `rupq/dataloaders`.

    2. Provide model definition in `rupq/models`.

    3. Customize training loop in `rupq/tools/training`, if needed.

    4. Create fp config, similar to the ones in `rupq/examples/float`. Train the fp model.

    5. Create quantization config, similarly to the ones in `rupq/examples/quantization`. Train the quantized model.


**Note**: Applying standardization to input activations requires additional hyperparameters tuning, therefore we recommend to try first the setting without this feature (remove `transformation` from `input_quantizer` in the config) for your custom model. Our experience tells that input activations standardization brings a very slight improvement for quantized model quality.  
