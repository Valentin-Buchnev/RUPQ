import torchvision

from rupq.models.object_detection.yolo_v3 import YOLOv3
from rupq.models.superresolution.edsr import EDSR
from rupq.models.superresolution.srresnet import SRResNet


def create_model(arch):
    if arch == "resnet18":
        return torchvision.models.resnet18()
    elif arch == "mobilenet_v2":
        return torchvision.models.mobilenet_v2()
    elif arch == "srresnet":
        return SRResNet()
    elif arch == "edsr":
        return EDSR()
    elif arch == "yolo_v3":
        return YOLOv3()
    else:
        raise Exception("Unknown architechure {}".format(arch))
