import models

MODEL_DISPATCHER = {
    'resnet34': models.ResNet34,
    'resnet50': models.ResNet50,
    'resnet101': models.ResNet101,
    'resnet152': models.ResNet152,
    'inceptionv3': models.InceptionV3,
    'squeezenet1_1':models.Squeezenet1_1,
    'densenet201':models.Densenet201,
    'resnext50': models.ResNeXt_50,
    'resnext101':models.ResNeXt_101,
    'ghostnet': models.GhostNet,
    'effectnet': models.EfficientNetWrapper
}