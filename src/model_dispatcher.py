import models

MODEL_DISPATCHER = {
    'resnet34': models.ResNet34,
    'resnet50': models.ResNet50,
    'resnet101': models.ResNet101,
    'resnet152': models.ResNet152,
    'inceptionv3': models.InceptionV3
}