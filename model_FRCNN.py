import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True, min_size = 1024)
    classes = 2
    input_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(input_features, classes)
    return model
