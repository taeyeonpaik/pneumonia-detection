import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def model(
    pretrained=True,
    min_size=1024,
    box_fg_iou_thresh=0.5,
    box_bg_iou_thresh=None,
):
    if box_bg_iou_thresh is None:
        box_bg_iou_thresh = box_fg_iou_thresh
    detector_kwargs = {
        "min_size": min_size,
        "box_fg_iou_thresh": float(box_fg_iou_thresh),
        "box_bg_iou_thresh": float(box_bg_iou_thresh),
    }
    if pretrained:
        try:
            from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                weights=weights,
                **detector_kwargs,
            )
        except (ImportError, AttributeError, TypeError):
            detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                pretrained=True,
                **detector_kwargs,
            )
    else:
        try:
            detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                weights=None,
                weights_backbone=None,
                **detector_kwargs,
            )
        except TypeError:
            detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                pretrained=False,
                **detector_kwargs,
            )

    classes = 2
    input_features = detector.roi_heads.box_predictor.cls_score.in_features
    detector.roi_heads.box_predictor = FastRCNNPredictor(input_features, classes)
    return detector
