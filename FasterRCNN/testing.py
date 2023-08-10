import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from model import model
import matplotlib.pyplot as plt
from pipeline import test_data_loader

def calculate_intersection_over_union(truth, prediction):
    width = min(truth[2], prediction[2]) - max(truth[0], prediction[0]) + 1
    height = min(truth[3], prediction[3]) - max(truth[1], prediction[1]) + 1
    union_area = ((truth[2] - truth[0] + 1) * (truth[3] - truth[1] + 1) + (prediction[2] - prediction[0] + 1) * (prediction[3] - prediction[1] + 1) - width * height)
    return 0.0 if width < 0 or height < 0 else width * height / union_area

def find_best_match(truth, prediction, prediction_index, threshold = 0.5, ious = None):
    best_match_intersection_over_union, best_match_index = -np.inf, -1
    for truth_index in range(len(truth)):
        if truth[truth_index][0] < 0:
            continue
        intersection_over_union = -1 if ious is None else ious[truth_index][prediction_index]
        if intersection_over_union < 0:
            intersection_over_union = calculate_intersection_over_union(truth[truth_index], prediction)
            if ious is not None:
                ious[truth_index][prediction_index] = intersection_over_union
        if intersection_over_union < threshold:
            continue
        if intersection_over_union > best_match_intersection_over_union:
            best_match_intersection_over_union = intersection_over_union
            best_match_index = truth_index
    return best_match_index

def calculate_precision(truth, predictions, threshold = 0.5, ious = None):
    n, true_positive, false_positive = len(predictions), 0, 0
    for prediction_index in range(n):
        best_match_truth_idx = find_best_match(truth, predictions[prediction_index], prediction_index, threshold = threshold, ious = ious)
        if best_match_truth_idx >= 0:
            true_positive += 1
            truth[best_match_truth_idx] = -1
        else:
            false_positive += 1
    return true_positive / (true_positive + false_positive)

def calculate_image_precision(truth, predictions, threshold = 0.5):
    image_precision = 0.0
    intersection_over_unions = np.ones((len(truth), len(predictions))) * -1
    precision_at_threshold = calculate_precision(truth.copy(), predictions, threshold = threshold, ious = intersection_over_unions)
    image_precision += precision_at_threshold
    return image_precision

def format_prediction_string(boxes, scores):
    prediction_strings = []
    for i in zip(scores, boxes):
        prediction_strings.append("{0:.4f} {1} {2} {3} {4}".format(i[0], int(i[1][0]), int(i[1][1]), int(i[1][2]), int(i[1][3])))
    return " ".join(prediction_strings)

def annotate_and_validate(dataloader, model, device, threshold):
    model.eval()
    results, test_precision = [], []
    with torch.no_grad():
        for i, (images, targets, image_ids) in tqdm(enumerate(dataloader), total = len(dataloader)):
            outputs = model(images)
            outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]
            images = list(image.to(device) for image in images)
            targets = [{k: torch.from_numpy(v).to(device) for k, v in t.items()} for t in targets]
            for i, _ in enumerate(images):
                boxes = outputs[i]['boxes'].data.numpy()
                labels = outputs[i]['labels'].data.numpy()
                truth_boxes = targets[i]['boxes'].numpy()
                preds_sorted_idx = np.argsort(labels)[::-1]
                preds_sorted = boxes[preds_sorted_idx]
                image_precision = calculate_image_precision(preds_sorted, truth_boxes, threshold = threshold)
                boxes = boxes[labels >= threshold].astype(np.int32)
                draw_boxes = boxes.copy()
                boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
                boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
                for box in draw_boxes:
                    cv2.rectangle(images, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (220, 0, 0), 3)
                plt.imshow(cv2.cvtColor(images, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.savefig(f"test_images_bbox/{images[i]}")
                plt.close()
            result = {'patientId': image_ids, 'PredictionString': format_prediction_string(boxes, labels) if len(outputs[0]['boxes']) != 0 else None}
            results.append(result)
            test_precision.append(image_precision)
            precision = np.mean(test_precision)
            submission_dataframe = pd.DataFrame(results, columns = ['patientId', 'PredictionString'])
            submission_dataframe.to_csv('submission.csv', index = False)
    return precision

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model().to(device)
annotate_and_validate(test_data_loader, model, device, 0.5)
