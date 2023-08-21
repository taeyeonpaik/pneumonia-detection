import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from model import model
from pipeline import train
import matplotlib.pyplot as plt
from dataset import get_data_loader

class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        return 0 if self.iterations == 0 else 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

def best_match(truth, prediction, prediction_index, threshold = 0.5, ious = None):
    best_match_intersection_over_union, best_match_index = -np.inf, -1
    for truth_index in range(len(truth)):
        if truth[truth_index][0] < 0:
            continue
        intersection_over_union = -1 if ious is None else ious[truth_index][prediction_index]
        if intersection_over_union < 0:
            width = min(truth[truth_index][2], prediction[2]) - max(truth[truth_index][0], prediction[0]) + 1
            height = min(truth[truth_index][3], prediction[3]) - max(truth[truth_index][1], prediction[1]) + 1
            union_area = ((truth[truth_index][2] - truth[truth_index][0] + 1) * (truth[truth_index][3] - truth[truth_index][1] + 1) + (prediction[2] - prediction[0] + 1) * (prediction[3] - prediction[1] + 1) - width * height)
            intersection_over_union = 0.0 if width < 0 or height < 0 else width * height / union_area
            if ious is not None:
                ious[truth_index][prediction_index] = intersection_over_union
        if intersection_over_union < threshold:
            continue
        if intersection_over_union > best_match_intersection_over_union:
            best_match_intersection_over_union = intersection_over_union
            best_match_index = truth_index
    return best_match_index

def calculate_image_precision(truth, predictions, threshold = 0.5):
    intersection_over_unions = np.ones((len(truth), len(predictions))) * -1
    image_precision, n, true_positive, false_positive = 0.0, len(predictions), 0, 0
    for prediction_index in range(n):
        best_match_truth_idx = best_match(truth.copy(), predictions[prediction_index], prediction_index, threshold = threshold, ious = intersection_over_unions)
        if best_match_truth_idx >= 0:
            true_positive += 1
            truth[best_match_truth_idx] = -1
        else:
            false_positive += 1    
    precision_at_threshold = true_positive / (true_positive + false_positive)
    image_precision += precision_at_threshold
    return image_precision

def format_prediction_string(boxes, scores):
    prediction_strings = []
    for i in zip(scores, boxes):
        prediction_strings.append("{0:.4f} {1} {2} {3} {4}".format(i[0], int(i[1][0]), int(i[1][1]), int(i[1][2]), int(i[1][3])))
    return " ".join(prediction_strings)

def validate(dataloader, model, device, threshold):
    valid_image_precision = []
    model.eval()
    with torch.no_grad():
        for images, targets, _ in dataloader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)
            for i, _ in enumerate(images):
                boxes = outputs[i]['boxes'].data.cpu().numpy()
                scores = outputs[i]['scores'].data.cpu().numpy()
                truth_boxes = targets[i]['boxes'].cpu().numpy()
                preds_sorted_idx = np.argsort(scores)[::-1]
                predictions_sorted = boxes[preds_sorted_idx]
                image_precision = calculate_image_precision(predictions_sorted, truth_boxes, threshold = threshold)
                valid_image_precision.append(image_precision)
        precision = np.mean(valid_image_precision)
    return precision

def annotate(model, device, threshold):
    test_images = os.listdir(f"/home/ec2-user/rsna/test_images_png")
    model.to(device).eval()
    results = []
    with torch.no_grad():
        for i, image in tqdm(enumerate(test_images), total = len(test_images)):
            original_image = cv2.imread(f"/home/ec2-user/rsna/test_images_png/{test_images[i]}", cv2.IMREAD_COLOR)
            image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            image = torch.tensor(image, dtype = torch.float).cuda()
            image = torch.unsqueeze(image, 0)
            outputs = [{k: v.to(device) for k, v in t.items()} for t in model(image)]
            for _ in range(len(outputs[0]['boxes'])):
                boxes = outputs[0]['boxes'].data.cpu().numpy()
                scores = outputs[0]['scores'].data.cpu().numpy()
                boxes = boxes[scores >= threshold]
                draw_boxes = boxes.copy()
                boxes[:, 2], boxes[:, 3] = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
            for box in draw_boxes:
                cv2.rectangle(original_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 3)
            plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.savefig(f"/home/ec2-user/rsna/test_images_bbox/{test_images[i]}")
            plt.close()
            result = {'patientId': test_images[i].split('.')[0], 'PredictionString': format_prediction_string(boxes, scores) if len(outputs[0]['boxes']) != 0 else None}
            results.append(result)
    submission_dataframe = pd.DataFrame(results, columns = ['patientId', 'PredictionString'])
    return submission_dataframe

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
loss_history = Averager()
model = model().to(device)
total_epochs, batch_size, threshold = 10, 32, 0.9
params = [p for p in model.parameters() if p.requires_grad]
train_data_loader, valid_data_loader, test_data_loader = get_data_loader(batch_size)
optimizer = torch.optim.SGD(params, lr = 0.005, momentum = 0.9, weight_decay = 0.0005)
learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 4, gamma = 0.1)

for epoch in range(total_epochs):
    train_loss_history, end, start = train(train_data_loader, learning_rate_scheduler, model, optimizer, device, epoch + 1, loss_history)
    print(f"Epoch #{epoch + 1}, Loss: {train_loss_history.value}, Time: {(end - start) / 60:.3f} Minutes")
    precision = validate(test_data_loader, model, device, threshold)
    print(f"Epoch #{epoch + 1}, Precision: {precision}")
annotate(model, device, threshold)
