import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
from torch import optim
from model_CNN import CNN
from pipeline_CNN import train
import matplotlib.pyplot as plt
from dataset import get_data_loader
from sklearn.metrics import roc_auc_score

def print_confusion_matrix(true_positives, false_positives, true_negatives, false_negatives, truth, prediction):
    confusion_matrix = np.array([[true_negatives, false_positives], [false_negatives, true_positives]])
    plt.imshow(confusion_matrix, interpolation = 'nearest', cmap = 'BuGn')
    tick_marks = np.arange(len(['Negative', 'Positive']))
    plt.colorbar()
    plt.title('Confusion Matrix')
    plt.xticks(tick_marks, ['Negative', 'Positive'], rotation = 45)
    plt.yticks(tick_marks, ['Negative', 'Positive'])
    threshold = confusion_matrix.max() / 2.0
    for i, j in np.ndindex(confusion_matrix.shape):
        plt.text(j, i, format(confusion_matrix[i, j], 'd'), horizontalalignment = "center", color = "black" if confusion_matrix[i, j] > threshold else "black")
    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    auroc = roc_auc_score(truth, prediction)
    print("AUROC:", auroc)

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

def calculate_image_precision(truths, predictions, thresholds):
    intersection_over_unions, image_precision = np.ones((len(truths), len(predictions))) * -1, 0.0
    for threshold in thresholds:
        true_positive, false_positive = 0, 0
        for prediction_index in range(len(predictions)):
            best_match_truth_idx = best_match(truths.copy(), predictions[prediction_index], prediction_index, threshold = threshold, ious = intersection_over_unions)
            if best_match_truth_idx >= 0:
                true_positive += 1
                truths[best_match_truth_idx] = -1
            else:
                false_positive += 1 
        false_negative = (truths.sum(axis = 1) > 0).sum()
        precision_at_threshold = true_positive / (true_positive + false_positive + false_negative)
        image_precision += precision_at_threshold / len(thresholds)
    return image_precision

def format_prediction_string(boxes, scores):
    prediction_strings = []
    for i in zip(scores, boxes):
        prediction_strings.append("{0:.4f} {1} {2} {3} {4}".format(i[0], int(i[1][0]), int(i[1][1]), int(i[1][2]), int(i[1][3])))
    return " ".join(prediction_strings)

def validate(dataloader, model, device, thresholds):
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
                image_precision = calculate_image_precision(predictions_sorted, truth_boxes, thresholds)
                valid_image_precision.append(image_precision)
        precision = np.mean(valid_image_precision)
    return precision

def annotate(model, device, train_loss, precision_history, threshold):
    test_images = os.listdir(f"/home/ec2-user/rsna/test_images_png") #"/Users/taeyeonpaik/Downloads/rsna/test_images_png"
    model.to(device).eval()
    results = []
    with torch.no_grad():
        for i, image in tqdm(enumerate(test_images), total = len(test_images)):
            original_image = cv2.imread(f"/home/ec2-user/rsna/test_images_png/{test_images[i]}", cv2.IMREAD_COLOR) #/Users/taeyeonpaik/Downloads/rsna/test_images_png/
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
            plt.savefig(f"/home/ec2-user/rsna/test_images_bbox/{test_images[i]}") #/Users/taeyeonpaik/Downloads/rsna/test_images_bbox/
            plt.close()
            result = {'patientId': test_images[i].split('.')[0], 'PredictionString': format_prediction_string(boxes, scores) if len(outputs[0]['boxes']) != 0 else None}
            results.append(result)
    submission_dataframe = pd.DataFrame(results, columns = ['patientId', 'PredictionString'])
    plt.figure()
    plt.plot(train_loss, label = 'Training Loss')
    plt.legend()
    plt.show()
    plt.savefig(f"/home/ec2-user/rsna/loss.png") #/Users/taeyeonpaik/Downloads/rsna/loss.png
    plt.figure()
    plt.plot(precision_history, label = 'Testing Precision')
    plt.legend()
    plt.show()
    plt.savefig(f"/home/ec2-user/rsna/precision.png") #/Users/taeyeonpaik/Downloads/rsna/precision.png
    return submission_dataframe

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr = 0.001)
mean_squared_error, binary_cross_entropy, loss_history = nn.MSELoss(), nn.BCELoss(), Averager()
total_epochs, batch_size, thresholds = 10, 16, (0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75)
train_data_loader, valid_data_loader, test_data_loader = get_data_loader(batch_size)

train_loss, precision_history = [], []
for epoch in range(total_epochs):
    train_loss_history, end, start = train(train_data_loader, model, optimizer, device, loss_history, binary_cross_entropy, mean_squared_error)
    print(f"Epoch #{epoch + 1}, Loss: {train_loss_history.value}, Time: {(end - start) / 60:.3f} Minutes")
    precision = validate(test_data_loader, model, device, thresholds)
    print(f"Epoch #{epoch + 1}, Precision: {precision}")
    train_loss.append(train_loss_history.value)
    precision_history.append(precision)
annotate(model, device, train_loss, precision_history, threshold = 0.9)
