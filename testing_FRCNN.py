import os
import cv2
import math
import time
import torch
import numpy as np
import pandas as pd
from model_FRCNN import model
import matplotlib.pyplot as plt
from pipeline_FRCNN import train
from dataset import get_data_loader

# TODO: TAKE OUT BOX THRESHOLD
# 2 parts for precision metric
# 1: Confusion matrix with tp, fp, tn, fn
# 2: Out of those tps, calculate mean ious
# Fix iou printing -1     
# Fix annotate and cf matrix methods

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

def calculate_iou(truth, prediction):
    # Calculate width of overlap between truth and prediction boxes
    width = min(truth[2], prediction[2]) - max(truth[0], prediction[0])
    height = (min(truth[3], prediction[3]) - max(truth[1], prediction[1]))
    # Calculate area of both boxes
    truth_area = (truth[2] - truth[0]) * (truth[3] - truth[1])
    prediction_area = (prediction[2] - prediction[0]) * (prediction[3] - prediction[1])
    return 0.0 if width <= 0 or height <= 0 else width * height / float(truth_area + prediction_area - width * height)

def calculate_image_iou(box_truth, score_truth, box_predictions, score_predictions, score_threshold, true_positive, false_positive, true_negative, false_negative):
    n, best_iou, best_match_score = len(box_predictions), -1, -1
    # Finding the best iou based on predicted score
    for prediction_index in range(n):
        if score_predictions[prediction_index] > best_match_score:
            best_match_score = score_predictions[prediction_index]
            best_iou = calculate_iou(box_truth, box_predictions[prediction_index])
    score = 1 if score_predictions[prediction_index] >= score_threshold else 0
    # Evaluating status of prediction
    if score_truth == 1 and score == 1: 
        true_positive += 1
    elif score_truth == 0 and score == 1:
        false_positive += 1
    elif score_truth == 0 and score == 0:
        true_negative += 1
    else:
        false_negative += 1
    return true_positive, false_positive, true_negative, false_negative, best_iou if score_truth == 1 and score == 1 else math.nan

def validate(dataloader, model, device, score_threshold, epoch, type):
    valid_image_iou, true_positive, false_positive, true_negative, false_negative = [], 0, 0, 0, 0
    model.eval()
    start = time.time()
    with torch.no_grad():
        for j, (images, targets, _) in enumerate(dataloader):
            batch_start = time.time()
            # Move images and target annotations to specified device
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)
            # Iterate through individual images in batch
            for i, _ in enumerate(images):
                # Extract predicted bounding boxes and scores for image
                predicted_boxes = outputs[i]['boxes'].data.cpu().numpy()
                predicted_scores = outputs[i]['scores'].data.cpu().numpy()
                # Extract ground truth bounding boxes for image
                truth_box = targets[i]['boxes'].cpu().numpy()
                truth_score = targets[i]['labels'].cpu().numpy()
                # Calculate image iou for current image
                true_positive, false_positive, true_negative, false_negative, image_iou = calculate_image_iou(truth_box, truth_score, predicted_boxes, predicted_scores, score_threshold, true_positive, false_positive, true_negative, false_negative)
                # Append image iou to list if not NaN
                if not math.isnan(image_iou):
                    valid_image_iou.append(image_iou)
            batch_end = time.time()
            print(f"Epoch #{epoch + 1}, Batch #{j + 1}/{len(dataloader)}, {type} mIoU: {np.mean(valid_image_iou):4f}, Time: {(batch_end - batch_start) / 60:.3f} Minutes", end = '\r')
        # Calculate mean iou across all images
        mean_iou = np.mean(valid_image_iou)
    end = time.time()
    print(f"\nEpoch #{epoch + 1}, {type} mIoU: {mean_iou:.4f}, Time: {(end - start) / 60:.3f} Minutes")
    return true_positive, false_positive, true_negative, false_negative, mean_iou

def annotate(model, device, train_loss, train_iou_history, valid_iou_history, test_iou_history, threshold):
    test_images = os.listdir(f"/home/ec2-user/rsna/test_images_png") # "/Users/taeyeonpaik/Downloads/rsna/test_images_png/"
    model.to(device).eval()
    results = []
    with torch.no_grad():
        for i, image in enumerate(test_images):
            batch_start = time.time()
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
            batch_end = time.time()
            print(f"Image #{i + 1}/{len(test_images)}, Time: {(batch_end - batch_start) / 60:.3f} Minutes", end = '\r')
    submission_dataframe = pd.DataFrame(results, columns = ['patientId', 'PredictionString'])
    # Training loss figure
    plt.figure()
    plt.plot(train_loss, label = 'Training Loss')
    plt.legend()
    plt.show()
    plt.savefig(f"/home/ec2-user/rsna/loss.png") # /Users/taeyeonpaik/Downloads/rsna/loss.png
    plt.figure()
    # Iou figure
    plt.plot(train_iou_history, label = 'Train mIoU')
    plt.plot(valid_iou_history, label = 'Valid mIoU')
    plt.plot(test_iou_history, label = 'Test mIoU')
    plt.legend()
    plt.show()
    plt.savefig(f"/home/ec2-user/rsna/iou.png") # /Users/taeyeonpaik/Downloads/rsna/iou.png
    return submission_dataframe

def confusion_matrix(true_positives, false_positives, true_negatives, false_negatives, truth, prediction):
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
    plt.savefig(f"/home/ec2-user/rsna/confusion_matrix.png")
    auroc = roc_auc_score(truth, prediction)
    print("AUROC:", auroc)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
loss_history = Averager()
model = model().to(device)
total_epochs, batch_size, score_threshold = 2, 16, 0.5
params = [p for p in model.parameters() if p.requires_grad]
train_data_loader, valid_data_loader, test_data_loader = get_data_loader(batch_size)
optimizer = torch.optim.SGD(params, lr = 0.005, momentum = 0.9, weight_decay = 0.0005)
learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 4, gamma = 0.1)

train_loss, train_iou_history, valid_iou_history, test_iou_history = [], [], [], []
for epoch in range(total_epochs):
    train_loss_history, end, start = train(train_data_loader, learning_rate_scheduler, model, optimizer, device, loss_history, epoch + 1)
    print(f"\nEpoch #{epoch + 1}, Loss: {train_loss_history.value:.4f}, Time: {(end - start) / 60:.3f} Minutes")
    train_tp, train_fp, train_tn, train_fn, train_iou = validate(train_data_loader, model, device, score_threshold, epoch, "Train")
    valid_tp, valid_fp, valid_tn, valid_fn, valid_iou = validate(valid_data_loader, model, device, score_threshold, epoch, "Valid")
    test_tp, test_fp, test_tn, test_fn, test_iou = validate(test_data_loader, model, device, score_threshold, epoch, "Test")
    train_loss.append(train_loss_history.value)
    train_iou_history.append(train_iou)
    valid_iou_history.append(valid_iou)
    test_iou_history.append(test_iou)
    if learning_rate_scheduler is not None:
        learning_rate_scheduler.step()
    # if epoch == total_epochs - 1:
    #     annotate(model, device, train_loss, train_iou_history, valid_iou_history, test_iou_history, threshold = 0.75)
    #     confusion_matrix(test_tp, test_fp, test_tn, test_fn, truths, predictions)
