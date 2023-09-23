import os
import cv2
import time
import torch
import numpy as np
import pandas as pd
from model_FRCNN import model
import matplotlib.pyplot as plt
from pipeline_FRCNN import train
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

def calculate_intersection_over_union(truth, prediction):
    # Calculate width of overlap between truth and prediction boxes
    width = min(truth[2], prediction[2]) - max(truth[0], prediction[0]) + 1
    # Calculate height of overlap between truth and prediction boxes
    height = min(truth[3], prediction[3]) - max(truth[1], prediction[1]) + 1
    # Calculate area of union of truth and prediction boxes
    union_area = ((truth[2] - truth[0] + 1) * (truth[3] - truth[1] + 1) + (prediction[2] - prediction[0] + 1) * (prediction[3] - prediction[1] + 1) - width * height)
    # Compute the iou by dividing overlap area by union area
    # If width or height is less than 0, there is no intersection
    return 0.0 if width < 0 or height < 0 else width * height / union_area

def find_best_match(truth, prediction, prediction_index, threshold = 0.5, ious=None):
    # Initialize variables to track best match and iou score
    best_match_intersection_over_union, best_match_index = -np.inf, -1
    # Iterate through each ground truth box
    for truth_index in range(len(truth)):
        # Check if truth box has already been matched to a prediction
        if truth[0] < 0: # if truth[truth_index, 0] < 0:
            # Already matched ground truth box, skip to next one
            continue
        # Initialize iou score to -1 if not precomputed
        intersection_over_union = -1 if ious is None else ious[truth_index][prediction_index]
        # If iou was not precomputed or is negative, calculate it
        if intersection_over_union < 0:
            intersection_over_union = calculate_intersection_over_union(truth, prediction)
            # Store computed iou in the precomputed list if available
            if ious is not None:
                ious[truth_index][prediction_index] = intersection_over_union
        # If iou is below threshold, skip to next truth box
        if intersection_over_union < threshold:
            continue
        # If iou is higher than best match, update best match information
        if intersection_over_union > best_match_intersection_over_union:
            best_match_intersection_over_union = intersection_over_union
            best_match_index = truth_index
    # Return index of the best-matching ground truth box
    return best_match_index

def calculate_precision(truth, predictions, threshold = 0.5, ious = None):
    n, true_positive, false_positive, false_negative = len(predictions), 0, 0, 0
    for prediction_index in range(n):
        best_match_truth_idx = find_best_match(truth, predictions[prediction_index], prediction_index, threshold = threshold, ious = ious)
        if best_match_truth_idx >= 0:
            # Predicted box matches ground truth box with iou above threshold
            true_positive += 1
            # Remove matched ground truth box
            truth[best_match_truth_idx] = -1
        else:
            # No match and predicted box has no associated ground truth box
            false_positive += 1
    # Ground truth box has not associated predicted box
    false_negative = (np.array(truth) > 0).sum() # false_negative = (np.array(truth).sum(axis = 1) > 0).sum()
    return true_positive / (true_positive + false_positive + false_negative)

def calculate_image_precision(truth, predictions, thresholds = 0.5):
    n_threshold, image_precision = len(thresholds), 0.0
    # Create empty matrix to store iou values initialized to -1




    intersection_over_unions = np.ones((len(truth), len(predictions))) * -1
    # intersection_over_unions = np.ones(len(predictions)) * -1




    # Calculate precisions for each given threshold
    for threshold in thresholds:
        precision_at_threshold = calculate_precision(truth.copy(), predictions, threshold = threshold, ious = intersection_over_unions)
        image_precision += precision_at_threshold / n_threshold
    return image_precision

def validate(dataloader, model, device, thresholds, epoch, type):
    valid_image_precision = []
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
                # Sort predictions by score in descending order
                predictions_sorted_index = np.argsort(predicted_scores)[::-1]
                predictions_sorted = predicted_boxes[predictions_sorted_index]
                # Calculate image precision for current image
                image_precision = calculate_image_precision(truth_box, predicted_boxes, thresholds)
                # Append image precision to list
                valid_image_precision.append(image_precision)
            batch_end = time.time()
            print(f"Epoch #{epoch + 1}, Batch #{j + 1}/{len(dataloader)}, {type} Precision: {np.mean(valid_image_precision):.4f}, Time: {(batch_end - batch_start) / 60:.3f} Minutes", end = '\r')
        # Calculate mean precision across all images
        precision = np.mean(valid_image_precision)
    end = time.time()
    print(f"", end = '\r')
    print(f"Epoch #{epoch + 1}, {type} Precision: {precision:.4f}, Time: {(end - start) / 60:.3f} Minutes")
    return precision

def annotate(model, device, train_loss, train_precision_history, valid_precision_history, test_precision_history, threshold):
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
    plt.figure()
    plt.plot(train_loss, label = 'Training Loss')
    plt.legend()
    plt.show()
    plt.savefig(f"/home/ec2-user/rsna/loss.png") # /Users/taeyeonpaik/Downloads/rsna/loss.png
    plt.figure()
    plt.plot(train_precision_history, label = 'Train Precision')
    plt.plot(valid_precision_history, label = 'Valid Precision')
    plt.plot(test_precision_history, label = 'Test Precision')
    plt.legend()
    plt.show()
    plt.savefig(f"/home/ec2-user/rsna/precision.png") # /Users/taeyeonpaik/Downloads/rsna/precision.png
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

def format_prediction_string(boxes, scores):
    prediction_strings = []
    for i in zip(scores, boxes):
        prediction_strings.append("{0:.4f} {1} {2} {3} {4}".format(i[0], int(i[1][0]), int(i[1][1]), int(i[1][2]), int(i[1][3])))
    return " ".join(prediction_strings)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
loss_history = Averager()
model = model().to(device)
total_epochs, batch_size, thresholds = 2, 16, (0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75)
params = [p for p in model.parameters() if p.requires_grad]
train_data_loader, valid_data_loader, test_data_loader = get_data_loader(batch_size)
optimizer = torch.optim.SGD(params, lr = 0.005, momentum = 0.9, weight_decay = 0.0005)
learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 4, gamma = 0.1)

train_loss, train_precision_history, valid_precision_history, test_precision_history = [], [], [], []
for epoch in range(total_epochs):
    # train_loss_history, end, start = train(train_data_loader, learning_rate_scheduler, model, optimizer, device, loss_history, epoch + 1)
    # print(f"", end = '\r')
    # print(f"Epoch #{epoch + 1}, Loss: {train_loss_history.value:.4f}, Time: {(end - start) / 60:.3f} Minutes")
    train_precision = validate(train_data_loader, model, device, thresholds, epoch, "Train")
    valid_precision = validate(valid_data_loader, model, device, thresholds, epoch, "Valid")
    test_precision = validate(test_data_loader, model, device, thresholds, epoch, "Test")
    train_loss.append(train_loss_history.value)
    train_precision_history.append(train_precision)
    valid_precision_history.append(valid_precision)
    test_precision_history.append(test_precision)
    if learning_rate_scheduler is not None:
        learning_rate_scheduler.step()
# annotate(model, device, train_loss, train_precision_history, valid_precision_history, test_precision_history, threshold = 0.75)
# confusion_matrix(true_positives, false_positives, true_negatives, false_negatives, truth, prediction)
