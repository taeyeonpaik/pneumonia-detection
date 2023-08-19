import torch
import numpy as np
import matplotlib.pyplot as plt
from c_dataset import get_data_loader
from e_pipeline import train_model, cnn
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

train_model()
cnn.eval()
with torch.no_grad():
    true_positives, false_positives, true_negatives, false_negatives, threshold, truth, prediction = 0, 0, 0, 0, 0.5, [], [] # threshold = 0.9
    for image, label, box in get_data_loader()[1]:
        output = cnn(image)
        label_prediction, box_prediction = 1 if output[0] >= threshold else , output[1:5]
        if label == 1 and label_prediction == 1 and box_prediction == box:
            true_positives += 1
        elif label == 0 and label_prediction == 1:
            false_positives += 1
        elif label == 0 and label_prediction == 0:
            true_negatives += 1
        elif label == 1 and label_prediction == 0:
            false_negatives += 1
        truth.append(label)
        prediction.append(label_prediction)
        accuracy = (label_prediction == label).sum().item() / float(label.size(0))
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)
    print(f'Accuracy: %.2f {accuracy}')
    print(f'Precision: %.2f {precision}')
    print(f'Recall: %.2f {recall}')
    print(f'F1 score: %.2f {f1_score}')
    print_confusion_matrix(true_positives, false_positives, true_negatives, false_negatives, truth, prediction)
