"""docstring"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from e_pipeline import train_model, cnn
from c_dataset import get_data_loader

train_model()
cnn.eval()
with torch.no_grad():
    TRUE_POSITIVES = 0
    FALSE_POSITIVES = 0
    TRUE_NEGATIVES = 0
    FALSE_NEGATIVES = 0
    THRESHOLD = 0.5
    true, pred = [], []
    for img, label, bbox in get_data_loader()[1]:
        output, last_layer = cnn(img)
        predicted = torch.Tensor.max(output, 1)[1].data.squeeze()
        if label == 1 and predicted >= THRESHOLD:
            TRUE_POSITIVES += 1
        elif label == 0 and predicted >= THRESHOLD:
            FALSE_POSITIVES += 1
        elif label == 0 and predicted < THRESHOLD:
            TRUE_NEGATIVES += 1
        elif label == 1 and predicted < THRESHOLD:
            FALSE_NEGATIVES += 1
        true.append(label)
        pred.append(predicted)
        accuracy = (predicted == label).sum().item() / float(label.size(0))
    precision = TRUE_POSITIVES / (TRUE_POSITIVES + FALSE_POSITIVES)
    recall = TRUE_POSITIVES / (TRUE_POSITIVES + FALSE_NEGATIVES)
    f1 = 2 * (precision * recall) / (precision + recall)

    cf_matrix = np.array([[TRUE_NEGATIVES, FALSE_POSITIVES], [FALSE_NEGATIVES, TRUE_POSITIVES]])
    classes = ['Negative', 'Positive']
    plt.imshow(cf_matrix, interpolation = 'nearest', cmap = 'BuGn')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)
    threshold = cf_matrix.max() / 2.0
    for i, j in np.ndindex(cf_matrix.shape):
        plt.text(j, i, format(cf_matrix[i, j], 'd'),
                horizontalalignment = "center",
                color = "black" if cf_matrix[i, j] > threshold else "black")
    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

    auroc = roc_auc_score(true, pred)
    print("AUROC:", auroc)

    print(f'Accuracy: %.2f {accuracy}')
    print(f'Precision: %.2f {precision}')
    print(f'Recall: %.2f {recall}')
    print(f'F1 score: %.2f {f1}')
