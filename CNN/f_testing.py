import torch
import numpy as np
from d_model import CNN
import matplotlib.pyplot as plt
from e_pipeline import train_model
from c_dataset import get_data_loader
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

def testing(epoch):
    model.eval()
    with torch.no_grad():
        true_positives, false_positives, true_negatives, false_negatives, threshold = 0, 0, 0, 0, 0.9
        box_truth, label_truth, box_prediction_list, label_prediction_list = [], [], [], []
        for image, label, box in get_data_loader()[2]:
            output = model(image)
            label_prediction, box_prediction = 1 if output[0][0] >= threshold else 0, output[0][1:5]
            if label == 1 and label_prediction == 1 and box_prediction == box:
                true_positives += 1
            elif label == 0 and label_prediction == 1:
                false_positives += 1
            elif label == 0 and label_prediction == 0:
                true_negatives += 1
            elif label == 1 and label_prediction == 0:
                false_negatives += 1
            box_truth.append(box)
            label_truth.append(label)
            box_prediction_list.append(box_prediction)
            label_prediction_list.append(label_prediction)
            accuracy = (label_prediction == label and box_prediction == box).sum().item() / float(label.size(0))
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1_score = 2 * (precision * recall) / (precision + recall)
        print(f'Epoch #Accuracy: {accuracy:.2f}')
        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')
        print(f'F1 score: {f1_score:.2f}')
        print(true_positives, false_positives, true_negatives, false_negatives)

loss_function_mean_squared_error, loss_function_classification = nn.MSELoss(), nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)
train_loader, epochs = get_data_loader()[0], 20
loss_list_class, loss_list_box, losses = [], [], []

model = CNN()
model.to('cuda')
for epoch in range(epochs):
    train_model(model, loss_list_class, loss_list_box, losses, train_loader)
    print(f"Epoch #{epoch + 1}, Loss: {sum(losses) / len(losses)}")

    testing(epoch)
    # print_confusion_matrix(true_positives, false_positives, true_negatives, false_negatives, np.array(label_truth), np.array(label_prediction_list))
