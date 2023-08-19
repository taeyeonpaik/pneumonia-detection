import torch
import numpy as np
from torch import nn
from d_model import CNN
from torch import optim
from torch.autograd import Variable
from c_dataset import get_data_loader

def train_model():
    cnn.train()
    loss_function_mean_squared_error, loss_function_classification = nn.MSELoss(), nn.BCELoss()
    optimizer = optim.Adam(cnn.parameters(), lr = 0.001)
    train_loader, epochs = get_data_loader()[0], 1 # epochs = 20
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}, loss =", end = ' ')
        loss_list_class, loss_list_box = [], []
        for i, (images, labels, boxes) in tqdm(enumerate(train_loader), len = len(train_loader)):
            images, labels = Variable(images), Variable(labels)
            predicted = cnn(images)
            score = torch.sigmoid(predicted[:, 0])
            predicted_box = predicted[:, 1:5]
            loss_class = loss_function_classification(score.float(), labels.float())
            loss_box = loss_function_mean_squared_error(box[labels == 1], predicted_box[labels == 1])
            optimizer.zero_grad()
            loss = 1000 * loss_class
            loss.backward() # retain_graph = True
            optimizer.step()
            loss_list_class.append(loss_class.item())
            loss_list_box.append(loss_box.item())
            print(f"Epoch #{epoch + 1}, Batch #{i + 1}, Loss: {loss_class.item()} and {loss_box.item()}")
            if i == 0: # Delete
                break # Delete
        print(f"Epoch #{epoch + 1}, Loss: {sum(loss_list_class) / len(loss_list_class)} and {sum(loss_list_box) / len(loss_list_box}")

if __name__ == '__main__':
    cnn = CNN()
    train_model()
    