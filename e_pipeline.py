"""docstring"""
import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from torch import nn
from d_model import CNN
from c_dataset import get_data_loader

cnn = CNN()

def train_model():
    """docstring"""
    loss_func_classification = nn.BCELoss()
    loss_func_mse = nn.MSELoss()
    optimizer = optim.Adam(cnn.parameters(), lr = 0.001)
    num_epochs = 10
    cnn.train()
    train_loader = get_data_loader()[0]
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}, loss =", end = ' ')
        loss_list_class, loss_list_bbox = [], []
        for (imgs, labels, bbox) in train_loader:
            imgs, labels = Variable(imgs), Variable(labels)
            predicted = cnn(imgs)
            print('PREDICTED', predicted)
            score = predicted[:, 0]
            # score = torch.sigmoid(score)
            print('SCORE', score.float())
            predicted_bbox = predicted[:, 1:5]
            loss_class = loss_func_classification(score.float(), labels.float())
            print(labels)
            mask = labels == 1
            print(mask)
            loss_bbox = loss_func_mse(bbox[mask], predicted_bbox[mask])
            optimizer.zero_grad()
            loss = 1000 * loss_class
            loss.backward(retain_graph = True)
            print(loss_class)
            print(loss_bbox)
            optimizer.step()
            loss_list_class.append(loss_class.item())
            loss_list_bbox.append(loss_bbox.item())
        print(np.array(loss_list_class).mean())
        print(np.array(loss_list_bbox).mean())
    print(cnn)

if __name__ == '__main__':
    train_model()
    