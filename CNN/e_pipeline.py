import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.autograd import Variable
from c_dataset import get_data_loader

def train_model(model, loss_list_class, loss_list_box, losses, train_loader):
    model.train()
        for i, (images, labels, boxes) in tqdm(enumerate(train_loader), total = len(train_loader)):
            predicted = model(images)
            score = torch.sigmoid(predicted[:, 0])
            predicted_box = predicted[:, 1:5]
            loss_class = loss_function_classification(score.float(), labels.float())
            loss_box = loss_function_mean_squared_error(predicted_box[labels == 1], boxes[labels == 1])
            optimizer.zero_grad()
            loss = loss_class + loss_box
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            loss_list_class.append(loss_class.item())
            loss_list_box.append(loss_box.item())
            print(f"Epoch #{epoch + 1}, Batch #{i + 1}, Loss: {loss}")
    