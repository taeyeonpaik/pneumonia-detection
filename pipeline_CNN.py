import time
import torch
import numpy as np
from tqdm import tqdm

def train(dataloader, model, optimizer, device, loss_history, loss_function_class, loss_function_box, epoch):
    model.train()
    start = time.time()
    loss_history.reset()
    progress_bar = tqdm(total = len(dataloader))
    for images, targets, _ in dataloader:
        batch_start = time.time()
        images = torch.stack([image.to(device) for image in images])
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        predicted = model(images)
        predicted_label = torch.sigmoid(predicted[:, 0])
        predicted_box = predicted[:, 1:5]
        labels = torch.Tensor([target['labels'] for target in targets]).to(device)
        boxes = torch.stack([target['boxes'] for target in targets]).to(device)
        boxes = boxes[labels.bool()]
        predicted_box = predicted_box[labels.bool()]
        loss_class = loss_function_class(predicted_label.float(), labels)
        loss_box = loss_function_box(predicted_box, boxes)
        loss = loss_class + 0.00001 * loss_box
        loss_history.send(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        progress_bar.update()
        batch_end = time.time()
        print(f"Epoch #{epoch}, Loss: {loss.item():.4f}, Time: {(batch_end - batch_start) / 60:.3f} Minutes", end = '\r')
    end = time.time()
    return loss_history, end, start
