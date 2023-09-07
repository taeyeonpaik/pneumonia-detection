import time
import torch
import numpy as np
from tqdm import tqdm

def train(dataloader, model, optimizer, device, loss_history, loss_function_class, loss_function_box):
    model.train()
    start = time.time()
    loss_history.reset()
    progress_bar = tqdm(total = len(dataloader))
    for images, targets, _ in dataloader:
        images = torch.stack([image.to(device) for image in images])
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Prediction
        predicted = model(images)
        predicted_label = torch.sigmoid(predicted[:, 0])
        predicted_box = predicted[:, 1:5]

        # Ground truth
        labels = torch.Tensor([target['labels'] for target in targets]).to(device)
        boxes = torch.stack([target['boxes'] for target in targets]).to(device)

        # Filter out bbox for negative samples
        mask = labels.bool()
        boxes = boxes[mask]
        predicted_box = predicted_box[mask]
        
        # Compute loss
        loss_class = loss_function_class(predicted_label.float(), labels)
        loss_box = loss_function_box(predicted_box, boxes)
        loss = loss_class + 0.00001 * loss_box

        loss_history.send(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        progress_bar.update()
    end = time.time()
    return loss_history, end, start
