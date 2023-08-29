import time
import torch
from tqdm import tqdm

def train(dataloader, model, optimizer, device, loss_history, loss_function_class, loss_function_box):
    model.train()
    start = time.time()
    loss_history.reset()
    progress_bar = tqdm(total = len(dataloader))
    for i, (images, targets, _) in enumerate(dataloader):
        images = list(image.to(device) for image in images)
        predicted = model(images)
        predicted_label = torch.sigmoid(predicted[:, 0])
        predicted_box = predicted[:, 1:5]
        loss_class = loss_function_class(predicted_label.float(), targets['labels'].tolist())
        loss_box = loss_function_box(predicted_box, targets['boxes'].tolist())
        loss = loss_class + 0.00001 * loss_box
        loss_history.send(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        progress_bar.update()
    end = time.time()
    return loss_history, end, start
