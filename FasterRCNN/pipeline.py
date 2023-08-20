import time
from tqdm import tqdm

def train(dataloader, learning_rate_scheduler, model, optimizer, device, epoch, loss_history):
    model.train()
    start = time.time()
    loss_history.reset()
    for i, (images, targets, _) in tqdm(enumerate(dataloader), total = len(dataloader)):
        start_batch = time.time()
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dictionary = model(images, targets)
        losses = sum(loss for loss in loss_dictionary.values())
        loss_value = losses.item()
        loss_history.send(loss_value)
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        end_batch = time.time()
        print(f"Epoch #{epoch}, Batch #{i + 1}, Loss: {loss_value}, Time: {(end_batch - start_batch) / 60:.3f} Minutes")
    if learning_rate_scheduler is not None:
        learning_rate_scheduler.step()
    end = time.time()
    return loss_history, end, start
