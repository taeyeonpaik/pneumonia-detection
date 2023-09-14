import time
from tqdm import tqdm

def train(dataloader, learning_rate_scheduler, model, optimizer, device, loss_history, epoch):
    model.train()
    start = time.time()
    loss_history.reset()
    progress_bar = tqdm(total = len(dataloader))
    for images, targets, _ in dataloader:
        batch_start = time.time()
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        print(targets.size)
        loss_dictionary = model(images, targets)
        losses = sum(loss for loss in loss_dictionary.values())
        loss_value = losses.item()
        loss_history.send(loss_value)
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        progress_bar.update()
        batch_end = time.time()
        print(f"Epoch #{epoch}, Loss: {loss.item():.4f}, Time: {(batch_end - batch_start) / 60:.3f} Minutes", end = '\r')
    end = time.time()
    return loss_history, end, start
