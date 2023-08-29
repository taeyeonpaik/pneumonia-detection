import time
from tqdm import tqdm

def train(dataloader, learning_rate_scheduler, model, optimizer, device, loss_history):
    model.train()
    start = time.time()
    loss_history.reset()
    progress_bar = tqdm(total = len(dataloader))
    for i, (images, targets, _) in enumerate(dataloader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dictionary = model(images, targets)
        losses = sum(loss for loss in loss_dictionary.values())
        loss_value = losses.item()
        loss_history.send(loss_value)
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        progress_bar.update()
    end = time.time()
    return loss_history, end, start
