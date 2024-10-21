import time
from contextlib import nullcontext

import torch
from tqdm import tqdm


def autocast_context(use_amp):
    return torch.cuda.amp.autocast() if use_amp else nullcontext()


def train(dataloader, model, optimizer, device, loss_history, scaler=None, use_amp=False):
    model.train()
    start = time.time()
    loss_history.reset()
    non_blocking = device.type == "cuda"

    for images, targets, _ in tqdm(dataloader, total=len(dataloader), desc="Training", leave=False):
        images = [image.to(device, non_blocking=non_blocking) for image in images]
        targets = [
            {key: value.to(device, non_blocking=non_blocking) for key, value in target.items()}
            for target in targets
        ]

        with autocast_context(use_amp):
            loss_dictionary = model(images, targets)
            losses = sum(loss for loss in loss_dictionary.values())
        loss_history.send(losses.item())

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None and use_amp:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

    end = time.time()
    return loss_history, end, start
