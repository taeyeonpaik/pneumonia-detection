import torch
from model import model
from dataset import get_data_loader

class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        return 0 if self.iterations == 0 else 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

def train(dataloader, learning_rate_scheduler, model, optimizer, loss_history):
    model.train()
    loss_history.reset()
    for images, targets, _ in dataloader:
        # images = list(image.to(device) for image in images)
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dictionary = model(images, targets)
        losses = sum(loss for loss in loss_dictionary.values())
        loss_value = losses.item()
        loss_history.send(loss_value)
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    if learning_rate_scheduler is not None:
        learning_rate_scheduler.step()
    return loss_history

model = model()
loss_history = Averager()
num_epochs, batch_size, threshold = 30, 8, 0.5
params = [p for p in model.parameters() if p.requires_grad]
train_data_loader, test_data_loader = get_data_loader(batch_size)
optimizer = torch.optim.SGD(params, lr = 0.005, momentum = 0.9, weight_decay = 0.0005)
learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 4, gamma = 0.1)

train_loss = []
for epoch in range(num_epochs):
    train_loss_history = train(train_data_loader, learning_rate_scheduler, model, optimizer, loss_history)
    print(f"Epoch #{epoch} Train loss: {train_loss_history.value}")
    train_loss.append(train_loss_history.value)
