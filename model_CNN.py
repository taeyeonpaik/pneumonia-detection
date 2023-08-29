from torch import nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 5, stride = 1, padding = 2,), nn.ReLU(), nn.MaxPool2d(2),)
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 5, stride = 1, padding = 2,), nn.ReLU(), nn.MaxPool2d(2),)
        self.out = nn.Linear(256 * 256 * 32, 5)

    def forward(self, input_data):
        input_data = self.conv1(input_data)
        input_data = self.conv2(input_data)
        input_data = input_data.view(input_data.size(0), -1)
        output = self.out(input_data)
        return output
