
from torch import nn
class TinyVGG(nn.Module):
    def __init__(self,input_shape:int,hidden_units:int,output_shape:int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, 3, 1, 1),
            nn.Conv2d(hidden_units, hidden_units, 3, 1, 1),
            nn.MaxPool2d(2),
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units * 2, 3, 1, 1),
            nn.Conv2d(hidden_units * 2, hidden_units * 2, 3, 1, 1),
            nn.MaxPool2d(2),
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(hidden_units * 2, hidden_units, 3, 1, 1),
            nn.Conv2d(hidden_units, hidden_units, 3, 1, 1),
            nn.MaxPool2d(2),
        )

        self.classifer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16,output_shape)
        )

    def forward(self,x):
        return self.classifer(self.conv_block_3(self.conv_block_2(self.conv_block_1(x))))
