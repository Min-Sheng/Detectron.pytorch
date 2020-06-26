import torch
import torch.nn as nn
import torch.nn.functional as F

# Few-shot Spatial Unification Network
class FSSUN(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(FSSUN, self).__init__()
        self._in_ch = in_channels 
        self._ksize = kernel_size

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
        )

        self.globalAvgPool = nn.AdaptiveAvgPool2d(4)

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(32 * 4 * 4, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def detectron_weight_mapping(self):
        mapping = {}
        orphan_in_detectron = []
        return mapping, orphan_in_detectron

    def forward(self, x):
        xs = self.localization(x)
        xs = self.globalAvgPool(xs)
        #print(xs.shape)
        xs = xs.view(-1, 32 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x



# Few-shot Spatial Alignment Network
class FSSAN(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(FSSAN, self).__init__()
        self._in_ch = in_channels
        self._ksize = kernel_size

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=self._ksize, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels//2, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False),
            nn.ReLU(True)
        )

        self.globalAvgPool = nn.AdaptiveAvgPool2d(4)

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(32 * 4 * 4, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def detectron_weight_mapping(self):
        mapping = {}
        orphan_in_detectron = []
        return mapping, orphan_in_detectron

    def forward(self, x, x_guide):
        xs = self.localization(torch.cat((x, x_guide), dim=1))
        xs = self.globalAvgPool(xs)
        #print(xs.shape)
        xs = xs.view(-1, 32 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x
