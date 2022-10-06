import torch
from torch import Tensor, nn


class THGStrainStressCNN(nn.Module):
    """Convolutional Neural Network (CNN) to calculate strain-stress features in THG skin images.

    Assumes a 2D RGB 258*258*3 input image.
    """

    def __init__(self, dropout: float, num_output_features: int) -> None:
        super(THGStrainStressCNN, self).__init__()

        self.network = nn.Sequential(
            # Layer one.
            nn.Conv2d(1, 64, 3),  # TODO: How many input channels? Only SHG?
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.MaxPool2d(2),
            nn.MaxPool2d(2),
            nn.MaxPool2d(2),
            # Layer two.
            nn.Conv2d(64, 64, 5),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.MaxPool2d(2),
            # Layer three.
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.MaxPool2d(2),
            # Layer four.
            nn.Conv2d(64, 64, 6),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            # MLP
            nn.Flatten(),
            # nn.Linear(1 * 1 * 64, 1 * 1 * 64),
            # nn.ReLU(),
            nn.Linear(1 * 1 * 64, 1 * 1 * 256),
            nn.ReLU(),
            nn.Linear(
                1 * 1 * 256, num_output_features
            ),  # TODO: How many output features are needed?
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# class THGStrainStressCNN(nn.Module):
#     """Convolutional Neural Network (CNN) to calculate strain-stress features in THG skin images.

#     Assumes a 2D RGB 258x258 grayscale input image.
#     """

#     def __init__(self, dropout: float, num_output_features: int):
#         super(THGStrainStressCNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 64, 3)
#         self.conv2 = nn.Conv2d(64, 64, 5)
#         self.conv3 = nn.Conv2d(64, 64, 3)
#         self.conv4 = nn.Conv2d(64, 64, 6)

#         self.fc1 = nn.Linear(1 * 1 * 64, 1 * 1 * 64)
#         self.fc2 = nn.Linear(1 * 1 * 64, 1 * 1 * 256)
#         self.fc3 = nn.Linear(
#             1 * 1 * 256, num_output_features
#         )  # TODO: How many output features are needed?

#         self.dropout = nn.Dropout2d(dropout)
#         self.max_pool = nn.MaxPool2d(2)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         # Layer one.
#         x = self.relu(self.conv1(x))
#         x = self.dropout(x)
#         for _ in range(3):
#             # x = self.relu(self.max_pool(x))
#             x = self.max_pool(x)
#         # Layer two.
#         x = self.relu(self.conv2(x))
#         print(2, x.size())
#         x = self.dropout(x)
#         # x = self.relu(self.max_pool(x))
#         x = self.max_pool(x)

#         # Layer three.
#         x = self.relu(self.conv3(x))
#         print(3, x.size())
#         x = self.dropout(x)
#         # x = self.relu(self.max_pool(x))
#         x = self.max_pool(x)

#         # Layer four.
#         x = self.relu(self.conv4(x))
#         print(4, x.size())
#         x = self.dropout(x)

#         # In order to feed it to the fully connected layer.
#         x = x.view(-1, self.num_flat_features(x))
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.fc3(x)

#         return x

#     def num_flat_features(self, x):
#         """Calculate the number of features in arg x."""
#         size = x.size()[1:]  # all dimensions except the batch dimension
#         num_features = 1
#         for s in size:
#             num_features *= s

#         return num_features
