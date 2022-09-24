from torch import nn


class VisCNN(nn.Module):
    """Convolutional Neural Network (CNN) to calculate stress-strain features in THG skin images.

    Assumes a 2D 258*258*1 input image.
    """

    def __init__(self):
        super(VisCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3)  # TODO: How many input channels? Only THG?
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 6)

        self.fc1 = nn.Linear(1 * 1 * 64, 1 * 1 * 64)
        self.fc2 = nn.Linear(1 * 1 * 64, 1 * 1 * 256)
        self.fc3 = nn.Linear(
            1 * 1 * 256, 3
        )  # TODO: How many output features are needed?

        self.dropout = nn.Dropout2d(0.3)
        self.max_pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        for _ in range(3):
            x = self.relu(self.max_pool(x))

        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.relu(self.max_pool(x))

        x = self.relu(self.conv3(x))
        x = self.dropout(x)
        x = self.relu(self.max_pool(x))

        x = self.relu(self.conv4(x))
        x = self.dropout(x)

        # In order to feed it to the fully connected layer.
        x = x.view(-1, self.num_flat_features(x))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def num_flat_features(self, x):
        """Calculate the number of features in arg x."""
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features
