import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def model_cnn_2layer(in_ch, in_dim, width, linear_size = 128):
    model = nn.Sequential(
        nn.Conv2d(in_ch, 4*width, 4, stride=2, padding = 1),
        nn.ReLU(),
        nn.Conv2d(4*width, 8*width, 4, stride=2, padding = 1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*width*(in_dim // 4)*(in_dim // 4), linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, 10)
    )
    return model

def prelu_model_cnn_2layer(in_ch, in_dim, width, linear_size = 128):
    model = nn.Sequential(
        nn.Conv2d(in_ch, 4*width, 4, stride=2, padding = 1),
        nn.PReLU(),
        nn.Conv2d(4*width, 8*width, 4, stride=2, padding = 1),
        nn.PReLU(),
        Flatten(),
        nn.Linear(8*width*(in_dim // 4)*(in_dim // 4), linear_size),
        nn.PReLU(),
        nn.Linear(linear_size, 10)
    )
    return model

def leaky_model_cnn_2layer(in_ch, in_dim, width, linear_size = 128):
    model = nn.Sequential(
        nn.Conv2d(in_ch, 4*width, 4, stride=2, padding = 1),
        nn.LeakyReLU(),
        nn.Conv2d(4*width, 8*width, 4, stride=2, padding = 1),
        nn.LeakyReLU(),
        Flatten(),
        nn.Linear(8*width*(in_dim // 4)*(in_dim // 4), linear_size),
        nn.LeakyReLU(),
        nn.Linear(linear_size, 10)
    )
    return model


def model_cnn_3layer_fixed(in_ch, in_dim, kernel_size, width, linear_size = None):
    if linear_size is None:
        linear_size = width * 64
    if kernel_size == 5:
        h = (in_dim - 4) // 4
    elif kernel_size == 3:
        h = in_dim // 4
    else:
        raise ValueError("Unsupported kernel size")
    model = nn.Sequential(
        nn.Conv2d(in_ch, 4*width, kernel_size=kernel_size, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(4*width, 8*width, kernel_size=kernel_size, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8*width, 8*width, kernel_size=4, stride=4, padding=0),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*width*h*h, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, 10)
    )
    return model

def prelu_model_cnn_3layer_fixed(in_ch, in_dim, kernel_size, width, linear_size = None):
    if linear_size is None:
        linear_size = width * 64
    if kernel_size == 5:
        h = (in_dim - 4) // 4
    elif kernel_size == 3:
        h = in_dim // 4
    else:
        raise ValueError("Unsupported kernel size")
    model = nn.Sequential(
        nn.Conv2d(in_ch, 4*width, kernel_size=kernel_size, stride=1, padding=1),
        nn.PReLU(),
        nn.Conv2d(4*width, 8*width, kernel_size=kernel_size, stride=1, padding=1),
        nn.PReLU(),
        nn.Conv2d(8*width, 8*width, kernel_size=4, stride=4, padding=0),
        nn.PReLU(),
        Flatten(),
        nn.Linear(8*width*h*h, linear_size),
        nn.PReLU(),
        nn.Linear(linear_size, 10)
    )
    return model

def model_cnn_4layer(in_ch, in_dim, width, linear_size):
    model = nn.Sequential(
        nn.Conv2d(in_ch, 4*width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(4*width, 4*width, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(4*width, 8*width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8*width, 8*width, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*width*(in_dim//4)*(in_dim//4), linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, 10)
    )
    return model

def prelu_model_cnn_4layer(in_ch, in_dim, width, linear_size):
    model = nn.Sequential(
        nn.Conv2d(in_ch, 4*width, 3, stride=1, padding=1),
        nn.PReLU(),
        nn.Conv2d(4*width, 4*width, 4, stride=2, padding=1),
        nn.PReLU(),
        nn.Conv2d(4*width, 8*width, 3, stride=1, padding=1),
        nn.PReLU(),
        nn.Conv2d(8*width, 8*width, 4, stride=2, padding=1),
        nn.PReLU(),
        Flatten(),
        nn.Linear(8*width*(in_dim//4)*(in_dim//4), linear_size),
        nn.PReLU(),
        nn.Linear(linear_size, linear_size),
        nn.PReLU(),
        nn.Linear(linear_size, 10)
    )
    return model

