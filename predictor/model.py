import torch.nn as nn
import torch


class DeepAntModel(nn.Module):
    def __init__(self, len_window, nb_dimensions, add_dropout=False):
        super(DeepAntModel, self).__init__()
        self.fitted = False
        self.add_dropout = add_dropout
        self.input_dim = len_window
        self.output_dim = nb_dimensions

        self.first_conv = nn.Sequential(
            nn.Conv1d(in_channels=len_window, out_channels=32, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2))

        self.second_conv = nn.Sequential(nn.Conv1d(in_channels=32, out_channels=32, kernel_size=2, stride=1, padding=1),
                                         nn.ReLU(),
                                         nn.MaxPool1d(kernel_size=2))

        ### COMPUTE INPUT DIMENSIONS ###
        x = torch.ones((3, self.input_dim, self.output_dim))
        with torch.no_grad():
            x = self.second_conv(self.first_conv(x))
        input_dims = x.shape[1] * x.shape[2]
        ### COMPUTE INPUT DIMENSIONS ###

        self.mlp = nn.Sequential(nn.Flatten(),
                                 nn.Linear(input_dims, input_dims),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.3) if add_dropout else nn.Identity(),
                                 nn.Linear(input_dims, nb_dimensions))

    def forward(self, x):
        x = self.first_conv(x)
        x = self.second_conv(x)
        return self.mlp(x)