import torch
import torch.nn as nn
import itertools as it

# from .blocks import ReLU, LeakyReLU, Linear

ACTIVATIONS = {"relu": nn.ReLU, "lrelu": nn.LeakyReLU}
POOLINGS = {"avg": nn.AvgPool2d, "max": nn.MaxPool2d}


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(CONV -> ACT)*P -> POOL]*(N/P) -> (FC -> ACT)*M -> FC
    """

    def __init__(
            self,
            in_size,
            out_classes: int,
            channels: list,
            pool_every: int,
            hidden_dims: list,
            conv_params: dict = {},
            activation_type: str = "relu",
            activation_params: dict = {},
            pooling_type: str = "max",
            pooling_params: dict = {},
    ):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param channels: A list of of length N containing the number of
            (output) channels in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        :param conv_params: Parameters for convolution layers.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        :param pooling_type: Type of pooling to apply; supports 'max' for max-pooling or
            'avg' for average pooling.
        :param pooling_params: Parameters passed to pooling layer.
        """
        super().__init__()
        assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims
        self.conv_params = conv_params
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.pooling_type = pooling_type
        self.pooling_params = pooling_params

        if activation_type not in ACTIVATIONS or pooling_type not in POOLINGS:
            raise ValueError("Unsupported activation or pooling type")

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # DONE: Create the feature extractor part of the model:
        #  [(CONV -> ACT)*P -> POOL]*(N/P)
        #  Apply activation function after each conv, using the activation type and
        #  parameters.
        #  Apply pooling to reduce dimensions after every P convolutions, using the
        #  pooling type and pooling parameters.
        #  Note: If N is not divisible by P, then N mod P additional
        #  CONV->ACTs should exist at the end, without a POOL after them.
        # ====== YOUR CODE: ======
        padding = self.conv_params['padding']
        stride = self.conv_params['stride']
        conv_kernel = self.conv_params['kernel_size']
        pool_kernel = self.pooling_params['kernel_size']

        N = len(self.channels)
        P = self.pool_every

        for i in range(N):
            # conv layer
            layers.append(nn.Conv2d(in_channels=in_channels,
                                    out_channels=self.channels[i],
                                    **self.conv_params))

            # relu / leaky relu layer
            if self.activation_type == "lrelu":
                layers.append(nn.LeakyReLU(**self.activation_params))
            else:
                layers.append(nn.ReLU(**self.activation_params))

            # update dimensions
            temp_dim = (2 * padding - conv_kernel)
            in_h = int(
                torch.floor(torch.tensor((in_h + temp_dim) / stride)) + 1)
            in_w = int(
                torch.floor(torch.tensor((in_w + temp_dim) / stride)) + 1)
            in_channels = self.channels[i]

            # pooling layer
            if (i + 1) % P == 0:
                if self.pooling_type == "avg":
                    layers.append(nn.AvgPool2d(**self.pooling_params))
                else:
                    layers.append(nn.MaxPool2d(**self.pooling_params))

                # update dimensions
                in_h = int(torch.floor(
                    torch.tensor((in_h - pool_kernel) / pool_kernel)) + 1)
                in_w = int(torch.floor(
                    torch.tensor((in_w - pool_kernel) / pool_kernel)) + 1)
        # ========================
        self.in_size = self.channels[-1], in_h, in_w
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        layers = []
        # DONE: Create the classifier part of the model:
        #  (FC -> ACT)*M -> Linear
        #  You'll first need to calculate the number of features going in to
        #  the first linear layer.
        #  The last Linear layer should have an output dim of out_classes.
        # ====== YOUR CODE: ======
        first_in_dims = int(
            self.channels[-1] * self.in_size[1] * self.in_size[2])

        for indims, outdims in zip([first_in_dims] + self.hidden_dims,
                                   self.hidden_dims):
            layers.append(nn.Linear(indims, outdims))

            if self.activation_type == "lrelu":
                layers.append(nn.LeakyReLU(**self.activation_params))
            else:
                layers.append(nn.ReLU(**self.activation_params))

        layers.append(nn.Linear(self.hidden_dims[-1], self.out_classes))
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # DONE: Implement the forward pass.
        #  Extract features from the input, run the classifier on them and
        #  return class scores.
        # ====== YOUR CODE: ======
        features = self.feature_extractor(x)
        flatten_features = features.view(features.size(0), -1)
        out = self.classifier(flatten_features)
        # ========================
        return out


class ResidualBlock(nn.Module):
    """
    A general purpose residual block.
    """

    def __init__(
            self,
            in_channels: int,
            channels: list,
            kernel_sizes: list,
            batchnorm=False,
            dropout=0.0,
            activation_type: str = "relu",
            activation_params: dict = {},
            **kwargs,
    ):
        """
        :param in_channels: Number of input channels to the first convolution.
        :param channels: List of number of output channels for each
            convolution in the block. The length determines the number of
            convolutions.
        :param kernel_sizes: List of kernel sizes (spatial). Length should
            be the same as channels. Values should be odd numbers.
        :param batchnorm: True/False whether to apply BatchNorm between
            convolutions.
        :param dropout: Amount (p) of Dropout to apply between convolutions.
            Zero means don't apply dropout.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        """
        super().__init__()
        assert channels and kernel_sizes
        assert len(channels) == len(kernel_sizes)
        assert all(map(lambda x: x % 2 == 1, kernel_sizes))

        if activation_type not in ACTIVATIONS:
            raise ValueError("Unsupported activation type")

        self.main_path, self.shortcut_path = None, None

        # DONE: Implement a generic residual block.
        #  Use the given arguments to create two nn.Sequentials:
        #  - main_path, which should contain the convolution, dropout,
        #    batchnorm, relu sequences (in this order).
        #    Should end with a final conv as in the diagram.
        #  - shortcut_path which should represent the skip-connection and
        #    may contain a 1x1 conv.
        #  Notes:
        #  - Use convolutions which preserve the spatial extent of the input.
        #  - Use bias in the main_path conv layers, and no bias in the skips.
        #  - For simplicity of implementation, assume kernel sizes are odd.
        #  - Don't create layers which you don't use! This will prevent
        #    correct comparison in the test.
        # ====== YOUR CODE: ======

        # We're assuming there are no dilation, stride or padding.
        # The default values we'll be used to calculate the conv.

        # main path
        # conv -> bias=True
        # dropout
        # batchnorm
        # relu
        # conv

        N = len(channels)
        main_layers = []
        last_in_channels = in_channels

        for i in range(N - 1):
            # conv layer
            main_layers.append(nn.Conv2d(in_channels=last_in_channels,
                                         out_channels=channels[i],
                                         kernel_size=kernel_sizes[i],
                                         padding=int((kernel_sizes[i] - 1) / 2),
                                         bias=True))
            # dropout
            main_layers.append(nn.Dropout2d(dropout))
            # batchnorm
            if batchnorm:
                main_layers.append(nn.BatchNorm2d(num_features=channels[i]))
            # relu / leaky relu layer
            if activation_type == "lrelu":
                main_layers.append(nn.LeakyReLU(**activation_params))
            else:
                main_layers.append(nn.ReLU(**activation_params))

            # update dimensions
            last_in_channels = channels[i]

        main_layers.append(nn.Conv2d(in_channels=last_in_channels,
                                     out_channels=channels[-1],
                                     kernel_size=kernel_sizes[-1],
                                     padding=int((kernel_sizes[-1] - 1) / 2),
                                     bias=True))
        self.main_path = nn.Sequential(*main_layers)

        # shortcut path
        # skip connection
        # 1X1 conv -> bias=False
        if in_channels == channels[-1]:
            self.shortcut_path = nn.Sequential(nn.Identity())
        else:
            self.shortcut_path = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                         out_channels=channels[-1],
                                                         kernel_size=1,
                                                         bias=False))

        # ========================

    def forward(self, x):
        out = self.main_path(x)
        out += self.shortcut_path(x)
        out = torch.relu(out)
        return out


class ResNetClassifier(ConvClassifier):
    def __init__(
            self,
            in_size,
            out_classes,
            channels,
            pool_every,
            hidden_dims,
            batchnorm=False,
            dropout=0.0,
            **kwargs,
    ):
        """
        See arguments of ConvClassifier & ResidualBlock.
        """
        self.batchnorm = batchnorm
        self.dropout = dropout
        super().__init__(
            in_size, out_classes, channels, pool_every, hidden_dims, **kwargs
        )

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # DONE: Create the feature extractor part of the model:
        #  [-> (CONV -> ACT)*P -> POOL]*(N/P)
        #   \------- SKIP ------/
        #  For the ResidualBlocks, use only dimension-preserving 3x3 convolutions.
        #  Apply Pooling to reduce dimensions after every P convolutions.
        #  Notes:
        #  - If N is not divisible by P, then N mod P additional
        #    CONV->ACT (with a skip over them) should exist at the end,
        #    without a POOL after them.
        #  - Use your own ResidualBlock implementation.
        # ====== YOUR CODE: ======
        N = len(self.channels)
        P = self.pool_every
        pool_kernel = self.pooling_params['kernel_size']

        for i in range(N // P):

            layers.append(ResidualBlock(in_channels=in_channels,
                                        channels=self.channels[i * P:P * (i + 1)],
                                        kernel_sizes=[3] * P,
                                        batchnorm=self.batchnorm,
                                        dropout=self.dropout,
                                        activation_type=self.activation_type,
                                        activation_params=self.activation_params))

            # update dimensions
            in_channels = self.channels[i]

            # pooling layer
            if self.pooling_type == "avg":
                layers.append(nn.AvgPool2d(**self.pooling_params))
            else:
                layers.append(nn.MaxPool2d(**self.pooling_params))

            in_h = int(torch.floor(
                torch.tensor((in_h - pool_kernel) / pool_kernel)) + 1)
            in_w = int(torch.floor(
                torch.tensor((in_w - pool_kernel) / pool_kernel)) + 1)

        if N % P != 0:
            size = N // P
            layers.append(ResidualBlock(in_channels=self.channels[size * P - 1],
                                        channels=self.channels[size * P:],
                                        kernel_sizes=[3] * (N % P),
                                        batchnorm=self.batchnorm,
                                        dropout=self.dropout,
                                        activation_type=self.activation_type,
                                        activation_params=self.activation_params))

        # ========================
        self.in_size = self.channels[-1], in_h, in_w
        seq = nn.Sequential(*layers)
        return seq


class YourCodeNet(ConvClassifier):
    def __init__(self, in_size, out_classes, channels, pool_every, hidden_dims):
        super().__init__(in_size, out_classes, channels, pool_every,
                         hidden_dims)

    # TODO: Change whatever you want about the ConvClassifier to try to
    #  improve it's results on CIFAR-10.
    #  For example, add batchnorm, dropout, skip connections, change conv
    #  filter sizes etc.
    # ====== YOUR CODE: ======

    # ========================
