import abc
import torch


class Block(abc.ABC):
    """
    A block is some computation element in a network architecture which
    supports automatic differentiation using forward and backward functions.
    """

    def __init__(self):
        # Store intermediate values needed to compute gradients in this hash
        self.grad_cache = {}
        self.training_mode = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """
        Computes the forward pass of the block.
        :param args: The computation arguments (implementation specific).
        :return: The result of the computation.
        """
        pass

    @abc.abstractmethod
    def backward(self, dout):
        """
        Computes the backward pass of the block, i.e. the gradient
        calculation of the final network output with respect to each of the
        parameters of the forward function.
        :param dout: The gradient of the network with respect to the
        output of this block.
        :return: A tuple with the same number of elements as the parameters of
        the forward function. Each element will be the gradient of the
        network output with respect to that parameter.
        """
        pass

    @abc.abstractmethod
    def params(self):
        """
        :return: Block's trainable parameters and their gradients as a list
        of tuples, each tuple containing a tensor and it's corresponding
        gradient tensor.
        """
        pass

    def train(self, training_mode=True):
        """
        Changes the mode of this block between training and evaluation (test)
        mode. Some blocks have different behaviour depending on mode.
        :param training_mode: True: set the model in training mode. False: set
        evaluation mode.
        """
        self.training_mode = training_mode

    def __repr__(self):
        return self.__class__.__name__


class LeakyReLU(Block):
    """
    Leaky version of Rectified linear unit.
    """

    def __init__(self, alpha: float = 0.01):
        super().__init__()
        if not (0 <= alpha < 1):
            raise ValueError("Invalid value of alpha")
        self.alpha = alpha

    def forward(self, x, **kw):
        """
        Computes max(alpha*x, x) for some 0<= alpha < 1.
        :param x: Input tensor of shape (N,*) where N is the batch
        dimension, and * is any number of other dimensions.
        :return: ReLU of each sample in x.
        """

        # DONE: Implement the LeakyReLU operation.
        # ====== YOUR CODE: ======
        out = torch.max(x, self.alpha * x)
        # ========================

        self.grad_cache["x"] = x
        return out

    def backward(self, dout):
        """
        :param dout: Gradient with respect to block output, shape (N, *).
        :return: Gradient with respect to block input, shape (N, *)
        """
        x = self.grad_cache["x"]

        # DONE: Implement gradient w.r.t. the input x
        # ====== YOUR CODE: ======
        dx = dout * torch.where(x > 0, torch.ones_like(x), self.alpha * torch.ones_like(x))
        # ========================

        return dx

    def params(self):
        return []

    def __repr__(self):
        return f"LeakyReLU({self.alpha=})"


class ReLU(LeakyReLU):
    """
    Rectified linear unit.
    """

    def __init__(self):
        # ====== YOUR CODE: ======
        super().__init__(alpha=0)
        # ========================

    def __repr__(self):
        return "ReLU"


class Sigmoid(Block):
    """
    Sigmoid activation function.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, **kw):
        """
        Computes s(x) = 1/(1+exp(-x))
        :param x: Input tensor of shape (N,*) where N is the batch
        dimension, and * is any number of other dimensions.
        :return: Sigmoid of each sample in x.
        """

        # DONE: Implement the Sigmoid function.
        #  Save whatever you need into grad_cache.
        # ====== YOUR CODE: ======
        out = 1/(1+torch.exp(-x))
        self.grad_cache["x"] = x
        # ========================

        return out

    def backward(self, dout):
        """
        :param dout: Gradient with respect to block output, shape (N, *).
        :return: Gradient with respect to block input, shape (N, *)
        """

        # DONE: Implement gradient w.r.t. the input x
        # ====== YOUR CODE: ======
        x = self.grad_cache["x"]
        dx = dout * (1/(1 + torch.exp(-x)) * (1 - 1/(1 + torch.exp(-x))))
        # ========================

        return dx

    def params(self):
        return []


class TanH(Block):
    """
    Hyperbolic tangent activation function.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, **kw):
        """
        Computes tanh(x) = (exp(x)-exp(-x))/(exp(x)+exp(-x))
        :param x: Input tensor of shape (N,*) where N is the batch
        dimension, and * is any number of other dimensions.
        :return: Sigmoid of each sample in x.
        """

        # DONE: Implement the tanh function.
        #  Save whatever you need into grad_cache.
        # ====== YOUR CODE: ======
        out = (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))
        self.grad_cache["x"] = x
        # ========================

        return out

    def backward(self, dout):
        """
        :param dout: Gradient with respect to block output, shape (N, *).
        :return: Gradient with respect to block input, shape (N, *)
        """

        # DONE: Implement gradient w.r.t. the input x
        # ====== YOUR CODE: ======
        x = self.grad_cache["x"]
        dx = dout * (4 / ((torch.exp(x) + torch.exp(-x))**2))
        # ========================

        return dx

    def params(self):
        return []


class Linear(Block):
    """
    Fully-connected linear layer.
    """

    def __init__(self, in_features, out_features, wstd=0.1):
        """
        :param in_features: Number of input features (Din)
        :param out_features: Number of output features (Dout)
        :param wstd: standard deviation of the initial weights matrix
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # DONE: Create the weight matrix (self.w) and bias vector (self.b).
        # ====== YOUR CODE: ======
        self.w = torch.normal(torch.zeros(out_features, in_features), wstd)
        self.b = torch.normal(torch.zeros(out_features), wstd)
        # ========================

        # These will store the gradients
        self.dw = torch.zeros_like(self.w)
        self.db = torch.zeros_like(self.b)

    def params(self):
        return [(self.w, self.dw), (self.b, self.db)]

    def forward(self, x, **kw):
        """
        Computes an affine transform, y = x W^T + b.
        :param x: Input tensor of shape (N,Din) where N is the batch
        dimension, and Din is the number of input features.
        :return: Affine transform of each sample in x.
        """

        # DONE: Compute the affine transform
        # ====== YOUR CODE: ======
        N = (x.shape[0])
        x = x.view(N, -1)
        out = torch.matmul(x, torch.transpose(self.w, 0, 1)) + self.b
        # ========================

        self.grad_cache["x"] = x
        return out

    def backward(self, dout):
        """
        :param dout: Gradient with respect to block output, shape (N, Dout).
        :return: Gradient with respect to block input, shape (N, Din)
        """
        x = self.grad_cache["x"]

        # DONE: Compute
        #   - dx, the gradient of the loss with respect to x
        #   - dw, the gradient of the loss with respect to w
        #   - db, the gradient of the loss with respect to b
        #  Note: You should ACCUMULATE gradients in dw and db.
        # ====== YOUR CODE: ======
        N = dout.shape[0]
        dx = torch.matmul(dout, self.w)
        self.dw += torch.matmul(dout.T, x)
        self.db += (torch.matmul(torch.ones(N), dout)).T
        # ========================
        return dx

    def __repr__(self):
        return f"Linear({self.in_features=}, {self.out_features=})"


class CrossEntropyLoss(Block):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        """
        Computes cross-entropy loss directly from class scores.
        Given class scores x, and a 1-hot encoding of the correct class yh,
        the cross entropy loss is defined as: -yh^T * log(softmax(x)).

        This implementation works directly with class scores (x) and labels
        (y), not softmax outputs or 1-hot encodings.

        :param x: Tensor of shape (N,D) where N is the batch
            dimension, and D is the number of features. Should contain class
            scores, NOT PROBABILITIES.
        :param y: Tensor of shape (N,) containing the ground truth label of
            each sample.
        :return: Cross entropy loss, as if we computed the softmax of the
            scores, encoded y as 1-hot and calculated cross-entropy by
            definition above. A scalar.
        """

        N = x.shape[0]

        # Shift input for numerical stability
        xmax, _ = torch.max(x, dim=1, keepdim=True)
        x = x - xmax

        # DONE: Compute the cross entropy loss using the last formula from the
        #  notebook (i.e. directly using the class scores).
        # ====== YOUR CODE: ======
        D = x.shape[1]
        x_y = x[range(N), y]

        # summing across rows
        sum = torch.sum(torch.exp(x), dim=1)
        loss = torch.mean(torch.log(sum) - x_y)
        # ========================

        self.grad_cache["x"] = x
        self.grad_cache["y"] = y
        return loss

    def backward(self, dout=1.0):
        """
        :param dout: Gradient with respect to block output, a scalar which
            defaults to 1 since the output of forward is scalar.
        :return: Gradient with respect to block input (only x), shape (N,D)
        """
        x = self.grad_cache["x"]
        y = self.grad_cache["y"]
        N = x.shape[0]

        # DONE: Calculate the gradient w.r.t. the input x.
        # ====== YOUR CODE: ======
        # summing across rows
        sum = torch.sum(torch.exp(x), dim=1)
        
        # softmax = e^x/sum(e^x)
        softmax = (torch.exp(x).T / sum).T
        
        softmax[range(N), y] -= 1
        dx = dout * softmax / N
        # ========================

        return dx

    def params(self):
        return []


class Dropout(Block):
    def __init__(self, p=0.5):
        """
        Initializes a Dropout block.
        :param p: Probability to drop an activation.
        """
        super().__init__()
        assert 0.0 <= p <= 1.0
        self.p = p

    def forward(self, x, **kw):
        # DONE: Implement the dropout forward pass.
        #  Notice that contrary to previous blocks, this block behaves
        #  differently a according to the current training_mode (train/test).
        # ====== YOUR CODE: ======
        if self.training_mode:
            drops = torch.bernoulli((1 - self.p) * torch.ones_like(x))
            out = x * drops / (1 - self.p)
            self.grad_cache['drops'] = drops
        else:
            out = x
        # ========================

        return out

    def backward(self, dout):
        # DONE: Implement the dropout backward pass.
        # ====== YOUR CODE: ======
        if self.training_mode:
            drops = self.grad_cache['drops']
            dx = dout * drops / (1 - self.p)
        else:
            dx = dout
        # ========================

        return dx

    def params(self):
        return []

    def __repr__(self):
        return f"Dropout(p={self.p})"


class Sequential(Block):
    """
    A Block that passes input through a sequence of other blocks.
    """

    def __init__(self, *blocks):
        super().__init__()
        self.blocks = blocks

    def forward(self, x, **kw):
        out = None

        # DONE: Implement the forward pass by passing each block's output
        #  as the input of the next.
        # ====== YOUR CODE: ======
        last_layer_res = x
        # we need a tmp variable to store results
        for block in self.blocks:
            last_layer_res = block.forward(last_layer_res, **kw)
        out = last_layer_res
        # ========================

        return out

    def backward(self, dout):
        din = None

        # DONE: Implement the backward pass.
        #  Each block's input gradient should be the previous block's output
        #  gradient. Behold the backpropagation algorithm in action!
        # ====== YOUR CODE: ======
        reversed_blocks_list = list(self.blocks)
        reversed_blocks_list.reverse()
        
        # Doing back iteration on the blocks
        last_layer_res = dout
        for block in reversed_blocks_list:
            last_layer_res = block.backward(last_layer_res)
            
        din = last_layer_res
        # ========================

        return din

    def params(self):
        params = []

        # DONE: Return the parameter tuples from all blocks.
        # ====== YOUR CODE: ======
        params = sum((b.params() for b in self.blocks), [])
        # ========================

        return params

    def train(self, training_mode=True):
        for block in self.blocks:
            block.train(training_mode)

    def __repr__(self):
        res = "Sequential\n"
        for i, block in enumerate(self.blocks):
            res += f"\t[{i}] {block}\n"
        return res

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, item):
        return self.blocks[item]


class MLP(Block):
    """
    A simple multilayer perceptron based on our custom Blocks.
    Architecture is (with ReLU activation):

        FC(in, h1) -> ReLU -> FC(h1,h2) -> ReLU -> ... -> FC(hn, num_classes)

    Where FC is a fully-connected layer and h1,...,hn are the hidden layer
    dimensions.
    If dropout is used, a dropout layer is added after every activation
    function.
    """

    def __init__(
        self,
        in_features,
        num_classes,
        hidden_features=(),
        activation="relu",
        dropout=0,
        **kw,
    ):
        super().__init__()
        """
        Create an MLP model Block.
        :param in_features: Number of features of the input of the first layer.
        :param num_classes: Number of features of the output of the last layer.
        :param hidden_features: A sequence of hidden layer dimensions.
        :param activation: Either 'relu' or 'sigmoid', specifying which 
        activation function to use between linear layers.
        :param: Dropout probability. Zero means no dropout.
        """
        blocks = []

        # DONE: Build the MLP architecture as described.
        # ====== YOUR CODE: ======
        linear_kw = {}
#         {'wstd':kw['wstd']} if 'wstd' in kw else {}
        for Din, Dout in zip([in_features] + hidden_features, hidden_features):
            blocks.append(Linear(in_features=Din, out_features=Dout, **linear_kw))
            if activation == "sigmoid":
                blocks.append(Sigmoid())
            else:
                blocks.append(ReLU())
            if dropout > 0:
                blocks.append(Dropout(dropout))
        blocks.append(Linear(in_features=hidden_features[-1], out_features=num_classes, **linear_kw))
        # ========================

        self.sequence = Sequential(*blocks)

    def forward(self, x, **kw):
        return self.sequence(x, **kw)

    def backward(self, dout):
        return self.sequence.backward(dout)

    def params(self):
        return self.sequence.params()

    def train(self, training_mode=True):
        self.sequence.train(training_mode)

    def __repr__(self):
        return f"MLP, {self.sequence}"
