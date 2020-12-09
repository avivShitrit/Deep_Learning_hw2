r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""

1. the jacobian matrix w.r.t to X will have the same shape as the weights matrix.
X's shape is 128x1024 so in order to get the out matrix with shape of 128x2048, the shape of the weights matrix will have to be 1024x2048 and that's the jacobians shape.

2. first, we will calculate how many bits we need to store in order to store the whole jacobian matrix.
1024*2048*32 = 67,108,864 bits.
now we need to convert it to gigabytes:
67,108,864/8/1024/1024/1024 = 0.0078125 GB of RAM memory
"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 0.01
    lr = 0.05
    reg = 0.1
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 0.01
    lr_vanilla = 0.03
    lr_momentum = 0.04
    lr_rmsprop = 0.000026
    reg = 0.01 
    
#         wstd = 0.1
#     lr_vanilla = 0.03
#     lr_momentum = 0.005
#     lr_rmsprop = 0.003
#     reg = 0.001 
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr = 0.01
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**

The graph results we received match our expectations. As you can see when introducing the dropouts the test training performance improves. 
Without any dropouts the model suffers from over-fitting to the training set (with almost 100 accuracy) and the test set performance are poor.
With 0.4 dropout the generalisation error decreases, and as a result the test set performance increased.
With 0.8 dropout the learning rate of the model is very slow, and the number of epoch preformed wasn't sufficient to the learning rate. 
But, we can see that the generalisation error decreased even more so that with the 0.4 dropout and so the training set and test set are preforming similarly.

"""

part2_q2 = r"""
**Your answer:**

Yes, it's possible but it depends on the training set size.
The Cross-Entropy (CE) loss function is a continues function that calculates the distance between the prediction to the ground truth. 
On the other hand the accuracy is a binary result calculated based on the correctness of the classification.
Therefor, in some cases when the accuracy is low and we have samples that have predictions, on the higher end (or lower end) of the 50% cut off point, we'll get that the loss increases which than effects the classification rate. 
On the other hand the accuracy also increases due to the fact that we classified (by chance) correctly.
When the training set is big enough this occurrences are more rare.

"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q5 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q6 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============
