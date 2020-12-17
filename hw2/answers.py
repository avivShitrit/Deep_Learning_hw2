r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""

1. the jacobian matrix w.r.t to X has a shape of (inputs size)*(outputs size).
X's shape is 128x1024 and the the ouput shape is 128x2048, the shape of the jacobians matrix will be (128*1024)x(128*2048).

2. first, we will calculate how many bits we need to store in order to store the whole jacobian matrix.
128*1024*128*2048*32 = 1,099,511,627,776
now we need to convert it to gigabytes:
1,099,511,627,776/8/1024/1024/1024 = 128 GB of RAM memory
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
1. we'll caculate the numper of parameters for each convolution layer.
bottleneck Residual block number of parameters:
conv1 = ((1*1*64)+1)*64 = 4160
conv2 = ((3*3*64)+1)*64 = 36928
conv3 = ((1*1*64)+1)*256 = 16640
total = 57728

regular residual block number of parameters:
conv1 = ((3*3*256)+1)*256 = 590080
conv2 = ((3*3*256)+1)*256 = 590080
total = 1180160

2. x*y*64 input
conv1 = 64*input

"""

part3_q2 = r"""
**Your answer:**
1. As we can see from the expirements results, the best accuracies is achived by the the most shallow networks with L=2 or L=4.
in the first expirement where K=32 the best accuracy is achived by the most shallow Net's got similar results with overfitting to the
test set (80% accuracy instead of 60% over the test set) also as the Net got deeper the accuracy decreased till a point where L=16 
that the accuracy didn't improve over time therefore the graph is flat.

2. As we said for L=16 the wasn't trainable The reason for this behavior can be that the information isn't reaching
the deepest layers duo to exsecive computations along the way so when the layer reaches the classifeir the 
information is almost random so we have no improvement in accuracy.
two things we can do to resolve this situation:
	1. we can add batchNorm layers to our network to noramlize and reduces the internal covariante shift so the inforamtion 
	at the end of the net is nore effected by the input data.
	2. we can add more pooling layers along the Net to reduce the number of parameters to learn and the amount of computatio
	n performed in the network which will cause some training on the deeper layers.

"""

part3_q3 = r"""
**Your answer:**

In this test we can see a conection between the K/L ratio and the accuracies, when L=2,4 we can see that the bset accuracies
are achived with K=32 and 64 respectivly so the K/L ratio in both was 16, also when L=8 we can see that K=128 and 256
got the best results which mean that a K/L ration between 16 to 32 will probably achived the best accuracy.
As for big K/L ration the results are not conclusive cause for L=2 and K=256 we can see that the model got better results the
L=2, K=128, but for L=4 we can see that K=256 the accuracy is the lowest.
As for small K/L ration we can see that when K=4,8 the models with the smallest ratio got the worst accuracies.
When we compare expirement 1.2 to expirement 1.1 we can come to the same conclution that a smallest K/L ratios preduced the
worst accuracies.

"""

part3_q4 = r"""
**Your answer:**

We can see that deeper networks are non trainable for L=3,4 the models accuracies didn't change over time.
For L=1,2 we got decent accuracies and we can see that the most shallow model got the best results, although we can see 
that where L=1 the globalization error is the highest so the is overfitted to the train set with almost 90% accuracy.

"""

part3_q5 = r"""
**Your answer:**

1.In exp 1.4 we can see that even models with more then 16 layers are trainable but the model results still best 
when L is the smallest.
We can see that most of the models in this experiment is overfitting to the train set.
When comparing to exp 1.1 and 1.3 we can coclude that the use of the resNet allowed us to created deeper models,
but still deeper models got worst results the the shllow ones.
When can also see that the resNet got lower accuracies and bigger generalization error then the CNN.

"""

part3_q6 = r"""
**Your answer:**

1.we'll explain each modifaction we made and the reason we made it:
	1. we added hidden layers to the classifier - to have deeper and more spesific clasification to the features that
	   extracted from the cov layers, in that way our classifier have more parameters to train.
	2. we used the CNN model - as we can see from the other experiments results the CNN model was more accurate
	   so we decided to use it.
	3. we added batchnorm and dropuot to the model - because exp 2 uses deep networks we added batchnorm and dropout
	   layers between conv layers to better cope with deep models and made then trainable.

2.
Non trainable models:
In exp 2 duo to our modifications all of the models are trainable even model with 36 layers got some improvment
in accuracy over time when in exp 1 when we used the CNN, models with more then 8 layers where non trainable.

Generalization Error:
In exp 2 we have small generalization error, the test set accuracy is matchnig the train set accuracy in all the models,
so the models are not overfitted to the spesific train set when in exp 1 all the models train set accuray was segnificantly
higher then the test set accuracy.

L w.r.t to accuracy:
In both experiments we can see that when L is the smallest the accuracies are higher, so we can conclude that 
having alot of adjacent layers with the same shape leading to worst results.

Overall comparison:
In exp 2 we got at best 60% accuracy for the test set when in exp 1 some of the models managed to have even 65% accuracy,
but as we mantioned before the generalization error in exp 2 was secnificantly better then exp 1 models so overall 
we think that the model from exp 2 will be better classifier the the models from exp 1 for unknown test sets.

"""
# ==============
