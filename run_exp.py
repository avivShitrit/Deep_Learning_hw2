import hw2.experiments as experiments
from hw2.experiments import load_experiment
from cs236781.plot import plot_fit

seed = 42
<<<<<<< HEAD
batches = 500
=======
batches = 300
>>>>>>> 5f5b73a43417d759204c15a4cf3ce9e522426be6
epochs = 20
bs_train = 50


def run_exp_1_1():
    for K in [[32], [64]]:
        for L in [2, 4, 8, 16]:
            experiments.run_experiment(
                'exp1_1', seed=seed, bs_train=bs_train, batches=batches,
                epochs=epochs, early_stopping=5,
                filters_per_layer=K, layers_per_block=L,
                pool_every=((L / 2) + 1), hidden_dims=[100],
                model_type='cnn',
            )


def run_exp_1_2():
    for L in [2, 4, 8]:
        for K in [[32], [64], [128], [256]]:
            experiments.run_experiment(
                'exp1_2', seed=seed, bs_train=bs_train, batches=batches,
                epochs=epochs, early_stopping=5,
                filters_per_layer=K, layers_per_block=L,
                pool_every=((L / 2) + 1), hidden_dims=[100],
                model_type='cnn',
            )


def run_exp_1_3():
    K = [64, 128, 256]
    for L in [1, 2, 3, 4]:
        experiments.run_experiment(
            'exp1_3', seed=seed, bs_train=bs_train, batches=batches,
            epochs=epochs, early_stopping=5,
            filters_per_layer=K, layers_per_block=L, pool_every=((L / 2) + 1),
            hidden_dims=[100],
            model_type='cnn',
        )


def run_exp_1_4():
    K = [32]
    for L in [8, 16, 32]:
        experiments.run_experiment(
            'exp1_4', seed=seed, bs_train=bs_train, batches=batches,
            epochs=epochs, early_stopping=5,
            filters_per_layer=K, layers_per_block=L, pool_every=((L / 2) + 1),
            hidden_dims=[100],
            model_type='resnet',
        )
    K = [64, 128, 256]
    for L in [2, 4, 8]:
        experiments.run_experiment(
            'exp1_4', seed=seed, bs_train=bs_train,bs_test=bs_train, batches=batches,
            epochs=epochs, early_stopping=5,
            filters_per_layer=K, layers_per_block=L, pool_every=((L / 2) + 1),
            hidden_dims=[100],
            model_type='resnet',
        )

def run_exp_2():
    K = [32, 64, 128]
    for L in [3,6,9,12]:
        experiments.run_experiment(
            'exp2', seed=seed, bs_train=50, batches=500,
            epochs=35, early_stopping=5,
            filters_per_layer=K, layers_per_block=L, pool_every=((L / 2) + 1),
            hidden_dims=[256,100,100],
            model_type='ycn'
        )

def run_all_exp():
    # run_exp_1_1()
    # run_exp_1_2()
    # run_exp_1_3()
    # run_exp_1_4()
    run_exp_2()

if __name__ == "__main__":
    EXPERIMENTS = {
        "1.1": run_exp_1_1,
        "1.2": run_exp_1_2,
        "1.3": run_exp_1_3,
        "1.4": run_exp_1_4
    }
    run_all_exp()
