import hw2.experiments as experiments
from hw2.experiments import load_experiment
from cs236781.plot import plot_fit

seed = 42
def run_exp_1_1():
	for K in [32,64]:
		for L in [2,4,8,16]:
			experiments.run_experiment(
			    'exp1_1', seed=seed, bs_train=200, batches=5, epochs=1, early_stopping=5,
			    filters_per_layer=[K], layers_per_block=L, pool_every=((L/2)+1), hidden_dims=[100],
			    model_type='resnet',
				)


if __name__ == "__main__":
	print("main")
	run_exp_1_1()