#!/bin/bash

echo "***  STARTING EXPERIMENTS ***"

echo "*** experiment 1.1 ***"

for K in 32 64
do
	for L in 2 4 8 16
	do
	exp_name="exp1_1" 
	srun -c 2 --gres=gpu:1  --pty python -m hw2.experiments run-exp -n ${exp_name} -K $K -L $L -P $((L/2+1)) -H 100 --batches 50
	done
done

echo "*** experiment 1.2 ***"

for L in 2 4 8
do
        for K in 32 64 128 256
        do
        exp_name="exp1_2"
        srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n ${exp_name} -K $K -L $L -P $((L/2+1)) -H 100 --batches 500
       done
done

echo "*** experiment 1.3 ***"

K="64 128 256"
for L in 1 2 3 4
do
	exp_name="exp1_3"
	srun -c 2 --gres=gpu:1  --pty python -m hw2.experiments run-exp -n ${exp_name} -K $K -L $L -P $((L/2+1)) -H 100 --batches 500
done


echo "*** experiment 1.4 ***"

K="32"
for L in 8 16 32
do
        exp_name="exp1_4"
        srun -c 2 --gres=gpu:1  --pty python -m hw2.experiments run-exp -n ${exp_name} -M resnet -K $K -L $L -P $((L/2+1)) -H 100 --batches 500
done


K="64 128 256"
for L in 2 4 8
do
        exp_name="exp1_4"
        srun -c 2 --gres=gpu:1  --pty python -m hw2.experiments run-exp -n ${exp_name} -M resnet -K $K -L $L -P $((L/2+1)) -H 100 --batches 500
done


echo "*** DONE ***"
