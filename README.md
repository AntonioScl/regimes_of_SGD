# regimes_of_SGD
Code to reproduce the paper "On the different regimes of Stochastic Gradient Descent"


What this code does:
1. Accepts many different [parameters](https://anonymous.4open.science/r/SGD_learning_regimes-9302/edm/__main__.py)
2. Perform a single training of a neural network (depending on the parameters)
3. Compute and save observables during and at the end of the trianing (depending on the parameters)

The results are saved in a `pickle` format compatible with [grid](https://anonymous.4open.science/r/grid-E629/README.md) (`grid` allows to make sweeps in the paramters)

## Paramters
A list of some of the paramters:

`--arch`    architecture  
`--act`    activation function  
`--h`    width  
`--L`   depth for `mlp` architecture, i.e. number of hidden layers     
`--alpha`   initialization scale, is the inverse of the hinge loss margin      
`--dataset`   dataset  
`--ptr`   number of training points  
`--pte`   number of test points  
`--loss`   loss function  
`--dynamics`   training dynamics  
`--bs`  batch size for `sgd` dynamics  
`--dt`   learning rate  
`--temp`   temperature, defined as `dt/(bs * h)` (it is alternative to defining the learning rate)       
`--ckpt_grad_stats`     number of train (test) points to compute the Gram matrix of the neural tangent kernel  
`--max_wall`     maximum wall time (in seconds)  
`--seed_init`  initialization seed  
`--data_chi`  depletion exponent for the teacher-student perceptron  


## Tuto: execute a single training

```
python -m edm --dataset mnist_parity --ptr 1024 --pte 2048 --arch mlp --act gelu --h 64 --L 8 --dynamics sgd --alpha 1 --dt 0.1 --bs 64 --max_wall 120 --output test.pk
```

Many parameters are set by default!

Then the data can be loaded using `pickle`
```python
import pickle

with open('test.pk', 'rb') as f:
    args = pickle.load(f)  # dict with the paramters
    data = pickle.load(f)  # all measurements
    
# data['sgd']['dynamics'] is a list of dict

print("Initial train loss is", data['sgd']['dynamics'][0]['train']['loss'])
print("Final test error is", data['sgd']['dynamics'][-1]['test']['err'])
```


## Tuto: sweeping over many parameters

Install [grid](https://anonymous.4open.science/r/grid-E629/README.md) and the current repository (`regimes_of_SGD`).
Execute the following line that makes a sweep along the parameter `dt`, note that `grid` accept python code to create the list of parameters to sweep along.

```
python -m grid tests "python -m edm --dataset mnist_parity --ptr 1024 --pte 2048 --arch mlp --act gelu --h 64 --L 8 --dynamics sgd --alpha 1 --bs 64 --max_wall 120" --dt "[2**i for i in range(-3, 1)]"
```

At the end of the execution, the runs are saved in the directory name `tests` (in this example) and can be loaded as follow
```python
import grid
runs = grid.load('tests')

print("values of dt for the different runs", [r['args']['dt'] for r in runs])
```

See more info on how to sweep and load runs using grid in the [readme](https://anonymous.4open.science/r/grid-E629/README.md).


# Phase diagrams learning rate - batch size
Perceptron with Gaussian data
```
python -m grid perceptron_chi0 "python -m edm --arch linear --alpha 128 --dataset depleted_sign --pte 32768 --loss hinge --dynamics sgd --max_wall 10000 --data_chi 0.0 --d 128 --ptr 8192" --seed_init "[i for i in range(5)]" --bs "[2**i for i in range(0, 14)]" --dt "[2**i for i in range(-5, 14)]"
```
Perceptron with data with depletion exponent chi=1
```
python -m grid perceptron_chi1 "python -m edm --arch linear --alpha 128 --dataset depleted_sign --pte 32768 --loss hinge --dynamics sgd --max_wall 10000 --data_chi 1.0 --d 128 --ptr 8192" --seed_init "[i for i in range(5)]" --bs "[2**i for i in range(0, 14)]" --dt "[2**i for i in range(-10, 13)]"
```

Fully-connected on MNIST
```
python -m grid FC_small_margin "python -m edm --dataset mnist_parity --pte 32768 --arch mlp --act gelu --L 5 --h 128 --max_wall 40000 --loss hinge --dynamics sgd --alpha 32768 --ptr 32768 --ckpt_grad_stats 128" --seed_init "[i for i in range(5)]" --bs "[2**i for i in range(11)]" --dt "[2**i for i in range(-11, 12)]"
```
```
python -m grid FC_margin_one  "python -m edm --dataset mnist_parity --pte 32768 --arch mlp --act gelu --L 5 --h 128 --max_wall 40000 --loss hinge --dynamics sgd --ckpt_grad_stats 128 --alpha 1 --ptr 16384" --seed_init "[i for i in range(5)]" --bs "[2**i for i in range(0, 13)]" --dt "[2**i for i in range(-5,8)]"
```

CNN on CIFAR
```
python -m grid CNN_small_margin "python -m edm --dataset cifar_animal --pte 32768 --arch mnas --act relu --h 32 --max_wall 80000 --loss hinge --dynamics sgd --ckpt_grad_stats 128 --alpha 32768 --ptr 16384" --seed_init "[i for i in range(5)]" --bs "[2**i for i in range(11)]" --dt "[2**i for i in range(-6,11)]"
```
```
python -m grid CNN_margin_one "python -m edm --dataset cifar_animal --pte 32768 --arch mnas --act relu --h 32 --max_wall 160000 --loss hinge --dynamics sgd --ckpt_grad_stats 128 --alpha 1 --ptr 16384" --seed_init "[i for i in range(5)]" --bs "[2**i for i in range(12)]" --dt "[2**i for i in range(-7,10)]"
```
