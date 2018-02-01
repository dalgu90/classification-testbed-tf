# classification-testbed-tf

A personal testbed for classification on MNIST / CIFAR datasets

### Download datasets

Before running a script, download the dataset you need as below(on the project root).

```
$ python scripts/download_mnist.py  # MNIST
$ python scripts/download_cifar.py --dataset cifar-10  # CIFAR-10
$ python scripts/download_cifar.py --dataset cifar-100  # CIFAR-100
```

### Run

```
$ ./lenet_mnist_scratch.sh
$ ./lenet5_cifar10_scratch.sh
```
