# Examples

## OpenVPN setup
In the beginning, you may need to use [OpenVPN](https://github.com/szegedai/examples/blob/master/READMEOpenvpn.md)

## Environment setup
Compatibility table

| Name                                                                      | Installation time | Python | Tensorflow | PyTorch | FoolBox |
|---------------------------------------------------------------------------|-------------------|--------|------------|---------|---------|
| [env0](https://github.com/szegedai/examples/tree/master/environment/env0) |       2 min            | 3.6    | 2.1        | N.A.    | 3.3.1   |
| [env1](https://github.com/szegedai/examples/tree/master/environment/env1) |       2 min            | 3.6    | N.A.       | 1.4.0   | 3.3.1   |
| [env2](https://github.com/szegedai/examples/tree/master/environment/env2) |       2 min            | 3.8.5  | 2.4.1      | N.A.    | 3.3.1   |
| [env3](https://github.com/szegedai/examples/tree/master/environment/env3) |       2 min            | 3.8.5  | 2.6.2      | N.A.    | 3.3.1   |

A description of custom environment setup can be found [here](https://github.com/szegedai/examples/tree/master/environment)

## Model training

| Link | Dataset | Model |
|------|----------|-------|
| [link](https://github.com/szegedai/examples/tree/master/training/mnist) | [MNIST](http://yann.lecun.com/exdb/mnist/)    | [madry](https://arxiv.org/abs/1706.06083) |


## Robustness evaluation

| Attack | Link |Library | Dataset | 
|------|----------|----------|--------|
| PGD | [link](https://github.com/szegedai/examples/tree/master/robustness/mnist) | TensorFlow |[MNIST](http://yann.lecun.com/exdb/mnist/)   |
| APGD | [link](https://github.com/szegedai/examples/tree/master/autopgd) | TensorFlow |[MNIST](http://yann.lecun.com/exdb/mnist/)   |
| AutoAttack | [link](https://github.com/szegedai/examples/tree/master/autoattack) | TensorFlow |[MNIST](http://yann.lecun.com/exdb/mnist/) |
| AutoAttack | [link](https://github.com/szegedai/examples/tree/master/autoattackpytorch) | Torch |[CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) |

## Defenses
| Link | Dataset |
|------|----------|
| [link](https://github.com/szegedai/examples/tree/master/defenses/mnist) | [MNIST](http://yann.lecun.com/exdb/mnist/)    |
| [link](https://github.com/szegedai/examples/tree/master/union) | [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)    |
