Using Adaptive Bound Learning to improve Certified Robustness
========================

Adaptive Bound Learning (ABL) is a *certified defense* based on ([CROWN-IBP](https://github.com/huanzhang12/CROWN-IBP)). ABL introduces adaptive Non-linear layer to futher improve the certified robustness. The details can be seen in our project report.

We mainly follow the implementation of [CROWN-IBP](https://github.com/huanzhang12/CROWN-IBP), in bound_layer.py and model_defs.py, we implement our own method.


To train ABL to gain the SOTA performance on mnist, run:
```python
python train.py --config config/prelu_mnist_crown_large.json
```

To train ABL on small mnist model, run:
```python
python train.py --config config/prelu_mnist_crown_small.json

python train.py --config config/prelu_mnist_crown_medium.json
```


To train ABL on cifar, run:
```python
python train.py --config config/prelu_cifar_dm-large_8_255.json

python train.py --config config/prelu_cifar_mid.json
```


Other config files is for origional CROWN-IBP.
