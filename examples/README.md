# Examples
To run the examples, make sure you have installed the package and moved to this examples folder (`cd examples`).

## `test_sen2venus.py`
The [test_sen2venus.py](./test_sen2venus.py) script showcases the `Sen2Venus` class (a torch `ConcatDataset` of `Sen2VenusSite` datasets) with a simple example of multi-site dataset  loading using 2 sites ('`SUDOUE-4`', '`FGMANAUS`').

```bash
python test_sen2venus.py
```


## `test_sen2venussite.py`
The [test_sen2venussite.py](./test_sen2venussite.py) script showcases the `Sen2VenusSite` class with a simple example of single-site dataset loading (`SUDOUE-4`').

```bash
python test_sen2venussite.py
```


## `train_spl.py`
The [train_spl.py](./train_spl.py) script showcases the full training of a SRGAN model.

```bash
# make sure you have lightning installed
pip install -U lightning
# clone the forked sr-pytorch-lightning
git submodule init
git submodule update
# use cuda or cpu, modify the root to use a read-efficient disk location such as a local SSD disk 
python train_spl.py --device cuda --root . 
# follow training evolution with tensorboard
tensorboard --logdir ./lightning_logs --host 0.0.0.0
```


## `visualize_samples.py`
The [visualize_samples.py](./visualize_samples.py) script showcases some data visualisation using matplotlib.
```bash
# make sure you have matplotlib - pip install matplotlib
python visualize_samples.py
```