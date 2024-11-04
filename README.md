# ENequIP
Nequip with external electric field.

This code is based on nequip (https://github.com/mir-group/nequip.git)
Pleas cite them as well.
## Installation

NequIP requires:

* Python >= 3.7
* PyTorch >= `1.11.*`

  ```
  git clone https://github.com/Fu-Xiaoyan/ENequIP.git
  cd ENequip
  pip install . 
  ```

## Tutorial 
### Preparing Data
The training dataset should contains applied external electric field. 
The dataset should be in `.extxyz` formate. 
The electric field should be given as array for every atom in `atom.magmom`. 
e.g. A atom with 1 electric field at Z direction should set  `atom.magmom=[0, 0, 1]`. 

### Training
Using the script `trainer.py` for MLP training
Using the script `trainer-Upzc.py` for PZC-NN training

### Trained model
`CuO-0615.pth` is the pretrained model with Cu and O elements.
`config-0615.json` is the corresponding configures.

### Using models in Python

An ASE calculator is provided, an example is shown as well. (`ASEcalculator.py`) 

The GCMC scripts are provided as well in  `optimizer` 
## Authors

ENequIP is being developed by:

 - Xiaoyan Fu
under the guidance of [Jianping Xiao at DICP](http://www.jpxiao.dicp.ac.cn/).

 - Chenhua Gen 