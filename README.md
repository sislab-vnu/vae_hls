# vae_hls

Variational AutoEncoder (VAE) with High-level Synthesis Acceleration

This work contains the design of a Variational AutoEncoder for
Hazelnut abnormaly detection using [the MVTec AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad).

This work optimizes [the original model at Kaggle](https://www.kaggle.com/code/jraska1/mvtect-hazelnut-variational-autoencode-ii) to make it suitable
for hardware implementation.

## Requirements
- Python 3.10
- Tensorflow 2.18
- Matplotlib 3.10
- Numpy 2.0.2
- Fxpmath 0.49
- Tqdm 4.67
- Vivado/Vitis HLS 2021.2

## Getting started

Download the repository and 

``` bash
git clone https://github.com/sislab-vnu/vae_hls.git
cd vae_hls
make dataset/hazelnut
make venv
```

Train the model using the default network:

``` bash
make train
```

Run the model with test data:

``` bash
make test
```

Run the HLS

``` bash
make hls
```

Generate bitstream for Pynq-Z2:

``` bash
make vae.bit
```

# Contributors:
- Thien-Duy HO, VNU-UET.
- Tung-Bach NGUYEN, VNU-UET.
- Tuan-Phong TRAN, Phenika University.
- Duy-Hieu BUI, VNU-ITI.
- Xuan-Tu TRAN, VNU-ITI.
