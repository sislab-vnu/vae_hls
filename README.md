# vae_hls

Variational AutoEncoder (VAE) with High-level Synthesis Acceleration

This work contains the design of a Variational AutoEncoder for
Hazelnut abnormaly detection using [the MVTec AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad).

This work optimizes [the original model at Kaggle](https://www.kaggle.com/code/jraska1/mvtect-hazelnut-variational-autoencode-ii) to make it suitable
for hardware implementation.

## Getting started

To train the model, run the following command:

``` bash
git clone https://github.com/sislab-vnu/vae_hls.git
cd vae_hls
make dataset/hazelnut
make venv
. venv/bin/activate
```

# Contributors:
- Thien-Duy HO
- Tung-Bach NGUYEN
- Tuan-Phong
- Duy-Hieu Bui
- Xuan-Tu Tran
