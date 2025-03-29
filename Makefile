fxp_models = vae_encoder_fixedpoint.keras vae_decoder_fixedpoint.keras
fp_models = vae_encoder_float.keras vae_decoder_float.keras
venv:
	echo "Create virtual environment"
	python3 -m venv venv
	./venv/bin/pip3 install -r ./src/requirements.txt
dataset/hazelnut:
	make -C dataset
train vae_encoder_fixedpoint.keras vae_decoder_fixedpoint.keras:
	export PYTHONPATH=$(PWD)/src
	./venv/bin/python3 src/train.py ./dataset/hazelnut
test: $(fp_models)
	export PYTHONPATH=$(PWD)/src
	./venv/bin/python3 src/test.py ./dataset/hazelnut $(fp_models)
quantize:
	export PYTHONPATH=$(PWD)/src
	./venv/bin/python3 src/quantize_model.py ./vae_encoder_float.h5 ./vae_decoder_float.h5
