venv:
	echo "Create virtual environment"
	python3 -m venv venv
	./venv/bin/pip3 install -r ./src/requirements.txt
dataset/hazelnut:
	make -C dataset
train:
	export PYTHONPATH=$(PWD)/src
	./venv/bin/python3 src/train.py ./dataset/hazelnut
quantize:
	export PYTHONPATH=$(PWD)/src
	./venv/bin/python3 src/quantize_model.py ./vae_encoder_float.h5 ./vae_decoder_float.h5
