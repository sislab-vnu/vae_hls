venv:
	echo "Create virtual environment"
	python3 -m venv venv
	./venv/bin/pip3 install -r ./src/requirements.txt
dataset/hazelnut:
	make -C dataset
