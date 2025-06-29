.PHONY: install train bench test all clean

install:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

train:
	.venv/bin/python train.py

bench:
	.venv/bin/python -m bench.sklearn_baseline

test:
	.venv/bin/python -m unittest discover tests

all: test train bench

clean:
	rm -rf runs/ data/raw/ __pycache__ */__pycache__ */*/__pycache__
