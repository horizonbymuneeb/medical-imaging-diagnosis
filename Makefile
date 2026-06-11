PY ?= python3
PKG := src.medimg

.PHONY: install test lint train demo clean

install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m compileall -q src tests

train:
	$(PY) -m $(PKG).train --config config.yaml

demo:
	$(PY) -m $(PKG).demo --config config.yaml

clean:
	rm -rf .pytest_cache build dist *.egg-info
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
