.PHONY: all venv test generate rescore reproduce clean

PYTHON ?= python
VENV_DIR ?= .venv

all: test

venv:
	$(PYTHON) -m venv $(VENV_DIR)
	./$(VENV_DIR)/bin/pip install --upgrade pip
	./$(VENV_DIR)/bin/pip install -e .
	./$(VENV_DIR)/bin/pip install pytest

test:
	$(PYTHON) -m pytest -q

generate:
	$(PYTHON) scripts/generate.py --split dev --seed 1 --profile full
	$(PYTHON) scripts/generate.py --split dev --seed 2 --profile full
	$(PYTHON) scripts/generate.py --split dev --seed 3 --profile full

rescore:
	$(PYTHON) scripts/rescore_from_archive.py --allow-dirty

reproduce: test generate rescore
	@echo "================================================================================"
	@echo "Reproduction complete. All numbers verified offline with zero API calls."
	@echo "================================================================================"
