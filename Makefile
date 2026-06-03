.PHONY: test lint check env

env:
	python scripts/check_environment.py

test:
	python -m unittest discover -s tests

lint:
	ruff check src tests scripts

check: env lint test
