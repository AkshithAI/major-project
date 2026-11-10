.PHONY: install dev lint test clean profile inspect

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

lint:
	ruff check src/ tests/
	mypy src/adhie/

test:
	pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info

profile:
	python -m adhie.cli.main profile

inspect:
	@echo "Usage: make inspect MODEL=/path/to/model.gguf"
	python -m adhie.cli.main inspect $(MODEL)

