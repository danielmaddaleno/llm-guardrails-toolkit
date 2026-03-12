.PHONY: install test lint clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short

lint:
	flake8 . --max-line-length=120
	mypy . --ignore-missing-imports

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	rm -rf .pytest_cache .mypy_cache htmlcov .coverage

format:
	black .
	isort .
