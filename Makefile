.PHONY: help install test lint format clean docker-up docker-down

help:
	@echo "ML-Monitor Makefile Commands:"
	@echo "  make install      - Install Python dependencies"
	@echo "  make test         - Run tests with coverage"
	@echo "  make lint         - Run linting (flake8, mypy)"
	@echo "  make format       - Format code with black"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make docker-up    - Start all Docker services"
	@echo "  make docker-down  - Stop all Docker services"

install:
	pip install -r requirements.txt
	pre-commit install

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/
	isort src/ tests/

clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	find . -type d -name '*.egg-info' -exec rm -rf {} +
	rm -rf build dist .coverage htmlcov/ .pytest_cache/ .mypy_cache/

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f
