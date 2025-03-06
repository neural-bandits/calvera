sources = src tests examples
mypy_sources = src tests

.PHONY: format
format:
	isort $(sources)
	black $(sources)

.PHONY: lint
lint:
	ruff check $(sources)
	isort $(sources) --check-only --df
	black $(sources) --check --diff

.PHONY: mypy
mypy:
	mypy $(ARGS) $(mypy_sources)

.PHONY: all
all:
	make format
	make lint
	make mypy
	make coverage

.PHONY: test
test:
	pytest -W ignore::DeprecationWarning  tests

.PHONY: coverage
coverage:
	pytest --cov=./src tests/ --cov-fail-under=85
