.PHONY: lint lint-fix format-check

lint:
	python -m ruff check .

lint-fix:
	python -m ruff check . --fix

format-check:
	python -m ruff format --check .