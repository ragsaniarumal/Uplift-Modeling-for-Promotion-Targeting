.PHONY: data analysis test clean

data:
	python -m pricesense.data

analysis:
	python -m pricesense.analysis

test:
	pytest -q

clean:
	rm -rf .pytest_cache src/pricesense/__pycache__ tests/__pycache__
