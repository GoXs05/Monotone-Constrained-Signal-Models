PY := uv run python

.PHONY: data features test synthetic walkforward learning-curve holdout lint

data:
	$(PY) -m msig.data.download
	$(PY) -m msig.data.load
	$(PY) -m msig.data.align

features:
	$(PY) scripts/build_features.py

test:
	uv run pytest -q

synthetic:
	$(PY) scripts/run_synthetic.py

walkforward:
	$(PY) scripts/run_walkforward.py

learning-curve:
	$(PY) scripts/run_learning_curve.py

holdout:
	$(PY) scripts/run_holdout.py

lint:
	uv run ruff check .
	uv run ruff format --check .
