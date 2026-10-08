.PHONY: install train notebook report slim app test lint all

install:          ## everything: training, notebook, dashboard, tests
	pip install -e ".[notebook,app,dev]"

train:            ## headless training of all configurations and seeds into artifacts/runs
	python -m fitting_room train --seeds all

notebook:         ## execute the notebook top to bottom (reuses cached runs)
	cd notebooks && jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 generalization_study.ipynb

report:           ## regenerate docs/RESULTS.md, compressed figures, and the README results block
	python -m fitting_room registry && python -m fitting_room report

slim:             ## small copy of the artifacts for committing or deploying the dashboard
	python -m fitting_room slim --out artifacts_slim --budget-mb 8

app:
	streamlit run app.py

test:
	pytest -q

lint:
	ruff check src tests app.py

all: train notebook report slim
