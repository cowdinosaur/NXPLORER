VENV=.venv
PY=${VENV}/bin/python
PIP=${VENV}/bin/pip

.PHONY: venv install create-dummy run validate test

venv:
	python3 -m venv ${VENV}

install: venv
	${PIP} install --upgrade pip
	${PIP} install -r requirements.txt

create-dummy: install
	${PY} scripts/create_dummy_model.py

run: install
	${PY} scripts/run_full_scan.py

validate: install
	${PY} scripts/run_full_scan.py validate

test: install
	${PY} -m pytest
