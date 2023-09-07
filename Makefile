MODEL_FILE=model
RESULT_FILE=result

.PHONY: desolve
desolve:
	python3 desolve.py 12

.PHONY: bkplot
bkplot:
	python3 -B bkplot.py

.PHONY: bksolve
bksolve:
	python3 -B bksolve.py 10
