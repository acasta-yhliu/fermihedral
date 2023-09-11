.PHONY: desolve
desolve:
	python3 desolve.py 5 2

.PHONY: bkplot
bkplot:
	python3 -B bkplot.py

.PHONY: bksolve
bksolve:
	python3 -B bksolve.py 10
