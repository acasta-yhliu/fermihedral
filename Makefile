MODEL_FILE=model
RESULT_FILE=result

benchmark:
	./driver.py 12 120 ${MODEL_FILE} ${RESULT_FILE}

bkplot:
	python3 -B bkplot.py

bksolve:
	python3 -B bksolve.py
	kissat -q bksolve
