.PHONY: all
all:
	@echo "avaliable targets:"
	@echo "    decent        use decent solver to solve optimal mapping"
	@echo "    spill         calculate weight when extra qubits are avaliable"
	@echo "    plot          plot something"

.PHONY: decent
decent:
	python3 decent-test.py 12 0

.PHONY: spill
spill:
	python3 spill-test.py 4

.PHONY: plot
plot:
	python3 -B plot.py
