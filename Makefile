.PHONY: all
all:
	@echo "avaliable targets:"
	@echo "    descent       	use descent solver to solve optimal mapping"
	@echo "                  	within small scales (~12 modes)"
	@echo ""
	@echo "    free-descent     use descent solver without setting algebraic"
	@echo "                     dependence to solve approxmiate optimal"
	@echo "                     mapping within small scales."
	@echo ""
	@echo "    probability   	test the probability of algebraic dependent"
	@echo "                  	within small scales (~12 modes) and strict"
	@echo "                  	minimum weight"
	@echo ""
	@echo "    clean         	clean up all experiment results and plots"

.PHONY: descent
descent:
	@python3 descent.py 8 true

.PHONY: free-descent
free-descent:
	@python3 descent.py 8 false

.PHONY: probability
probability:
	@python3 probability.py

.PHONY: clean
clean:
	@rm *.csv *.png *.cnf *.sol
