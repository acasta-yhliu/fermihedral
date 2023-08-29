from .model import MajoranaModel


class DecentSolver:
    def __init__(self, n: int, progess: bool = False) -> None:
        self.n = n
        self.weight = 2 * n * n
        self.model = MajoranaModel(n, self.weight, progess)
        self.progess = progess

    def solve(self):
        satisfied_model = None

        while True:
            if self.progess:
                print(f"solving with weight {self.weight} on {self.n} qubits")

            if self.model.check():
                satisfied_model = self.model.solve()
                self.weight -= 1
                self.model.restrict_weight(self.weight)
            else:
                return self.weight + 1, satisfied_model
