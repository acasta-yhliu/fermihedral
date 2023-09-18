from os import system, environ
from typing import Any, Optional


class SATSolver:
    def __init__(self) -> None:
        pass

    def __call__(self, goal_path: str, model_path: str) -> bool:
        ...


class Kissat(SATSolver):
    def __init__(self, timeout: int = -1) -> None:
        super().__init__()
        self.timeout = timeout

    def __call__(self, goal_path: str, model_path: str) -> bool:
        timeout = "" if self.timeout == -1 else f"--time={self.timeout}"

        return system(f"kissat {timeout} -q {goal_path} >{model_path}") == 10


class AmazonSolver(SATSolver):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, goal_path: str, model_path: str) -> bool:
        return system(f"./amazon-solver-driver {goal_path} {model_path}")


class SATModel:
    @staticmethod
    def from_file(filename: str, *, renaming: Optional[str] = None):
        if renaming:
            with open(renaming, "r") as renaming_file:
                # filter out the renaming part
                renamings = ((tuple(i.strip().split(
                    ' ')[1:])) for i in renaming_file.readlines() if i[0] == "c")
                renamings = dict((int(i[1]), int(i[0]))
                                 for i in renamings if i[1][0] != "k")

        with open(filename, "r") as file:
            return SATModel(file.read(), renamings)

    def __init__(self, string: str, renaming: dict[int, int]) -> None:
        # parse model file
        lines = string.splitlines()

        self.sat = False
        self.model = [True]
        self.renaming = renaming

        for line in lines:
            line = line.strip()

            if len(line) == 0:
                continue

            identifier = line[0]

            if identifier == "s":
                _s, sat = line.split(' ')
                self.sat = sat == "SATISFIABLE"
            elif identifier == "v":
                _v, *partial_solutions = line.split(' ')
                self.model.extend(map(lambda x: int(x) > 0, partial_solutions))

    def __getitem__(self, key: int):
        if self.sat:
            return self.model[self.renaming[key]]
        else:
            return None
