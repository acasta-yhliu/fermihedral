from os import system
from os.path import exists
from typing import List


def mkname(predname: str, ext: str):
    if exists(predname):
        return mkname(predname + "_", ext)
    else:
        return predname + ext


class SATSolver:
    def __init__(self, dimacs: str) -> None:
        self._GOAL = "__goal.cnf"
        self._MODEL = "__model.sol"

        self._model = None

        # dimacs file format handler
        self.clauses = []
        self.comments = []

        for line in dimacs.splitlines():
            if line[0] == "c":
                self.comments.append(line)
            else:
                self.clauses.append(line + '\n')

    def invoke(self):
        ...

    def block(self):
        blocks = []
        for i in self._model.renaming.values():
            truth = self._model.model[i]
            blocks.append(-i if truth else i)
        self.clauses.append(f"{' '.join(map(str, blocks))} 0\n")
        self._model = None

    def check(self) -> bool:
        if self._model is None:
            with open(self._GOAL, "w+") as goal:
                goal.writelines(self.clauses)
            self.invoke()
            self._model = SATModel.from_file(self._MODEL, self.comments)
        return self._model.sat

    def model(self) -> "SATModel":
        return self._model


class Kissat(SATSolver):
    def __init__(self, dimacs: str, timeout: int = -1, relaxed: bool = False) -> None:
        super().__init__(dimacs)
        self.timeout = timeout
        self.relaxed = relaxed

    def invoke(self):
        timeout = "" if self.timeout == -1 else f"--time={self.timeout}"
        relaxed = "--relaxed" if self.relaxed else ""
        system(f"./kissat/build/kissat {timeout} {relaxed} -q {self._GOAL} >{self._MODEL}")


class Cadical(SATSolver):
    def __init__(self, dimacs: str, timeout: int = -1, relaxed: bool = False) -> None:
        super().__init__(dimacs)
        self.timeout = timeout
        self.relaxed = relaxed

    def invoke(self):
        timeout = "" if self.timeout == -1 else f"-t {self.timeout}"
        relaxed = "--relaxed" if self.relaxed else ""
        system(f"cadical {timeout} {relaxed} -q {self._GOAL} >{self._MODEL}")


class SATModel:
    @staticmethod
    def from_file(filename: str, renaming: List[str]):
        renamings = (i.split(' ')[1:] for i in renaming)
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
