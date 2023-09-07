from os import system
from typing import Any


class SATSolver:
    def __init__(self) -> None:
        pass

    def __call__(self, model_path: str, solution_path: str) -> bool:
        ...


class Kissat(SATSolver):
    def __call__(self, goal_path: str, model_path: str) -> bool:
        return system(f"kissat -q {goal_path} >{model_path}") == 10


class SATModel:
    @staticmethod
    def from_file(filename: str, *, renaming: str | None = None):
        if renaming:
            with open(renaming, "r") as renaming_file:
                # filter out the renaming part
                renamings = ((tuple(i.strip().split(
                    ' ')[1:])) for i in renaming_file.readlines() if i[0] == "c")
                renamings = [int(i[0]) for i in renamings if i[1][0] != "k"]

        with open(filename, "r") as file:
            return SATModel(file.read(), renamings)

    def __init__(self, string: str, renaming: list[int]) -> None:
        # parse model file
        lines = string.splitlines()

        # first line indicates the model result
        [_s, model_satisfiability] = lines[0].split(' ')
        assert _s == "s"
        self.sat = model_satisfiability == "SATISFIABLE"

        # following lines are the model
        self.model = [True]
        for solution_string in lines[1:]:
            _v, *solution = solution_string.split(' ')
            assert _v == "v"
            self.model.extend(map(lambda x: int(x) > 0, solution))
        
        # print(renaming)
        self.renaming = renaming

    def __getitem__(self, key: Any):
        if self.sat:
            return self.model[self.renaming[key - 1]]
        else:
            return None
