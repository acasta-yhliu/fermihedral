from .model import MajoranaModel

if __name__ == '__main__':
    print("solving with example: 4 qubits, 32 max weights")
    majorana_model = MajoranaModel(4, 32)
    print(majorana_model.solve())