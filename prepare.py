import subprocess
import time
import pathlib


def run(args, prompt):
    print(prompt)
    start_time = time.time()
    subprocess.run(args, stdout=subprocess.DEVNULL)
    end_time = time.time()
    print(f"  ...done, elapsed in {end_time - start_time:.2f} seconds")


def main():
    run(["python3", "-m", "venv", "venv"],
        "> creating virtual environment in ./venv/")
    run(["./venv/bin/pip3", "install", "-r", "requirements.txt"],
        "> installing pip dependencies")
    run(["git", "clone", "https://github.com/arminbiere/kissat.git"], "> cloning kissat")
    run(["/bin/bash", "-c", "cd ./kissat && ./configure && make -j"],
        "> building kissat")

    pathlib.Path("./data").mkdir(exist_ok=True)

    print("\n> everything done, don't forget to activate the virtual environment")


if __name__ == "__main__":
    main()
