import subprocess
import sys

def main():
    for dataset in['zhang']:
        print(f"Calling train.py with argument: {dataset}")
        for i in [0]:
            fold = str(i)
            print(f"Calling train.py with argument: {fold}")
            subprocess.run([sys.executable, "train.py", "--fold", fold, "--dataset", dataset])


if __name__ == "__main__":
    main()