import subprocess
import sys


def main():
    for i in [0, 4]:
        fold = str(i)
        print(f"Calling train.py with argument: {fold}")
        subprocess.run([sys.executable, "train.py", "--fold", fold])


if __name__ == "__main__":
    main()
