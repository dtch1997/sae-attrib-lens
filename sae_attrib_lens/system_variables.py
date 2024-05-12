""" Set system variables as you see fit """


import pathlib

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    print(PROJECT_DIR)