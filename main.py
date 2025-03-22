# This is a sample Python script.
from multiprocessing import Pool, cpu_count
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import time
import numpy as np

def print_cube(num):
    time.sleep(num)
    print(f"hello from process{num}")
    aa1 = num * num * num
    time.sleep(2)
    return aa1


def main():
    mat = np.arange(1, 10).reshape(3, 3)
    print(np.round(mat/7, 2))


if __name__ == "__main__":
    main()