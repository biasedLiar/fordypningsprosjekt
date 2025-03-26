# This is a sample Python script.
from multiprocessing import Pool, cpu_count
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import time
import numpy as np
from helper.fileHelper import *
import markdown as md


def main():
    layers = [1, 2, 3]
    lr = 2
    dir = 'result'

    with open('markdown\\readme.md', 'ab+') as f:
        f.write('## Layer sizes\n'.encode())

        for layer in layers:
            f.write(f'* {layer}\n'.encode())

        f.write(f'## Learning rate is {lr}\n'.encode())
        f.write('## Init\n'.encode())
        f.write('Add note about init\n'.encode())

    #md.markdownFromFile(input=open('markdown\\readme.md', "rb"), output=open('markdown_out\\out.md', "wb+"))


def write_file():
    path =""

if __name__ == "__main__":
    main()