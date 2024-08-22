from pathlib import Path

def createDirIfNotExist(path):
    Path(path).mkdir(parents=True, exist_ok=True)
