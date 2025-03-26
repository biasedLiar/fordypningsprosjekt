from pathlib import Path

def createDirIfNotExist(path, linux=False):
    path = osFormat(path, linux)
    print(f"Path being created: '{path}'")
    Path(path).mkdir(parents=True, exist_ok=True)

def osFormat(path, linux):
    if linux:
        return path.replace(f"\\", "/")
    return path