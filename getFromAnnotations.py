def getFromAnnotations(folder):
    with open(folder) as f:
        lines = f.read().splitlines()
        return lines
