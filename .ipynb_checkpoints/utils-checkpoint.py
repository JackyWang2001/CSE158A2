import os


def parse(path):
    f = open(path, 'r')
    for l in f:
        yield eval(l)
        

def read_data(path):
    data = [d for d in parse(data)]
    return data

