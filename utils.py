

def set_random_seed(seed=42):
    gens = (np.random.seed, torch.manual_seed, torch.cuda.manual_seed)
    for set_state in gens:
        set_state(seed)


def parse(path):
    f = open(path, 'r')
    for l in f:
        yield eval(l)

