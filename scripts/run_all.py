
from itertools import product
from runner import Runner

param_space = {
    'lock': ['mcs', 'clh', 'ticket', 'ttas', 'ttas_b'],
    'threads': [1, 4, 8, 14, 28, 56],
    'pin': [1]
}

log_dir = 'files/logs'

def main():
    keys = param_space.keys()
    values = param_space.values()

    for params in product(*values):
        params_dict = dict(zip(keys, params))
        output_dir = f"{log_dir}/{params_dict['lock']}_{params_dict['threads']}_{params_dict['pin']}"
        print(f"starting runs with with params: {params_dict}")
        # avg across 10 runs
        for i in range(10):
            runner = Runner(params_dict, output_dir, str(i))
            runner()

if __name__ == "__main__":
    main()