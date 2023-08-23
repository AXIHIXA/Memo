import time

import numpy as np

import build.cpx as cpx


def main():
    size: int = 1000000000
    runs: int = 2
    res: float = 0.0  # placeholder
    rng = np.random.default_rng()

    start: float = time.time()
    arr: np.ndarray = rng.random(size, np.float64)  # float32 overflows for this test
    end: float = time.time()
    print(f'Python initialization took: {(end - start) * 1000} ms')

    start = time.time()
    for _ in range(runs):
        res = np.sum(arr).item()
    end = time.time()
    print(f'Python built-in reduction took: {(end - start) * 1000} ms')
    print(f'Python built-in reduction returns: {res}')

    start = time.time()
    for _ in range(runs):
        res = cpx.cu(arr)
    end = time.time()
    print(f'C++ STL (TBB-based) parallel reduction took: {(end - start) * 1000} ms')
    print(f'C++ STL (TBB-based) parallel reduction returns: {res}')


if __name__ == '__main__':
    main()
