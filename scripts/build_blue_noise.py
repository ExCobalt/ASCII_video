"""Generate the project's periodic 64×64 blue-noise threshold ranks."""

from pathlib import Path

import numpy as np


def blue_noise_ranks():
    size = 64
    cells = size * size
    pattern = np.random.Generator(np.random.PCG64(0)).permutation(cells).reshape(size, size) < cells // 2
    # Binomial weights give an integer approximation to a Gaussian neighborhood.
    weights = np.array([1, 8, 28, 56, 70, 56, 28, 8, 1])
    kernel = np.zeros((size, size), dtype=np.int64)
    kernel[:9, :9] = np.outer(weights, weights)
    kernel = np.roll(kernel, (-4, -4), axis=(0, 1))

    def neighborhood(index):
        return np.roll(kernel, divmod(int(index), size), axis=(0, 1))

    density = sum(neighborhood(index) for index in np.flatnonzero(pattern))
    # Move the most crowded dot into a less crowded hole until no move improves it.
    while True:
        dot = np.where(pattern, density, -1).argmax()
        density -= neighborhood(dot)
        hole = np.where(~pattern, density, kernel.sum() + 1).argmin()
        if density.flat[hole] >= density.flat[dot]:
            density += neighborhood(dot)
            break
        pattern.flat[dot], pattern.flat[hole] = False, True
        density += neighborhood(hole)

    ranks = np.empty((size, size), dtype="<u2")
    for occupied in (True, False):
        remaining = pattern.copy() if occupied else ~pattern
        crowding = density.copy() if occupied else kernel.sum() - density
        for step in range(cells // 2):
            index = np.where(remaining, crowding, -1).argmax()
            ranks.flat[index] = cells // 2 - 1 - step if occupied else cells // 2 + step
            remaining.flat[index] = False
            crowding -= neighborhood(index)
    return ranks


if __name__ == "__main__":
    np.save(Path(__file__).resolve().parents[1] / "assets/blue-noise.npy", blue_noise_ranks())
