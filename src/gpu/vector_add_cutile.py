import numpy as np
import cupy as cp
import cuda.tile as ct


@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)

    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))

    result = a_tile + b_tile

    ct.store(c, index=(pid,), tile=result)


def main():
    vector_size = 2 ** 12
    tile_size = 2 ** 4

    rng = cp.random.default_rng()
    a = rng.random(vector_size)
    b = rng.random(vector_size)
    c = cp.zeros_like(a)

    grid = (ct.cdiv(vector_size, tile_size), 1, 1)

    ct.launch(
        cp.cuda.get_current_stream(),
        grid,
        vector_add,
        (a, b, c, tile_size),
    )

    expected = cp.asnumpy(a + b)
    actual = cp.asnumpy(c)

    np.testing.assert_array_almost_equal(actual, expected)

    print("cuTile vector_add test passed!")


if __name__ == "__main__":
    main()