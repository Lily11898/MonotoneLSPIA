"""Compare dense and CSR collocation storage and one LSPIA kernel."""

import time

import numpy as np

from monotone_lspia import basis_matrix

rng = np.random.default_rng(20260730)
print("samples,control_points,density,dense_bytes,sparse_bytes,dense_seconds,sparse_seconds")
for sample_size, control_points in zip((1000, 5000, 10000), (16, 32, 64), strict=True):
    t = np.linspace(0, 1, sample_size)
    sparse = basis_matrix(t, control_points)
    dense = sparse.toarray()
    y = np.sin(t) + 0.02 * rng.standard_normal(sample_size)
    coefficients = np.zeros(control_points)
    step = 0.99 * 2 / np.max(np.sum(dense.T @ dense, axis=1))

    def kernel(matrix, response=y, initial=coefficients, step_size=step):
        return initial + step_size * (matrix.T @ (response - matrix @ initial))

    start = time.perf_counter()
    for _ in range(20):
        kernel(dense)
    dense_seconds = (time.perf_counter() - start) / 20
    start = time.perf_counter()
    for _ in range(20):
        kernel(sparse)
    sparse_seconds = (time.perf_counter() - start) / 20
    sparse_bytes = sparse.data.nbytes + sparse.indices.nbytes + sparse.indptr.nbytes
    print(
        f"{sample_size},{control_points},{sparse.nnz / dense.size:.8g},"
        f"{dense.nbytes},{sparse_bytes},{dense_seconds:.8g},{sparse_seconds:.8g}"
    )
