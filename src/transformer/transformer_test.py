
from src.transformer.transformer import Matrix,Vector,softmax_matrix
import math
def test_matmul():
    A = Matrix(
    [0, 1],
    [0, 1, 2],
    [
        Vector([0, 1, 2], [1, 2, 3]),
        Vector([0, 1, 2], [4, 5, 6]),
    ]
    )

    B = Matrix(
        [0, 1, 2],
        [0, 1],
        [
            Vector([0, 1], [7, 8]),
            Vector([0, 1], [9, 10]),
            Vector([0, 1], [11, 12]),
        ]
    )

    C = A.matmul_matrix(B)

    assert C.values_set[0].values_set == [58, 64]
    assert C.values_set[1].values_set == [139, 154]

    v = Vector(
    [0, 1, 2],
    [10, 20, 30]
)

    result = A.matmul_vector(v)

    assert result.values_set == [140, 320]

def print_matmul():
    A = Matrix(
    [0, 1],
    [0, 1],
    [
        Vector([0, 1], [1, 2]),
        Vector([0, 1], [3, 4]),
    ]
    )

    B = Matrix(
        [0, 1],
        [0, 1],
        [
            Vector([0, 1], [5, 6]),
            Vector([0, 1], [7, 8]),
        ]
    )

    C = A.matmul_matrix(B)

    print(C.values_set[0].values_set)
    print(C.values_set[1].values_set)

def test_attention():
    # D x N
    X = Matrix(
        [0, 1],
        [0, 1, 2],
        [
            Vector([0, 1, 2], [1, 2, 3]),
            Vector([0, 1, 2], [4, 5, 6]),
        ]
    )

    Wq = Matrix(
        [0, 1],
        [0, 1],
        [
            Vector([0, 1], [1, 0]),
            Vector([0, 1], [0, 1]),
        ]
    )

    Wk = Wq
    Wv = Wq

    Q = Wq.matmul_matrix(X)
    K = Wk.matmul_matrix(X)
    V = Wv.matmul_matrix(X)

    scores = Q.transpose().matmul_matrix(K)

    scaled = scores.scalar_mul(
        1 / math.sqrt(2)
    )

    attention = softmax_matrix(scaled)

    contextual = V.matmul_matrix(attention)

    assert contextual.shape == (2, 3)

    for vector in attention.values_set:
        assert abs(sum(vector.values_set) - 1.0) < 1e-10

if __name__=="__main__":
    test_matmul()
    test_attention()
    print_matmul()
    