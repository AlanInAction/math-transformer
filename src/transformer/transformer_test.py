
from src.transformer.transformer import Matrix,Vector
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

if __name__=="__main__":
    test_matmul()
    print_matmul()