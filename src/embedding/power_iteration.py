
# power_iteration.py
import random
from src.embedding.sparse_matrix import SparseMatrix
from src.embedding.vector import dot, norm, normalize


def xtx_matvec(X: SparseMatrix, v: list[float]) -> list[float]:
    """计算 X^T X v"""
    return X.rmatvec(X.matvec(v))


def random_unit_vector(dim: int) -> list[float]:
    """生成 dim 维随机单位向量"""
    while True:
        # 理论上选择全1的向量然后normalize也可以，只是可能存在正交的可能
        v = [random.gauss(0,1) for _ in range(dim)]
        len = norm(v)
        if len>0:
            return normalize(v)
        

def power_iteration(
    X: SparseMatrix,
    basis:list[list[float]],
    max_iter: int = 1000,
    tolerance: float = 1e-8,
):
    """
    求 X^T X 最大特征值对应的单位特征向量
    """
    dim = X.cols
    v = random_unit_vector(dim)
    # 先保证初始向量就在正交补空间
    v= orthogonalize(v,basis)
    v = normalize(v)

    cur_iter = 0
    while cur_iter<max_iter:
        u = xtx_matvec(X,v)

        # 去掉已经找到的方向
        u = orthogonalize(u,basis)

        norm_u = normalize(u)
        """
        判断方向是否开始收敛，这里可以采用不同的判断方式:
        - delta向量的长度
        - cos(theta)值
        """
        delta = norm([norm_u[i]-v[i] for i in range(dim)])
        v = norm_u
        if delta<tolerance:
            break
        cur_iter+=1

    XTXv = xtx_matvec(X,v)
    eigenvalue = dot(v,XTXv)
    return eigenvalue,v

# 进行deflation，去除掉之前计算得到的矩阵中的主要方向，这里考虑的是一个单位球上的方向
def orthogonalize(v:list[float],basis:list[list[float]]):
    pass
    




