from src.embedding.vector import dot
class SparseMatrix:
    def __init__(self, rows: int, cols: int):
        # nxm的矩阵
        self.rows = rows
        self.cols = cols

        # 只保存非零元素
        # data[i][j] = value
        # python中的写法 dict[int,float]
        self.data: dict[int, dict[int, float]] = {}

    def set(self, i: int, j: int, value: float) -> None:
        """设置 X[i, j]"""
        # 额外增加了值覆盖的情况，需要考虑到pop
        if value == 0:
            row_i_dict = self.data.get(i)
            if row_i_dict is not None:
                self.data.pop(j,None)
            return

        row_i_dict = self.data.get(i)
        if row_i_dict == None:
            row_i_dict:dict[int,float] = {}
            self.data[i] = row_i_dict
        self.data[i][j] = value

    def get(self, i: int, j: int) -> float:
        """获取 X[i, j]，不存在则认为是 0"""
        row_i_dict = self.data.get(i)
        if row_i_dict==None:
            return 0.0
        cell_ij = row_i_dict.get(j)
        if cell_ij == None:
            return 0.0
        return cell_ij

    def matvec(self, v: list[float]) -> list[float]:
        """计算 Xv"""
        if len(v)!=self.cols:
            raise ValueError("dimension mismatch!")
        # Mat:n*m v:m*1 计算得到n*1，从原始的矩阵计算->稀疏计算
        result:list[float] = [0.0] * self.rows
        for i,row in self.data.items():
            for j,value in row.items():
                result[i] += value*v[j]
        return result

    def transpose(self) -> "SparseMatrix":
        """计算 X^T"""
        t_mat = SparseMatrix(self.cols,self.rows)
        t_mat.data = {} # dict[int,dict[int,float]]
        cur_cell_val = None
        for i in range(self.rows):
            for j in range(self.cols):
                cur_cell_val = self.get(i,j)
                if cur_cell_val!=None:
                    t_mat.set(j,i,cur_cell_val)
        return t_mat

    def rmatvec(self, v: list[float]) -> list[float]:
        """计算 X^T v"""
        # t_mat = self.transpose()
        # if len(v)!=t_mat.cols:
        #             raise ValueError("dimension mismatch!")
        # return t_mat.matvec(v)

        # 更便捷的计算方式
        result:list[float] = [0.0] * self.cols
        for i,row in self.data.items():
            for j,value in row.items():
                result[j] += value * v[i]
        return result

    def nnz(self) -> int:
        """非零元素数量"""
        # 稀疏矩阵到稀疏计算
        return sum(len(row) for row in self.data.values())


def xtx_matvec(X: SparseMatrix, v: list[float]) -> list[float]:
    return X.rmatvec(X.matvec(v))   

if __name__=="__main__":
    X = SparseMatrix(2, 3)
    X.set(0, 0, 1)
    X.set(0, 2, 2)
    X.set(1, 1, 3)

    v = [1,2,3]
    print(X.matvec(v))

    u = [4, 5]
    print(X.rmatvec(u))

    print(X.nnz())

    u = X.matvec(v)
    z = X.rmatvec(u)

    left = dot(u,u)
    right = dot(v,z)

    print(left-right)

