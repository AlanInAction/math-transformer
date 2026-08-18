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
        row_i_dict = self.data.get(i)
        if row_i_dict == None:
            row_i_dict:dict[int,float] = {}
            self.data[i] = row_i_dict
        self.data[i][j] = value

    def get(self, i: int, j: int) -> float:
        """获取 X[i, j]，不存在则认为是 0"""
        row_i_dict = self.data.get(i)
        if row_i_dict==None:
            return 0
        cell_ij = row_i_dict.get(j)
        if cell_ij == None:
            return 0
        return cell_ij

    def matvec(self, v: list[float]) -> list[float]:
        """计算 Xv"""
        if len(v)!=self.cols:
            raise ValueError("dimension mismatch!")
        # Mat:n*m v:m*1 计算得到n*1
        result:list[float] = list()
        cur_cell_val = 0
        # 1. 循环计算第i个分量
        for i in range(self.rows):
            cur_cell_val = 0
            for j in range(self.cols):
                cur_cell_val += self.get(i,j)*v[j]
            result.append(cur_cell_val)
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
        t_mat = self.transpose()
        if len(v)!=t_mat.cols:
                    raise ValueError("dimension mismatch!")
        return t_mat.matvec(v)

    def nnz(self) -> int:
        """非零元素数量"""
        nzero_count = 0
        cur_val = 0
        for i in range(self.rows):
            for j in range(self.cols):
                cur_val = self.get(i,j)
                if cur_val>0:
                    nzero_count += 1
        return nzero_count
        

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

