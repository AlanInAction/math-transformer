import logging
import os
from src.tokenizer import bpe
from src.embedding import co_occurrence_embedding
import math
import random

logger = logging.getLogger(__name__)
cur_path = os.path.curdir
SPARSE_JOIN_KEY = "-"
MONKEY_KING = "孙悟空"
ZUSHI = "祖师"
DASHENG = "齐天大圣"

# # delta function in FSM (Finite State Machine)
# def attention_delta(
#         state: torch.Tensor,
#         w_q:torch.Tensor,
#         w_k:torch.Tensor,
#         w_v:torch.Tensor
# ) -> torch.Tensor:
#     q = state @ w_q
#     k = state @ w_k
#     v = state @ w_v

#     scores = q @ k.transpose(-2,-1)
#     attention = torch.softmax(scores,dim=-1)
#     delta = attention @ v
#     return delta

class Vector:
    """
    Finite-dimensional function:
    V:I->R
    """
    def __init__(self,index_set,values_set):
        if len(index_set)!=len(values_set):
                    raise ValueError("can't map I->R ,index set cardinal:{}->" \
                    "value set cardinal:{}".format(len(index_set),len(values_set)))
        self.index_set = index_set        # index set [1:d],simplified as d 
        self.values_set = values_set
        
    @property   
    def dimension(self):
        return len(self.index_set)

    @classmethod        # class method like java, create a new vector,rather than called by an exist vector
    def random(cls,dimension):
        index_set = list(range(0,dimension))  # [0,dimension)
        value_set = [random.uniform(-1,1) for _ in index_set]   # random.random()
        return cls(index_set,value_set)

    @classmethod
    def zeros(cls,dimension):
        index_set = list(range(0,dimension))
        values_set = [0.0] * dimension
        return cls(index_set,values_set)


    def vec_add(self,y:"Vector")->"Vector":
        if y.dimension != self.dimension:
             raise ValueError("can't perform vec add in different dimension {}:" \
             "{}".format(self.dimension,y.dimension)) 
        values_set = [self.values_set[i]+y.values_set[i] for i in self.index_set]
        return Vector(self.index_set,values_set)

    def scalar_mul(self,scalar:float)->"Vector":
        values_set = [self.values_set[i] * scalar for i in self.index_set ]
        return Vector(self.index_set,values_set) 

    def dot(self,y:"Vector")->float:
        # if self.dimension!=y.dimension:
        #      raise ValueError("can't perform dot in different dimension {}:" \
        #      "{}".format(self.dimension,y.dimension))
        if self.index_set != y.index_set:
            raise ValueError("can't perform dot in different Index Set {}:" \
             "{}".format(self.dimension,y.dimension))
        values_set = [ self.values_set[i] * y.values_set[i] for i in self.index_set ]
        dot_sum  = sum(values_set)
        return dot_sum



class Matrix:
    def __init__(self,outter_index_set,inner_index_set,values_set:list[Vector]):
        """
        Matrix 
        Multi Index Set Functino
        IxJ->R
        """

        self.outter_index_set = outter_index_set
        self.inner_index_set = inner_index_set
        # {M(i,·) ｜ i /belongs I}
        self.values_set = values_set

        # check I->value set
        if len(self.outter_index_set)!=len(self.values_set):
                raise ValueError("can't map I->value set ,index set cardinal:{}->" \
                "value set cardinal:{}".format(len(self.outter_index_set),len(self.values_set)))
        # check column_index_set -> Vector
        for i in range(len(values_set)):
            if values_set[i].dimension!=len(self.inner_index_set):
                raise ValueError("can't map I->value set ,index set cardinal:{}->" \
                "value set cardinal:{}".format(len(self.inner_index_set),values_set[i].dimension))

    @property
    def shape(self):
         return (len(self.outter_index_set),len(self.inner_index_set))

    @classmethod
    def random(cls,m,n)->"Matrix":
        outter_index_set = list(range(0,m)) # [0,m)
        inner_index_set = list(range(0,n))
        values_set = [Vector.random(len(inner_index_set)) for _ in outter_index_set]
        return cls(outter_index_set,inner_index_set,values_set)

    @classmethod
    def zeros(cls,m,n)->"Matrix":
        outter_index_set = list(range(0,m)) # [0,m)
        inner_index_set = list(range(0,n))
        values_set = [Vector.zeros(len(inner_index_set)) for _ in outter_index_set]
        return cls(outter_index_set,inner_index_set,values_set)

    def matmul_vector(self,vector:Vector)->Vector:
        """
        Matrix: IxJ 
        vector: Jx1
        result: Ix1
        """
        if len(self.inner_index_set)!=vector.dimension:
                     raise ValueError("can't perform vector multiply on different Index Set{}->" \
                     "{}".format(len(self.inner_index_set),vector.dimension))
        index_set = self.outter_index_set
        values_set = [self.values_set[i].dot(vector) for i in index_set]
        return Vector(index_set,values_set)

    

    def matmul_matrix(self,other:"Matrix")->"Matrix":
        """
        MatrixA: IxJ 
        MatrixB: JxK
        result: IxK
        """
        outter_index_set = self.outter_index_set 
        common_index_set = self.inner_index_set
        inner_index_set = other.inner_index_set
        if len(common_index_set)!=len(other.outter_index_set):
             raise ValueError("can't perform matrix multiply on different Index Set {}->" \
             "{}".format(len(common_index_set),len(other.outter_index_set)))
        """
        Matrix A是固定I 进行存储    I {A(i,·):J->R}
        Matrix B是固定J 进行存储    J {B(j,·):K->R}
        """
        # first I
        outter_values_list = list()
        for i in self.outter_index_set: 
            # then K
            inner_values_list = list()
            for k in other.inner_index_set:
                # common J
                cur_inner_dot_val = 0
                for j in common_index_set: 
                    cur_inner_dot_val+=self.values_set[i].values_set[j]*other.values_set[j].values_set[k]
                inner_values_list.append(cur_inner_dot_val)
            # M(i,k)
            # K->R
            cur_vec = Vector(other.inner_index_set,inner_values_list)
            # i->Vec
            outter_values_list.append(cur_vec)
        # I->(K->R)
        return Matrix(outter_index_set,inner_index_set,outter_values_list)

    def transpose(self)->"Matrix":
        """
        MatrixA:     IxJ
        MatrixA^T:   JxI
        """
        outter_index_set = self.inner_index_set
        inner_index_set = self.outter_index_set
        values_set = list()
        # 原先固定I，现在是固定J
        for j in self.inner_index_set:
            cur_vec_values = list()
            for i in self.outter_index_set:
                 cur_vec_values.append(self.values_set[i].values_set[j])
            values_set.append(Vector(inner_index_set,cur_vec_values))
        return Matrix(outter_index_set,inner_index_set,values_set)

    def scalar_mul(self,scalar:float)->"Matrix":
        values_set = [
             vector.scalar_mul(scalar) for vector in self.values_set
        ]
        return Matrix(self.outter_index_set,self.inner_index_set,values_set) 

    def print(self):
        for i in self.outter_index_set:
            for j in self.inner_index_set:
                print("{}\t".format(self.values_set[i].values_set[j]))
            print("\n")

    def __str__(self):
        rows = []

        for vector in self.values_set:
            row = " ".join(
                "{:8.4f}".format(value)
                for value in vector.values_set
            )
            rows.append("[{}]".format(row))

        return "\n".join(rows)

         
    

        
# 数值稳定版，指数函数加法与乘法之间的特性
def softmax(vector:Vector)->Vector:
    index_set = vector.index_set
    values_set = vector.values_set
    max_value = max(values_set)
    exp_val = [math.pow(math.e,value-max_value) for value in values_set]
    values_sum = sum(exp_val)
    exp_normalized_valset = [val/values_sum for val in exp_val] 
    return Vector(index_set,exp_normalized_valset)

def softmax_matrix(matrix: Matrix) -> Matrix:
    values_set = [
        softmax(vector)
        for vector in matrix.values_set
    ]

    return Matrix(
        matrix.outter_index_set,
        matrix.inner_index_set,
        values_set
    )



if __name__=="__main__":
    logging.basicConfig(
        level=logging.DEBUG 
    )
    path = "/Users/alan/dev/refactor/math-lab/corpus/西游记.txt"
    dir_name = os.path.dirname(path)
    file_name = os.path.basename(path)
    file_prefix = file_name.split(".")[0]
    join_symbol = "|"
    symbol_sequence_fname = "-symbol_sequence.txt"
    token_id_map_fname = "-vocabulary.json"
    n_limit = 10000
    ite_limit = 1000
    # 1.tokenizer
    target_symbol_seq_fname = os.path.join(dir_name,"{}{}".format(file_prefix,symbol_sequence_fname))
    if os.path.exists(target_symbol_seq_fname):
        with open(target_symbol_seq_fname,"r",encoding="UTF-8") as f:
            lines = f.readlines()
        symbol_sequence = lines[0].split(join_symbol)
    else:
        corpus = bpe.get_corpus(path,50000)
        symbol_sequence,cardinal_map = bpe.bpe(corpus,ite_limit)
        print("symbol_sequence after bpe:{}".format(symbol_sequence))
        with open(target_symbol_seq_fname,"w+",encoding="UTF-8") as f:
            f.write(join_symbol.join(symbol_sequence))
    
    # 2.embedding
    window_size = 4 # window contains symbol_i
    bidirection = True

    co_matrix_tuple = co_occurrence_embedding.co_occurrence_embedding(symbol_sequence,token_id_map_fname,window_size,bidirection)

    token_id_map,sparse_co_occurrence_map,dimension = co_matrix_tuple
    token_id_of_monkey_king = token_id_map.get(MONKEY_KING)
    vector_of_monkey_king = co_occurrence_embedding.get_sparse_matrix_vector(sparse_co_occurrence_map,token_id_of_monkey_king,dimension)
    logger.debug("孙悟空向量:{}".format(vector_of_monkey_king))
    token_id_of_zushi = token_id_map.get(ZUSHI)
    vector_of_zushi = co_occurrence_embedding.get_sparse_matrix_vector(sparse_co_occurrence_map,token_id_of_zushi,dimension)
    logger.debug("祖师向量:{}".format(vector_of_zushi))

    token_id_of_dasheng = token_id_map.get(DASHENG)
    vector_of_dasheng = co_occurrence_embedding.get_sparse_matrix_vector(sparse_co_occurrence_map,token_id_of_dasheng,dimension)
    logger.debug("齐天大圣向量:{}".format(vector_of_dasheng))

    # cos_similarity = calc_cos_similarity(vector_of_monkey_king,vector_of_zushi)
    logger.debug("{}-{} 向量相似度：{}".format(MONKEY_KING,ZUSHI,co_occurrence_embedding.calc_cos_similarity(vector_of_monkey_king,vector_of_zushi)))
    logger.debug("{}-{} 向量相似度：{}".format(MONKEY_KING,DASHENG,co_occurrence_embedding.calc_cos_similarity(vector_of_monkey_king,vector_of_dasheng)))
    logger.debug("{}-{} 向量相似度：{}".format(ZUSHI,DASHENG,co_occurrence_embedding.calc_cos_similarity(vector_of_zushi,vector_of_dasheng)))

    # 3.transformer
    input_symbols = ["孙悟空","在","花果山"]
    # 3.1 symbol list -> token id list
    input_token_ids = [
        token_id_map.get(symbol)
        for symbol in input_symbols
    ]
    logger.debug("输入序列的token_ids:{}".format(input_token_ids))

    # 3.2 token id embedding
    index_set = list(range(0,dimension))
    outter_values_list  = list()
    for token_id in input_token_ids:
       # D->R  
        values_list = co_occurrence_embedding.get_sparse_matrix_vector(sparse_co_occurrence_map,token_id,dimension)
        # print("values_list:{}".format(values_list))
        outter_values_list.append(Vector(index_set,values_list))
    outter_index_set = list(range(0,len(input_token_ids)))
    # NxD -> DxN
    input_matrix = Matrix(outter_index_set,index_set,outter_values_list).transpose()
    print("shape:{}".format(input_matrix.shape))

    # 3.3 Q K V
    Wq = Matrix.random(dimension,dimension)
    Wk = Matrix.random(dimension,dimension)
    Wv = Matrix.random(dimension,dimension)
    #   dxd  dxn -> dxn
    Q = Wq.matmul_matrix(input_matrix)
    K = Wk.matmul_matrix(input_matrix)
    V = Wv.matmul_matrix(input_matrix)
    #   nxd dxn -> nxn
    scores = Q.transpose().matmul_matrix(K)
    print(scores)

    # 4.normalized
    scaled_scores = scores.scalar_mul(
         1/math.sqrt(dimension)
    )
    print("after scaled\n{}".format(scaled_scores))
    attention_weights = softmax_matrix(scaled_scores)
    print("after softmax\n{}".format(attention_weights))


    """
    5.graph propagation nxn
    restore to dxn representation
    """
    contextual = V.matmul_matrix(attention_weights)
    print("contextual\n{}".format(contextual))











    
    