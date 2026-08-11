from src.tokenizer import bpe
import os
import json
cur_path = os.path.curdir
SPARSE_JOIN_KEY = "-"
MONKEY_KING = "孙悟空"
ZUSHI = "祖师"
# co_occurrence embedding input:symbol_sequence after tokenizer
def co_occurrence_embedding(symbol_sequece:list,token_id_map_fname:str):
    print("start embedding")
    # 1. symbol sequce Seuence S
    
    # 2. vocabulary V
    vocabulary_occur = dict()
    token_id_map = dict()
    reverse_id_token_map = dict()
    for i in range(len(symbol_sequece)):
        bpe.countByOne(vocabulary_occur,symbol_sequece[i])
    cur_token_id = 0
    for k,v in vocabulary_occur.items():
        # print("{}-{}".format(k,v))
        token_id_map[k] = cur_token_id
        reverse_id_token_map[cur_token_id] = k
        cur_token_id+=1
    dimension = cur_token_id

    target_token_id_map_fname = os.path.join(cur_path,token_id_map_fname)
    with open(target_token_id_map_fname,"w+",encoding="UTF-8") as f:
        json.dump(token_id_map,f,ensure_ascii=False,indent=2)    
    # 3. co-occurrence
    sparse_co_occurrence_map = dict()
    for i in range(len(symbol_sequece)):
        right_index = i+1
        if right_index>=len(symbol_sequece):
            break
        set_sparse_matrix_val(sparse_co_occurrence_map,token_id_map.get(symbol_sequece[i]),token_id_map.get(symbol_sequece[right_index]))
    co_occur_val = get_sparse_matrix_val(sparse_co_occurrence_map,0,1)
    print("{}-{}:{}".format(reverse_id_token_map.get(0),reverse_id_token_map.get(1),co_occur_val))

    token_id_of_monkey_king = token_id_map.get(MONKEY_KING)
    vector_of_monkey_king = get_sparse_matrix_vector(sparse_co_occurrence_map,token_id_of_monkey_king,dimension)
    print("孙悟空向量:{}".format(vector_of_monkey_king))
    token_id_of_zushi = token_id_map.get(ZUSHI)
    vector_of_zushi = get_sparse_matrix_vector(sparse_co_occurrence_map,token_id_of_zushi,dimension)
    print("祖师向量:{}".format(vector_of_zushi))

def get_sparse_matrix_vector(sparse_co_occur_map:dict,token_id,dimension:int):
    vector = list()
    for i in range(dimension):
        vector.append(get_sparse_matrix_val(sparse_co_occur_map,token_id,i))
    return vector


def set_sparse_matrix_val(sparse_co_occur_map:dict,i,j:int):
    # sparse_key = "{}{}{}".format(i,SPARSE_JOIN_KEY,j)
    sparse_key = (i,j)
    co_occur_val = sparse_co_occur_map.get(sparse_key)
    if co_occur_val==None:
        sparse_co_occur_map[sparse_key] = 1
    else:
        sparse_co_occur_map[sparse_key] = co_occur_val+1

def get_sparse_matrix_val(sparse_co_occur_map:dict,i,j:int):
    # sparse_key = "{}{}{}".format(i,SPARSE_JOIN_KEY,j)
    sparse_key = (i,j)
    # co_occur_val = 0
    # if i==j:
    #     co_occur_val = 1
    # else:
    #     co_occur_val = sparse_co_occur_map.get(sparse_key)
    co_occur_val = sparse_co_occur_map.get(sparse_key)
    
    if co_occur_val == None:
        co_occur_val = 0
    return co_occur_val

    
if __name__=="__main__":
    
    path = "/Users/alan/dev/refactor/math-lab/corpus/西游记.txt"
    join_symbol = "|"
    symbol_sequence_fname = "symbol_sequence.txt"
    token_id_map_fname = "vocabulary.json"
    n_limit = 10000
    ite_limit = 300
    # 1.tokenizer
    target_symbol_seq_fname = os.path.join(cur_path,symbol_sequence_fname)
    if os.path.exists(target_symbol_seq_fname):
        with open(target_symbol_seq_fname,"r",encoding="UTF-8") as f:
            lines = f.readlines()
        symbol_sequence = lines[0].split(join_symbol)
    else:
        corpus = bpe.get_corpus(path,50000)
        symbol_sequence = bpe.bpe(corpus,ite_limit)
        print("symbol_sequence after bpe:{}".format(symbol_sequence))
        with open(target_symbol_seq_fname,"w+",encoding="UTF-8") as f:
            f.write(join_symbol.join(symbol_sequence))
    
    # 2.embedding
    co_occurrence_embedding(symbol_sequence,token_id_map_fname)