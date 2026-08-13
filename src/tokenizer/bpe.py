import unicodedata
import logging
import json
import os
logger = logging.getLogger(__name__)
SYMBOL_SUFFIX="-symbol.txt"
VOCAB_SUFFIX="-vocab.json"
# BPE: byte pair encoding, start from sigma N=1, construct vocabulary by symbol merge
def bpe(charseq: str, n_ite_limit: int):

    # 1. set Character Sequence
    symbol_list = list()
    seq_n = len(charseq)
    for i in range(seq_n):
        symbol_list.append(charseq[i])
   
    # 2. metric
    cardinal_map = dict()
    pair_count_map = dict()
    pair_index_map = dict()


    for i in range(seq_n):
        # 2.1 count symbol cardinality
        countByOne(cardinal_map,symbol_list[i])

    cur_ite = 0
    # 3. per iteration
    while cur_ite<n_ite_limit:
        symbol_list_len = len(symbol_list)
        for i in range(symbol_list_len):
            pair_index = i+1
            if pair_index<symbol_list_len:
                # 3.1 count symbol pair cardinality
                byte_pair = (symbol_list[i],symbol_list[pair_index])
                index_pair = (i,pair_index)
                countByOne(pair_count_map,byte_pair)
                addPairIndex(pair_index_map,byte_pair,index_pair)
        # 4. sort symbol pair cardinality
        sorted_pair_count_list = sorted(pair_count_map.items(),key=lambda item:-item[1])
        top1_symbol_item = sorted_pair_count_list[0]
        top1_symbol_pair = top1_symbol_item[0]
        top1_symbol_count = top1_symbol_item[1]
        merged_top1_symbol = tuple_symbol_merge(top1_symbol_pair)
        logger.debug("top1 symbol:{}->{},count:{}".format(top1_symbol_pair,merged_top1_symbol,top1_symbol_count))

        # 5. merge symbol pair，reverse pop
        loc_list = pair_index_map.get(top1_symbol_pair)

        new_loc_list = merge_loc_list(loc_list)

        merge_len = len(new_loc_list)
        for i in range(merge_len):
            reverse_index = merge_len-1-i
            reverse_index = new_loc_list[reverse_index]
            # replace symbol tuple to merged single symbol
            symbol_list.pop(reverse_index[1])
            symbol_list[reverse_index[0]] = merged_top1_symbol

        # 6. update cardinal map
        # can't use pop,here should just -1

        # cardinal_map.pop(top1_symbol_pair[0])
        # cardinal_map.pop(top1_symbol_pair[1])
        decreByVal(cardinal_map,top1_symbol_pair[0],merge_len)
        decreByVal(cardinal_map,top1_symbol_pair[1],merge_len)

        cardinal_map[merged_top1_symbol] = top1_symbol_count

        # 7.update pair count cardinality, can reuse pair count map in one iteration 
        # but no more iteration,so just clear it,recalculate
        pair_count_map.clear()
        pair_index_map.clear()

        cur_ite+=1
    return symbol_list,cardinal_map

def tuple_symbol_merge(top1_symbol_pair:tuple):
    return "{}{}".format(top1_symbol_pair[0],top1_symbol_pair[1])

def addPairIndex(pair_index_map:dict,byte_pair:tuple,index_tuple:tuple):
    pair_index_list = pair_index_map.get(byte_pair)
    if pair_index_list == None:
        pair_index_list = list()
    pair_index_list.append(index_tuple)
    pair_index_map[byte_pair] = pair_index_list
        
def decreByVal(cardinal_map: dict, symbol: str,decreVal:int):
    pre_cardinal = cardinal_map.get(symbol)
    if pre_cardinal>=decreVal:
        cardinal_map[symbol] = pre_cardinal - decreVal
    else:
        logger.fatal("this path should not access,symbol:{},decreVal:{},cardinal_map:{}",symbol,decreVal,cardinal_map)


# bpe_v0 基于递归的实现方式，TODO 优化递归实现过程中的内存泄漏问题
def bpe_v0(charseq: str, n_ite_limit: int):
    # 1. Split string to N=1 symbol
    seq_n = len(charseq)
    # 2. Set
    symbol_list, cur_level, cur_iter = list(), 1, 0
    # 3. Metric
    cardinal_map = dict()
    total_cardinal = 0
    symbol_dist = dict()
    
    for i in range(seq_n):
        symbol_list.append(charseq[i])
        countByOne(cardinal_map, charseq[i])
        total_cardinal += 1

    logger.debug(symbol_list)
    logger.debug(cardinal_map)
    for i in range(seq_n):
        logger.debug("{}-{}".format(cardinal_map[charseq[i]], total_cardinal))
        symbol_dist[charseq[i]] = cardinal_map[charseq[i]] / total_cardinal
        
    # print("初始概率:{}".format(symbol_dist))
    metric_tuple = (cardinal_map, total_cardinal, symbol_dist)
    set_tuple = (symbol_list, cur_level, cur_iter)
    bpe_param_tuple = (n_ite_limit,) # 修复：单元素元组需要加逗号
    new_symbol_list = bpe_rec(set_tuple, metric_tuple, bpe_param_tuple)
    
    # sorted_symbol_list = sorted(new_symbol_list, key=lambda v: -len(v))
    # print("final symbol list:{}".format(sorted_symbol_list))
    # return sorted_symbol_list
    return new_symbol_list

def top_n(dist, n):
    return sorted(
        dist.items(),
        key=lambda item: item[1],
        reverse=True
    )[:n]

def bpe_rec(set_tuple: tuple, metric_tuple: tuple, bpe_param_tuple: tuple):
    # 1. 解析tuple
    cardinal_map, total_cardinal, symbol_dist = metric_tuple
    symbol_list, cur_level, cur_iter = set_tuple
    n_ite_limit = bpe_param_tuple[0] # 修复：从元组中正确提取参数
    
    # 2. with BPE, cur_level always be 1
    cur_level = 2
    # if cur_level >= nLimit:
    #     return
    if cur_iter >= n_ite_limit:
        return symbol_list
        
    # 3. metric
    seq_n = len(symbol_list)
    pair_dist = dict()
    # count cardinality
    cardinal_map = dict()
    prefix_cardinal_map = dict()
    symbol_index_map = dict()
    pair_count = dict()

    # while循环进行滑动窗口处理
    left_index = 0
    right_index = 0
    right_limit = seq_n
    while left_index < right_limit:
        # [left,right] 左闭右开
        right_index = left_index + cur_level - 1
        if right_index >= right_limit:
            break
        countByOne(cardinal_map, symbol_merge(symbol_list[left_index:right_index+1]))
        countByOne(pair_count, symbol_merge(symbol_list[left_index:right_index+1]))
        countByOne(prefix_cardinal_map, symbol_merge(symbol_list[left_index:right_index])) # -1+1
        left_index += 1
        
    # metric update
    for left_index in range(seq_n):
        right_index = left_index + cur_level - 1
        if right_index >= right_limit:
            break
        # seq N-1
        pre_seq = symbol_merge(symbol_list[left_index:right_index])
        cur_seq = symbol_merge(symbol_list[left_index:right_index+1])
        pair_dist[cur_seq] = symbol_dist[pre_seq] * (cardinal_map[cur_seq] / prefix_cardinal_map[pre_seq])
        symbol_sub_seq_record(cur_seq, symbol_index_map, (left_index, right_index+1))
        # print("序列：{}-{}-概率：{}-{}\{}".format(pre_seq, cur_seq, symbol_dist[pre_seq], cardinal_map[cur_seq], prefix_cardinal_map[pre_seq]))

    # bpe merge
    # sorted_dict = sorted(pair_dist.items(), key=lambda item: -item[1])
    sorted_dict = sorted(pair_count.items(), key=lambda item: -item[1])
    # print("排序后dict:{}".format(sorted_dict))
    top1_symbol = sorted_dict[0][0]
    
    # update symbol list
    new_symbol_list = list()
    # print("subseq indexmap:{}".format(symbol_index_map))
    loc_list = symbol_index_map[top1_symbol]
    new_loc_list = merge_loc_list(loc_list)
    
    # update symbol dist
    # update new symbol list
    index = 0
    for i in range(len(new_loc_list)):
        left_index, right_index = new_loc_list[i] # left close right open
        while index < len(symbol_list):
            if index < left_index:
                new_symbol_list.append(symbol_list[index])
            elif index == left_index:
                new_symbol_list.append(top1_symbol)
                safe_pop(symbol_dist, symbol_list[index])
                safe_pop(cardinal_map, symbol_list[index])
            elif index < right_index:
                safe_pop(symbol_dist, symbol_list[index])
                safe_pop(cardinal_map, symbol_list[index])
                index += 1
                continue
            elif index == right_index:
                break
            index += 1
            
    while index < len(symbol_list):
        new_symbol_list.append(symbol_list[index])
        index += 1
        
    # print("bpe new symbol_list:{}".format(new_symbol_list))

    # symbol_list change
    total_cardinal = len(new_symbol_list)
    del symbol_dist
    symbol_dist = dict()
    logger.debug("cardinal map:{}".format(cardinal_map))
    del cardinal_map
    cardinal_map = dict()
   
    
    for i in range(len(new_symbol_list)):
        if cardinal_map.get(new_symbol_list[i]):
            pass
        else:
            countByOne(cardinal_map, new_symbol_list[i])
            
    for i in range(len(new_symbol_list)):
        symbol_dist[new_symbol_list[i]] = cardinal_map[new_symbol_list[i]] / total_cardinal

    del symbol_list
    metric_tuple = (cardinal_map, total_cardinal, symbol_dist)
    cur_iter += 1
    set_tuple = (new_symbol_list, cur_level, cur_iter)
    bpe_param_tuple = (n_ite_limit,) # 修复：单元素元组需要加逗号

    return bpe_rec(set_tuple, metric_tuple, bpe_param_tuple)


def merge_loc_list(loc_list: list):
    selected = list()
    last_right = -1
    last_left = -1
    for i in range(len(loc_list)):
        cur_loc = loc_list[i]
        left, right = cur_loc
        if left >= last_right:
            selected.append((left, right))
            last_left = left
            last_right = right
    # [1,3) [2,4)
    # elif right > last_right:
    #     selected.append(last_right, right)
    # else:
    #     pass
    return selected

def safe_pop(symbol_dist: dict, symbol: str):
    if symbol_dist.get(symbol):
        symbol_dist.pop(symbol)

def symbol_sub_seq_record(sub_seq: str, symbol_index_map: dict, sub_seq_tuple: tuple):
    sub_seq_list = symbol_index_map.get(sub_seq)
    if not sub_seq_list:
        sub_seq_list = list()
    sub_seq_list.append(sub_seq_tuple)
    symbol_index_map[sub_seq] = sub_seq_list

def symbol_merge(sub_symbol_seq: list):
    return "".join(sub_symbol_seq)

def countByOne(cardinal_map: dict, symbol: str):
    pre_cardinal = cardinal_map.get(symbol)
    if pre_cardinal:
        cardinal_map[symbol] = pre_cardinal + 1
    else:
        cardinal_map[symbol] = 1

def get_corpus(path: str, line_limit: int):
    with open(path, "r", encoding="UTF-8") as f:
        lines = f.readlines(line_limit)
    char_seq = "".join(lines)
    lint_char_seq = "".join(ch for ch in char_seq if not (unicodedata.category(ch).startswith("P") or ch.isspace()))
    return lint_char_seq

def bpe_chapter(path: str, line_limit: int, ite: int):
    lint_char_seq = get_corpus(path, line_limit)
    
    new_symbol_list,cardinal_map = bpe(lint_char_seq, ite)
    sorted_symbol_list = sorted(cardinal_map.items(),key=lambda item:(-len(item[0]),-item[1]))
    sorted_cardinal_map = dict()

    file_name = os.path.basename(path)
    file_prefix = file_name.split(".")[0]
    dir_name = os.path.dirname(path)
    for i in range(len(sorted_symbol_list)):
        sorted_cardinal_map[sorted_symbol_list[i][0]] = sorted_symbol_list[i][1]
    
    with open(os.path.join(dir_name,"{}{}".format(file_prefix,SYMBOL_SUFFIX)), "w+", encoding="UTF-8") as f:
        f.write("|".join(new_symbol_list))

    with open(os.path.join(dir_name,"{}{}".format(file_prefix,VOCAB_SUFFIX)),"w+",encoding="UTF-8") as f:
        json.dump(sorted_cardinal_map,f,ensure_ascii=False,indent=2)
    

def bpe_test():
    test_str = "林冲风雪山神庙，武松喋血岳阳楼，林冲误入白虎堂，鲁智深智取二龙山" 
    logger.debug(test_str[0:4])
    bpe(test_str, 4)

if __name__ == "__main__": 
    logging.basicConfig(
        level=logging.DEBUG,
    )
    bpe_test()
    path = "/Users/alan/dev/refactor/math-lab/corpus/西游记.txt"
    n_limit = 100000
    ite_limit = 1000
    bpe_chapter(path, n_limit, ite_limit)