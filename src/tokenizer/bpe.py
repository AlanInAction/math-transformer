import unicodedata
import logging

# BPE: byte pair encoding, start from sigma N=1, construct vocabulary by symbol merge

def bpe(charseq: str, n_ite_limit: int):
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

    print(symbol_list)
    print(cardinal_map)
    for i in range(seq_n):
        print("{}-{}".format(cardinal_map[charseq[i]], total_cardinal))
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
    del cardinal_map
    cardinal_map = dict()
    print("cardinal map:{}".format(cardinal_map))
    
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
    new_symbol_list = bpe(lint_char_seq, ite)
    with open(path + "res.txt", "w+", encoding="UTF-8") as f:
        f.write("|".join(new_symbol_list))

def bpe_test():
    test_str = "林冲风雪山神庙，武松喋血岳阳楼，林冲误入白虎堂，鲁智深智取二龙山" # 修复：避免使用内置关键字 str 作为变量名
    print(test_str[0:4])
    bpe(test_str, 4)

if __name__ == "__main__": # 修复：还原被替换的 __name__ 和 __main__
    bpe_test()
    path = "/Users/alan/dev/refactor/math-lab/corpus/西游记.txt"
    n_limit = 100000
    ite_limit = 300
    bpe_chapter(path, n_limit, ite_limit)