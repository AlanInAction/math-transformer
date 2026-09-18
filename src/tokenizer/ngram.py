import logging
logger = logging.getLogger(__name__)
# ngram，metric symbol space and prefix prob
def ngram(charseq:str,nLimit,topN:int):
    # 1. Split string to N=1 symbol
    seq_n = len(charseq)
    # 2. Set
    symbol_list,cur_level = list(),1
    # 3. Metric
    cardinal_map = dict()
    total_cardinal = 0
    symbol_dist = dict()
    for i in range(seq_n):
        symbol_list.append(charseq[i])
        countByOne(cardinal_map,charseq[i])
        total_cardinal+= 1

    logger.debug(symbol_list)
    logger.debug(cardinal_map)
    symbol_dist[cur_level] = dict()
    # normalize
    for i in range(seq_n):
        logger.debug("{}-{}".format(cardinal_map[charseq[i]],total_cardinal))
        symbol_dist[cur_level][charseq[i]] = cardinal_map[charseq[i]]/total_cardinal
    logger.debug("初始概率:{}".format(symbol_dist))
    metric_tuple = (cardinal_map,total_cardinal,symbol_dist)
    set_tuple = (symbol_list,cur_level)
    bpe_param_tuple = (nLimit,topN)
    ngram_step(charseq,set_tuple,metric_tuple,bpe_param_tuple)
    
    # sorted_symbol_dist = sorted(symbol_dist.items(),key=lambda item:(-item[1],-len(item[0])))
    for level, dist in symbol_dist.items():
        logger.debug(f"N={level}: {top_n(dist, topN)}")

def top_n(dist, n):
    return sorted(
        dist.items(),
        key=lambda item: item[1],
        reverse=True
    )[:n]

def ngram_step(charSeq:str,set_tuple:tuple,metric_tuple:tuple,bpe_param_tuple:tuple):
    # 1. 解析tuple
    cardinal_map,total_cardinal,symbol_dist = metric_tuple
    symbol_list,cur_level = set_tuple
    nLimit,topN = bpe_param_tuple
    # 2.BPE N ++
    cur_level+=1
    if cur_level>=nLimit:
        return
    symbol_dist[cur_level] = dict()
    # 3.metric
    seq_n = len(symbol_list)
    # count cardinality
    cardinal_map = dict()
    prefix_cardinal_map = dict()
    

    # while循环进行滑动窗口处理
    left_index = 0
    right_index = 0
    right_limit = seq_n
    while left_index<right_limit:
        # [left,right] 左闭右开
        right_index = left_index+cur_level-1
        if right_index>=right_limit:
            break
        countByOne(cardinal_map,charSeq[left_index:right_index+1])
        countByOne(prefix_cardinal_map,charSeq[left_index:right_index]) # -1+1
        left_index+=1
    # metric update
    for left_index in range(seq_n):
        right_index = left_index+cur_level-1
        if right_index>=right_limit:
            break
        # seq N-1
        pre_seq = charSeq[left_index:right_index]
        cur_seq = charSeq[left_index:right_index+1]
        symbol_dist[cur_level][cur_seq] = symbol_dist[cur_level-1][pre_seq]*\
        (cardinal_map[cur_seq]/prefix_cardinal_map[pre_seq])
        logger.debug("序列：{}-{}-概率：{}-{}\{}".format(pre_seq,cur_seq,symbol_dist[cur_level-1][pre_seq],cardinal_map[cur_seq],prefix_cardinal_map[pre_seq]))
    metric_tuple = (cardinal_map,total_cardinal,symbol_dist)
    set_tuple = (symbol_list,cur_level)
    bpe_param_tuple = (nLimit,topN)
    ngram_step(charSeq,set_tuple,metric_tuple,bpe_param_tuple)

def countByOne(cardinal_map:dict,symbol:str):
    pre_cardinal = cardinal_map.get(symbol)
    if pre_cardinal:
        cardinal_map[symbol] = pre_cardinal+1
    else:
        cardinal_map[symbol] = 1

if __name__=="__main__":
    logging.basicConfig(
        level=logging.DEBUG
    )
    str = "林冲风雪山神庙，武松喋血岳阳楼，林冲误入白虎堂"
    logger.debug(str[0:4])
    ngram(str,4,3)



