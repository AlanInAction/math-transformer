import math

def dot(v1: list[float], v2: list[float]) -> float:
    """向量内积 <v1, v2>"""
    dot_val = 0
    for i in range(len(v1)):
        dot_val += v1[i] * v2[i]
    return dot_val


def norm(v: list[float]) -> float:
    """向量长度 ||v||"""
    dot_val = 0
    for i in range(len(v)):
        dot_val += v[i] * v[i]
    return math.sqrt(dot_val)


def normalize(v: list[float]) -> list[float]:
    """归一化 v / ||v||"""
    v_norm = [0.0] * len(v)
    norm_of_v = norm(v)
    for i in range(len(v)):
        v_norm[i] = v[i]/norm_of_v
    return v_norm

