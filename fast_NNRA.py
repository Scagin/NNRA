import numpy as np


def init(sim_matrix, tags):
    """
        初始化集团度矩阵和文档隶属度矩阵
    """
    tag_set = list(set(tags))
    matrix_E = build_matrixE(sim_matrix, tags, tag_set)
    matrix_A = build_matrixA(sim_matrix, tags, tag_set)
    return matrix_E, matrix_A, tag_set


def get_candidate(candidate_set, matrix_E, matrix_A, tags, tag_set):
    """
        根据算法计算出分类错误的样本
    """
    max_prob_tags = np.argmax(matrix_A, axis=1)
    for i, (mpt, t) in enumerate(zip(max_prob_tags, tags)):
        if mpt != tag_set.index(t):
            old = tag_set.index(t)
            new = mpt
            a_old = np.sum(matrix_E[old])
            a_new = np.sum(matrix_E[new])
            d_i = np.sum(matrix_A[i])
            delta_Q = matrix_A[i][new] - matrix_A[i][old] - (d_i * (a_new - a_old) + d_i ** 2 / 2)
            if delta_Q > 0:
                candidate_set.add((i, new, delta_Q))
    return candidate_set


def get_sample_indices(tags, tag_set):
    sample_indices = []
    for tag in tag_set:
        ind = np.where(tags == tag)[0]
        sample_indices.append(ind)
    return sample_indices


def build_matrixA(sim_matrix, tags, tag_set):
    """
        构建文档隶属度矩阵
    """
    tot = np.sum(sim_matrix) / 2
    class_num = len(tag_set)
    sample_indices = get_sample_indices(tags, tag_set)

    matrix_A = np.zeros((sim_matrix.shape[0], class_num))
    for i, s in enumerate(sim_matrix):
        for j, ind in enumerate(sample_indices):
            matrix_A[i][j] = np.sum(s[ind]) / tot

    return matrix_A


def build_matrixE(sim_matrix, tags, tag_set):
    """
        构建集团关联强度矩阵
    """
    tot = np.sum(sim_matrix) / 2
    sample_indices = get_sample_indices(tags, tag_set)
    class_num = len(tag_set)
    matrix_E = np.zeros((class_num, class_num))
    for i, _ in enumerate(tag_set):
        for j, _ in enumerate(tag_set):
            matrix_E[i][j] = np.sum(sim_matrix[sample_indices[i]][:, sample_indices[j]]) / (2 * tot)
    return matrix_E


def update_matrixE(matrix_E, sub_vec, old_class, new_class):
    """
        更新集团关联强度矩阵
    """
    row, col = matrix_E.shape
    for i in range(row):
        for j in range(col):
            if i == old_class and j == old_class:
                matrix_E[i][j] -= sub_vec[old_class]
            elif i == new_class and j == new_class:
                matrix_E[i][j] += sub_vec[new_class]
            elif (i == new_class and j == old_class) or (i == old_class and j == new_class):
                matrix_E[i][j] += (sub_vec[old_class] - sub_vec[new_class]) / 2
            elif i == old_class:
                matrix_E[i][j] -= sub_vec[j] / 2
            elif j == old_class:
                matrix_E[i][j] -= sub_vec[i] / 2
            elif i == new_class:
                matrix_E[i][j] += sub_vec[j] / 2
            elif j == new_class:
                matrix_E[i][j] += sub_vec[i] / 2
            else:
                pass
    return matrix_E


def set_category(candidate_set, sim_matrix, matrix_E, tags, tag_set, alpha=1.0):
    """
        根据算法计算结果，修改成正确的标签
    """
    reversed_candidate = sorted(candidate_set, key=lambda x:x[2], reverse=True)
    max_delta_Q = reversed_candidate[0][2]
    mod_count = 0
    while len(reversed_candidate) > 0:
        (ind, new_class, delta_Q) = reversed_candidate[0]
        if delta_Q <= alpha * max_delta_Q:
            tags[ind] = tag_set[new_class]
            reversed_candidate.pop(0)
            max_delta_Q = delta_Q
            mod_count += 1
        else:
            reversed_candidate = []
    matrix_A = build_matrixA(sim_matrix, tags, tag_set)
    matrix_E = build_matrixE(sim_matrix, tags, tag_set)
    candidate_set = set()
    return matrix_E, matrix_A, tags, candidate_set, mod_count


def fix(sim_matrix, tags, alpha=1.0):
    """
        使用算法修复数据集中的噪声（错分类数据）

        sim_matrix: 数据集的相似度矩阵，shape=(m, m)，m为数据集的大小
        tags: 每条数据的原始标签tag

        return: 修复后的新标签tag
    """
    print("Start algorithm!")
    # 初始化集团度矩阵和隶属度矩阵
    matrix_E, matrix_A, tag_set = init(sim_matrix, tags)
    new_tags = tags
    candidate_set = set()
    iteration = 1
    while True:
        # 通过计算数据集的模块度增量(delta_Q)，得到增量为正数的候选修改方案
        candidate_set = get_candidate(candidate_set, matrix_E, matrix_A, new_tags, tag_set)
        diff_num = len(candidate_set)
        print("iteration {}, {} labels need to fix...".format(iteration, diff_num))
        iteration += 1
        
        if diff_num == 0:
            break
        # 按照修改方案，修改数据集的标签        
        matrix_E, matrix_A, new_tags, candidate_set, fix_num = set_category(candidate_set, sim_matrix, matrix_E, new_tags, tag_set, alpha)
        print("{} labels were fixed!".format(fix_num))

    return new_tags


if __name__ == "__main__":
    sim = np.array([[1, 0.1, 0.1, 0.1, 0.9, 0.9], 
                    [0.1, 1, 0.9, 0.9, 0.1, 0.1], 
                    [0.1, 0.9, 1, 0.9, 0.1, 0.1], 
                    [0.1, 0.9, 0.9, 1, 0.1, 0.1],
                    [0.9, 0.1, 0.1, 0.1, 1, 0.9],
                    [0.9, 0.1, 0.1, 0.1, 0.9, 1]])
    tags = np.array(["0", "1", "1", "0", "0", "0"])

    new_tags = fix(sim, tags)

    print(new_tags)

