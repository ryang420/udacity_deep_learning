def augmentMatrix(A, b):
    return [AA + bb for AA, bb in zip(A, b)]


def swapRows(M, r1, r2):
    M[r1], M[r2] = M[r2], M[r1]


def scaleRow(M, r, scale):
    if scale == 0:
        raise ValueError('scale不能是0')
    M[r] = [a * scale for a in M[r]]


def addScaledRow(M, r1, r2, scale):
    M[r1] = [a + b * scale for a, b in zip(M[r1], M[r2])]


def matxRound(M, decPts=4):
    for j, row in enumerate(M):
        for i, c in enumerate(row):
            M[j][i] = round(c, decPts)


def gj_Solve(A, b, decPts=4, epsilon=1.0e-16):
    # 检查A，b是否行数相同
    if len(A) != len(b):
        return None

    # 构造增广矩阵Ab
    Ab = augmentMatrix(A, b)

    for j in range(len(Ab)):
        dic = {}
        for i in range(len(Ab) - j):
            dic[i] = abs(Ab[i][j])

        # 寻找列c中 对角线以及对角线以下所有元素（行 c~N）的绝对值的最大值
        max_value_key = max(dic.items(), key=lambda x: x[1])[0]
        max_value = max(dic.items(), key=lambda x: x[1])[1]

        # 如果绝对值最大值为0, 那么A为奇异矩阵，返回None
        if max_value < epsilon:
            return None

        if (len(dic)) != 1:
            swapRows(Ab, j, max_value_key + j)

        # 当前列的对角线元素缩放为1
        Ab[j] = [x / Ab[j][j] for x in Ab[j]]

        # 当前列的其他元素消为0
        for i in range(j):
            addScaledRow(Ab, i, j, -1 * (Ab[i][j] / Ab[j][j]))

        for i in range(j + 1, len(Ab)):
            addScaledRow(Ab, i, j, -1 * (Ab[i][j] / Ab[j][j]))

    # 每个元素四舍五入到特定小数数位
    matxRound(Ab, decPts)

    return [[x[-1]] for x in Ab]


A = [
    [-7, -1, 3, -5, -5, -9, 6, 0],
    [2, -8, 1, 2, 4, -10, 4, 2],
    [8, -8, -3, -6, -10, 3, -8, 3],
    [-1, -2, -1, 2, -9, -6, -2, 4],
    [5, 7, 0, -7, -5, 5, 1, 5],
    [9, -4, 5, -1, 2, 3, 7, 6],
    [0, -10, -10, -9, 5, 6, -10, 1]]

b = [
    [1],
    [0],
    [2],
    [3],
    [4],
    [5],
    [6]
]

# {0: 2, 1: 6, 2: 8}
print(gj_Solve(A, b))
