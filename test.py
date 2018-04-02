import unittest
import numpy as np

from decimal import *


class LinearRegressionTestCase(unittest.TestCase):
    """Test for linear regression project"""

    def augmentMatrix(self,A, b):
        return [AA + bb for AA, bb in zip(A, b)]

    def swapRows(self, M, r1, r2):
        M[r1], M[r2] = M[r2], M[r1]

    def scaleRow(self, M, r, scale):
        if scale == 0:
            raise ValueError('scale不能是0')
        M[r] = [a * scale for a in M[r]]

    def addScaledRow(self,M, r1, r2, scale):
        M[r1] = [a + b * scale for a, b in zip(M[r1], M[r2])]

    def matxRound(self,M, decPts=4):
        for j, row in enumerate(M):
            for i, c in enumerate(row):
                M[j][i] = round(c, decPts)

    def gj_Solve(self,A, b, decPts=4, epsilon=1.0e-16):
        # 检查A，b是否行数相同
        if len(A) != len(b):
            return None

        # 构造增广矩阵Ab
        Ab = self.augmentMatrix(A, b)

        for j in range(len(Ab)):
            dic = {}
            for i in range(j, len(Ab)):
                dic[i] = abs(Ab[i][j])

            # 寻找列c中 对角线以及对角线以下所有元素（行 c~N）的绝对值的最大值
            max_value_key = max(dic.items(), key=lambda x: x[1])[0]
            max_value = max(dic.items(), key=lambda x: x[1])[1]

            # 如果绝对值最大值为0, 那么A为奇异矩阵，返回None
            if max_value < epsilon:
                return None

            if (len(dic)) != 1 and j != max_value_key:
                self.swapRows(Ab, j, max_value_key)

            # 当前列的对角线元素缩放为1
            Ab[j] = [x / Ab[j][j] for x in Ab[j]]

            # 当前列的其他元素消为0
            for i in range(j):
                self.addScaledRow(Ab, i, j, -1 * (Ab[i][j] / Ab[j][j]))

            for i in range(j + 1, len(Ab)):
                self.addScaledRow(Ab, i, j, -1 * (Ab[i][j] / Ab[j][j]))

        # 每个元素四舍五入到特定小数数位
        self.matxRound(Ab, decPts)

        return [[x[-1]] for x in Ab]

    def test_gj_Solve(self):

        for _ in range(9999):
            r = np.random.randint(low=3, high=10)
            A = np.random.randint(low=-10, high=10, size=(r, r))
            b = np.arange(r).reshape((r, 1))

            x = self.gj_Solve(A.tolist(), b.tolist(), epsilon=1.0e-8)

            if np.linalg.matrix_rank(A) < r:
                self.assertEqual(x, None, "Matrix A is singular")
            else:
                self.assertNotEqual(x, None, "Matrix A is not singular")
                self.assertEqual(np.array(x).shape, (r, 1),
                                 "Expected shape({},1), but got shape{}".format(r, np.array(x).shape))
                Ax = np.dot(A, np.array(x))
                loss = np.mean((Ax - b) ** 2)
                self.assertTrue(loss < 0.1, "Bad result.")


if __name__ == '__main__':
    unittest.main()
