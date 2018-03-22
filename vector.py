import math
from decimal import Decimal, getcontext

getcontext().prec = 30


class Vector(object):
    CANNOT_NORMALIZE_ZERO_VECTOR_MSG = 'Cannot normalize the zero vector'

    def __init__(self, coordinates):
        try:
            if not coordinates:
                raise ValueError
            self.coordinates = tuple([Decimal(x) for x in coordinates])
            self.dimension = len(coordinates)
        except ValueError:
            raise ValueError('The coordinates must be nonempty')
        except TypeError:
            raise TypeError('The coordinates must be an iterable')

    def __str__(self):
        return 'Vector: {}'.format(self.coordinates)

    def __eq__(self, v):
        return self.coordinates == v.coordinates

    def plus(self, v):
        new_coordinates = [x + y for x, y in zip(self.coordinates, v.coordinates)]
        return Vector(new_coordinates)

    def minus(self, v):
        new_coordinates = [x - y for x, y in zip(self.coordinates, v.coordinates)]
        return Vector(new_coordinates)

    def times_scalar(self, c):
        new_coordinates = [x * Decimal(c) for x in self.coordinates]
        return Vector(new_coordinates)

    def magnitude(self):
        return math.sqrt(sum([x ** 2 for x in self.coordinates]))

    def normalize(self):
        magnitude = self.magnitude()
        try:
            return self.times_scalar(1.0 / magnitude)
        except ZeroDivisionError:
            raise Exception(self.CANNOT_NORMALIZE_ZERO_VECTOR_MSG)

    def dot_product(self, v):
        coordinates = self.coordinates
        return sum([x * y for x, y in zip(coordinates, v.coordinates)])

    def angle_with(self, v, in_degrees=False):
        try:
            u1 = self.normalize()
            u2 = v.normalize()
            angle_in_radians = math.acos(u1.dot_product(u2))
            if in_degrees:
                return angle_in_radians * 180. / math.pi
            else:
                return angle_in_radians
        except Exception as e:
            if str(e) == self.CANNOT_NORMALIZE_ZERO_VECTOR_MSG:
                raise Exception('Cannot compute an angle with the zero vector')
            else:
                raise e


# 加减和标量乘法
vector1 = Vector(['8.218', '-9.341'])
vector2 = Vector(['-1.129', '2.111'])
print(vector1.plus(vector2))

vector1 = Vector(['7.119', '8.215'])
vector2 = Vector(['-8.223', '0.878'])
print(vector1.minus(vector2))

vector1 = Vector(['1.671', '-1.012', '-0.318'])
s = 7.41
print(vector1.times_scalar(s))

# 编写大小和方向函数
print('--------------------------')
print('编写大小和方向函数')
vector = Vector(['-0.221', '7.437'])
print(vector.magnitude())
vector = Vector(['8.813', '-1.331', '-6.247'])
print(vector.magnitude())

vector = Vector(['5.581', '-2.136'])
print(vector.normalize())

print('--------------------------')
print('编写点积和夹角函数')
vector1 = Vector(['-5.955', '-4.904', '-1.874'])
vector2 = Vector(['-4.496', '-8.755', '7.103'])
print(vector1.dot_product(vector2))

vector1 = Vector(['3.183', '-7.627'])
vector2 = Vector(['-2.668', '5.319'])
print(vector1.angle_with(vector2))

vector1 = Vector(['7.35', '0.221', '5.188'])
vector2 = Vector(['2.751', '8.259', '3.985'])
print(vector1.angle_with(vector2, True))
#
# print(vector.vector_angle_degree(, ))
#
# print(vector.is_parallel([-7.579, -7.88], [22.737, 23.64]))
# print(vector.is_orthogonal([-7.579, -7.88], [22.737, 23.64]))
#
# print(vector.is_parallel([-2.029, 9.97, 4.172], [-9.231, -6.639, -7.245]))
# print(vector.is_orthogonal([-2.029, 9.97, 4.172], [-9.231, -6.639, -7.245]))
#
# print(vector.is_parallel([-2.328, -7.284, -1.214], [-1.821, 1.072, -2.94]))
# print(vector.is_orthogonal([-2.328, -7.284, -1.214], [-1.821, 1.072, -2.94]))

# print(vector.is_parallel([2.118, 4.827], [0, 0]))
# print(vector.is_orthogonal([2.118, 4.827], [0, 0]))
