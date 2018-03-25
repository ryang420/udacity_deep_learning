import math
from decimal import Decimal, getcontext

getcontext().prec = 32


class Vector(object):
    CANNOT_NORMALIZE_ZERO_VECTOR_MSG = 'Cannot normalize the zero vector'
    NO_UNIQUE_PARALLEL_COMPONENT_MSG = 'no unique parallel component'
    ONLY_DEFINED_IN_TWO_THREE_DIMS_MSG = 'only defined in two three dimensions vector'

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

    # 计算向量的大小(长度)
    def magnitude(self):
        return math.sqrt(sum([x ** 2 for x in self.coordinates]))

    # 计算向量标准化
    def normalized(self):
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
            u1 = self.normalized()
            u2 = v.normalized()
            dot_res = u1.dot_product(u2)
            if dot_res > 1:
                dot_res = Decimal('1')
            elif dot_res < -1:
                dot_res = Decimal('-1')
            angle_in_radians = math.acos(dot_res)
            if in_degrees:
                return angle_in_radians * 180. / math.pi
            else:
                return angle_in_radians
        except Exception as e:
            if str(e) == self.CANNOT_NORMALIZE_ZERO_VECTOR_MSG:
                raise Exception('Cannot compute an angle with the zero vector')
            else:
                raise e

    def is_parallel(self, v):
        return (self.is_zero()
                or v.is_zero()
                or self.angle_with(v) == 0
                or self.angle_with(v) == math.pi)

    def is_zero(self, tolerance=1e-10):
        return self.magnitude() < tolerance

    def is_orthogonal(self, v, tolerance=1e-10):
        return abs(self.dot_product(v)) < tolerance

    # 向量v在向量b上的投影向量
    def component_parallel_to(self, basis):
        try:
            u = basis.normalized()
            weight = self.dot_product(u)
            return u.times_scalar(weight)
        except Exception as e:
            if str(e) == self.CANNOT_NORMALIZE_ZERO_VECTOR_MSG:
                raise Exception(self.NO_UNIQUE_PARALLEL_COMPONENT_MSG)
            else:
                raise e

    # 正交向量
    def component_orthogonal_to(self, basis):
        try:
            projection = self.component_parallel_to(basis)
            return self.minus(projection)
        except Exception as e:
            if str(e) == self.NO_UNIQUE_PARALLEL_COMPONENT_MSG:
                raise Exception(self.NO_UNIQUE_PARALLEL_COMPONENT_MSG)
            else:
                raise e

    # 向量积
    def cross_product(self, v):
        try:
            x_1, y_1, z_1 = self.coordinates
            x_2, y_2, z_2 = v.coordinates
            new_coordinates = [
                y_1 * z_2 - y_2 * z_1,
                -(x_1 * z_2 - x_2 * z_1),
                x_1 * y_2 - y_1 * x_2
            ]
            return Vector(new_coordinates)
        except ValueError as e:
            msg = str(e)
            if msg == 'need more than 2 values to unpack':
                self_embedded_in_r3 = Vector(self.coordinates + ('0',))
                v_embedded_in__r3 = Vector(v.coordinates + ('0',))
                return self_embedded_in_r3.cross_product(v_embedded_in__r3)
            elif (msg == 'too many values to unpack'
                  or 'Need more than 1 value to unpack'):
                raise Exception(self.ONLY_DEFINED_IN_TWO_THREE_DIMS_MSG)
            else:
                raise e

    def area_of_parallelogram_with(self, v):
        cross_product = self.cross_product(v)
        return cross_product.magnitude()

    def are_of_triangle_with(self, v):
        return self.area_of_parallelogram_with(v) / Decimal('2.0')


'''
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
print(vector.normalized())


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


print('--------------------------')
print('检查是否平行或正交')
vector1 = Vector(['-7.579', '-7.88'])
vector2 = Vector(['22.737', '23.64'])
print(vector1.is_parallel(vector2))

vector1 = Vector(['-7.579', '-7.88'])
vector2 = Vector(['22.737', '23.64'])
print(vector1.is_orthogonal(vector2))

vector1 = Vector(['-2.328', '-7.284', '-1.214'])
vector2 = Vector(['-1.821', '1.072', '-2.94'])
print(vector1.is_parallel(vector2))
print(vector1.is_orthogonal(vector2))

vector1 = Vector(['2.118', '4.827'])
vector2 = Vector(['0', '0'])
print(vector1.is_parallel(vector2))
print(vector1.is_orthogonal(vector2))

print('--------------------------')
print('向量投影')
vector1 = Vector([3.039, 1.879])
vector2 = Vector([0.825, 2.036])
print('投影', vector1.component_parallel_to(vector2))

vector1 = Vector([-9.88, -3.264, -8.159])
vector2 = Vector([-2.155, -9.353, -9.473])
print('正交', vector1.component_orthogonal_to(vector2))

print('--------------------------')
print('向量积')
vector1 = Vector([-8.462, 7.893, -8.187])
vector2 = Vector([6.984, -5.975, 4.778])
print(vector1.cross_product(vector2))

vector1 = Vector([-8.987, -9.838, 5.031])
vector2 = Vector([-4.268, -1.861, -8.866])
print(vector1.area_of_parallelogram_with(vector2))
'''
