import math


class Vector(object):
    def __init__(self, coordinates):
        try:
            if not coordinates:
                raise ValueError
            self.coordinates = tuple(coordinates)
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
        new_coordinates = [x * c for x in self.coordinates]
        return new_coordinates


# 加减和标量乘法
vector1 = Vector([8.218, -9.341])
vector2 = Vector([-1.129, 2.111])
print(vector1.plus(vector2))

vector1 = Vector([7.119, 8.215])
vector2 = Vector([-8.223, 0.878])
print(vector1.minus(vector2))

vector1 = Vector([1.671, -1.012, -0.318])
s = 7.41
print(vector1.times_scalar(s))
# print(vector.vector_dot([7.887, 4.138], [-8.802, 6.776]))
#
# print(vector.vector_dot([-5.955, -4.904, -1.874], [-4.496, -8.755, 7.103]))
#
# print(vector.vector_anagle([3.183, -7.627], [-2.668, 5.319]))
#
# print(vector.vector_angle_degree([7.35, 0.221, 5.188], [2.751, 8.259, 3.985]))
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
