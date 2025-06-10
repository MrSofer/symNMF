import sys


class Point:
    def __init__(self, position):
        self.position = position

    def distance(self, other):
        dist = 0
        for coordinate_1, coordinate_2 in zip(self.position, other.position):
            dist += (coordinate_1 - coordinate_2) ** 2
        return dist ** 0.5


class Centroid(Point):
    def __init__(self, position):
        super().__init__(position)
        self.next_position = [0 for _ in position]
        self.attached = 0
        self.attached_points = list()

    def push_position(self):
        if self.attached == 0:
            return 0
        self.next_position = [c / self.attached for c in self.next_position]
        delta = Point(self.position).distance(Point(self.next_position))
        self.position, self.next_position = self.next_position, [
            0 for _ in self.next_position]
        self.attached = 0
        self.attached_points = list()
        return delta

    def attach_point(self, point):
        self.attached += 1
        self.attached_points.append(point)
        for i in range(len(point.position)):
            self.next_position[i] += point.position[i]

    def __repr__(self):
        return ','.join([f'{coordinate:.4f}' for coordinate in self.position])


def parse(file_path, k):
    points = list()
    centroids = list()

    with open(file_path) as f:
        lines = f.read().splitlines()
    for idx, line in enumerate(lines):
        for n in line.split(','):
            try:
                _ = float(n)
            except ValueError:
                print("An Error Has Occurred")
                exit(1)
        if idx < k:
            centroids.append(Centroid([float(n) for n in line.split(',')]))
        points.append(Point([float(n) for n in line.split(',')]))

    return points, centroids

def parse_np(data, k):
    points = list()
    centroids = list()

    for idx, line in enumerate(data):
        if idx < k:
            centroids.append(Centroid([float(n) for n in line]))
        points.append(Point([float(n) for n in line]))

    return points, centroids

def get_closest_centroid(point, centroids):
    closest_centroid = centroids[0]
    centroid_distance = point.distance(closest_centroid)

    for centroid in centroids[1:]:
        current_distance = point.distance(centroid)
        if current_distance < centroid_distance:
            closest_centroid = centroid
            centroid_distance = current_distance
    return closest_centroid


def iterate(points, centroids):
    for point in points:
        closest = get_closest_centroid(point, centroids)
        closest.attach_point(point)

    delta = 0
    for centroid in centroids:
        delta += centroid.push_position()

    return delta


def main():
    if len(sys.argv) == 3:
        k = sys.argv[-2]
        iters = 200
        path = sys.argv[-1]
    elif len(sys.argv) == 4:
        k = sys.argv[-3]
        iters = sys.argv[-2]
        path = sys.argv[-1]
    else:
        print("An Error Has Occurred")
        exit(1)

    try:
        iters = int(iters)
        if not (1 < iters < 1000):
            raise ValueError()
    except ValueError:
        print("An Error Has Occurred")
        exit(1)

    try:
        k = int(k)
    except ValueError:
        print("An Error Has Occurred")
        exit(1)

    points, centroids = parse(path, k)

    if not (1 < k < len(points)):
        print("An Error Has Occurred")
        exit(1)

    epsilon = 0.001

    for _ in range(iters):
        delta = iterate(points, centroids)
        if delta < epsilon:
            break

    for centroid in centroids:
        print(centroid)


def fit(data, k, iters):
    points, centroids = parse_np(data, k)

    if not (1 < k < len(points)):
        print("An Error Has Occurred")
        exit(1)

    epsilon = 0.001

    for _ in range(iters):
        delta = iterate(points, centroids)
        if delta < epsilon:
            break
    
    labels = list()
    for p in points:
        c = get_closest_centroid(p, centroids)
        labels.append(centroids.index(c))
    
    return labels

        

if __name__ == '__main__':
    main()
