from collections import deque


class Edge:
    def __init__(self, v1, v2):
        self.v1, self.v2 = sorted([v1, v2])

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        else:
            return (self.v1, self.v2) == (other.v1, other.v2)

    def __hash__(self):
        return hash([self.v1, self.v2])


class Vertex:
    def __init__(self, label, neighbors):
        self.label = label
        self._neighbors = set(neighbors)

    def __contains__(self, item):
        if not isinstance(item, Vertex):
            return False
        else:
            return item in self.connected_vertices()

    def __hash__(self):
        return hash(self.label)

    def __eq__(self, other):
        return isinstance(other, Vertex) and self.label == other.label

    def __lt__(self, other):
        return isinstance(other, Vertex) and self.label < other.label

    def __le__(self, other):
        return self < other or self == other

    def __gt__(self, other):
        return not (self < other or self == other)

    def __ge__(self, other):
        return not (self < other)

    def neighboring_vertices(self):
        return set(self._neighbors)

    def neighboring_edges(self):
        return {Edge(self, v) for v in self._neighbors}

    def connected_vertices(self):
        return _gen_vertices_bf(start_list=[self])

    def connected_edges(self):
        return _gen_edges_bf(start_list=[self])

    def add_neighbor(self, v):
        self._neighbors.add(v)


class Graph:
    def __init__(self, vertices):
        self._vertices = _disjoint_vertices(vertices=vertices)

    def disjoint_vertices(self):
        return list(self._vertices)

    def vertices(self):
        return _gen_vertices_bf(start_list=self._vertices)

    def edges(self):
        return _gen_edges_bf(start_list=self._vertices)


def _bfs(start_list, fn, rsf, endfn=None):
    for v in _gen_vertices_bf(start_list=start_list):
        rsf = fn(v, rsf)
        if endfn is not None and endfn(rsf):
            break
    return rsf


def _gen_edges_bf(start_list):
    seen = deque()
    for v in _gen_vertices_bf(start_list):
        for e in v.neighboring_edges():
            if e not in seen:
                yield e
                seen.append(e)


def _gen_vertices_bf(start_list):
    return _gen_vertices(start_list, f=lambda x: x.popleft())


def _gen_vertices(start_list, f):
    seen = deque()
    tosee = deque(start_list)
    while len(tosee) > 0:
        v = f(tosee)
        yield v
        if v not in seen:
            tosee.extend(v.neighbors)
            seen.append(v)


def _disjoint_vertices(vertices):
    disjoint = []
    for v1 in vertices:
        minv = _bfs(start_list=[v1], fn=lambda v, rsf: min((v, rsf)), rsf=v1)
        if minv not in disjoint:
            disjoint.append(v1)
    return sorted(disjoint)


def make_graph(neighbors):
    pass


class UnitCell2D:
    def __init__(
            self,
            particle_labels,
            particle_positions,
            internal_connections,
            external_connections
    ):
        pass


class Lattice2D:
    def __init__(self):
        pass