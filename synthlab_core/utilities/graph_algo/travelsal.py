import numpy as np

"""
Method to return bounding box of foreground objs on mask
Input:
    mask: np.ndarray, binary mask of the image, True for foreground
Output:
    list of boxes: np.ndarray, [N, x1, y1, x2, y2] where N is the number of boxes
"""


def mask2boxes(mask: np.ndarray, limit=100):
    results = []

    xx, yy = mask.shape
    mark = np.zeros_like(mask, dtype=np.uint8)
    d4_x, d4_y = [0, 1, 0, -1, 1, 1, -1, -1], [1, 0, -1, 0, 1, -1, 1, -1]

    def inside(x, y):
        return 0 <= x < xx and 0 <= y < yy

    def adj(x, y):
        for i in range(8):
            x1, y1 = x + d4_x[i], y + d4_y[i]
            if inside(x1, y1):
                yield x1, y1, mask[x1, y1]

    def valid(x, y):
        if mask[x, y] == 0:
            return False

        if x + 1 == xx or y + 1 == yy:
            return True

        adjacent = list(adj(x, y))
        return any(x[2] == 0 for x in adjacent)

    def valid_box(x1, y1, x2, y2):
        return (x2 - x1) * (y2 - y1) >= limit

    for x in range(xx):
        for y in range(yy):
            if mark[x, y] or mask[x, y] == 0:
                continue

            que = [(x, y)]
            mark[x, y] = 1

            x1, y1, x2, y2 = x, y, x, y  # bottom left and upper right
            while que:
                x, y = que.pop()

                x1, y1 = min(x1, x), min(y1, y)
                x2, y2 = max(x2, x), max(y2, y)

                for x_next, y_next, val in adj(x, y):
                    if valid(x_next, y_next) and not mark[x_next, y_next]:
                        que.append((x_next, y_next))
                        mark[x_next, y_next] = 1

            if valid_box(y1, x1, y2, x2):
                results.append([y1, x1, y2, x2])

    if len(results) == 0:
        return None

    return np.array(results, dtype=np.int32)


def topo_order(adj):
    """
    Topological sort of a graph
    Input:
        adj: list of list, adj[i] is the list of nodes that i depends on
    Output:
        list of order: list of int, order[i] is the order of node i in the topological sort
    """
    V = len(adj)
    vorder = [-1 for i in range(V)]
    depends = [0 for i in range(V)]

    for i in range(V):
        for j in adj[i]:
            depends[j] += 1

    que = []
    order = 0

    for i in range(V):
        if depends[i] == 0:
            que.append(i)

    while que:
        u = que.pop(0)
        vorder[u] = order
        order = order + 1

        for v in adj[u]:
            depends[v] -= 1

            if depends[v] == 0:
                que.append(v)

    if order != V:
        return None

    return vorder
