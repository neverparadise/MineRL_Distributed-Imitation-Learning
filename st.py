import numpy


# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class SumTree:
    write = 0

    def __init__(self, capacity):

        """
        self.capacity : 원소의 총 개수
        self.tree : capacity * 2 - 1로 트리를 만듬. 왜? 인덱스가 2의배수로 늘어나기 때문
        self.data : 무엇을 저장하기 위한 변수인가? TD Error를 저장할 것 같음
        """
        self.capacity = capacity
        self.tree = numpy.zeros(2 * capacity - 1)
        self.data = numpy.zeros(capacity, dtype=object) # 제일 하위 트리의 값들.
        self.n_entries = 0 # 총 노드의 개수

    # update to the root node
    # 인덱스는 0부터 시작
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2 # 부모는 현재 트리의 인덱스 -1 // 2로 구할 수 있음.
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    # sum값이 어디에 있는지 탐색
    # 재귀형태로 구현
    def _retrieve(self, idx, s):
        # 현재 인덱스에서 자식의 왼쪽, 오른쪽 인덱스를 구한다.
        left = 2 * idx + 1
        right = left + 1

        # 왼쪽의 인덱스가 트리의 길이보다 길면 재귀적으로 현재의 인덱스가 마지막임.
        if left >= len(self.tree):
            return idx

        # 입력으로 들어온 s가 왼쪽 자식의 값보다 작은경우 한번더 왼쪽으로 가야한다.
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        # 왼쪽보다 크면 오른쪽으로 이동해야 한다.
        else:
            return self._retrieve(right, s - self.tree[left])

    # 제일 처음의 트리값은 트리의 노드들의 총합이다.
    def total(self):
        return self.tree[0]

    # store priority and sample
    # 트리에 data를 추가한다.
    def add(self, p, data):
        # 해당 데이터의 priority 업데이트를 위한 인덱스 계산
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    # 우선순위를 업데이트
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])