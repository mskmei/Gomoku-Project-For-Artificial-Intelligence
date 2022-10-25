from copy import deepcopy
import time

points = [0, 2, 8, 1e1, 1e2, 1e3, 1e4, 1e5]

FIVE = 7
FOUR = 6
SFOUR = 5
THREE = 4
STHREE = 3
TWO = 2
STWO = 1
ONE = 0

w = 1

tolerance = 10

max_depth = 5

max_actions_num=10


class Board:

    def __init__(self, board, width, height):
        self.board = board[0:width][0:height]
        self.width = width
        self.height = height
        self.count = [[0 for chess_shape in range(8)] for role in range(2)]
        self.involved = [[[0, 0, 0, 0] for x in range(width)] for y in range(height)]

    def adjacent(self,x,y):
        for i in range(-1,2):
            for j in range(-1,2):
                if i==0 and j==0:
                    pass
                if not self.board[x][y]==0:
                    return True
        return False

    def get_actions(self, role):

        actions=[]
        five=[]
        live_four=[]
        for x in range(self.width):
            for y in range(self.height):
                if self.board[x][y] == 0 and self.adjacent(x,y):
                    # x,y 可落子且附近有子
                    m_score,o_score=self.point_score(x,y,role)
                    action=(max(m_score,o_score),x,y)
                    if m_score>=points[FIVE] or o_score>=points[FIVE]:
                        five.append(action)
                    elif m_score>=points[FOUR] or o_score>=points[FOUR]:
                        live_four.append(action)
                    actions.append(action)
        if len(five)>0:
            return five
        if len(live_four)>0:
            return live_four
        actions.sort(reverse=True)
        if len(actions)>max_actions_num:
            actions=actions[0:max_actions_num]

        return actions

    def point_score(self, x, y, role):

        self.count = [[0 for chess_shape in range(8)] for role in range(2)]
        directions = [(1, 0), (0, 1), (1, -1), (-1, 1)]
        self.board[x][y] = role
        for direction in directions:
            self.check_line(x, y, direction, role, self.count[role - 1])
        m_score=self.get_point_score(role)
        self.board[x][y] = 3 - role
        self.count = [[0 for chess_shape in range(8)] for role in range(2)]
        for direction in directions:
            self.check_line(x, y, direction, 3 - role, self.count[2 - role])
        o_score=self.get_point_score(3-role)
        return m_score,o_score

    def get_point_score(self, role):
        score = 0
        if self.count[role - 1][FIVE] > 0:
            return points[FIVE], 0

        if self.count[role - 1][FOUR] >= 0:
            return points[FOUR]

        if self.count[role - 1][SFOUR] > 1:
            score += self.count[role - 1][SFOUR] * points[SFOUR]
        elif self.count[role - 1][SFOUR] > 0 and self.count[role - 1][THREE] > 0:
            score += self.count[role - 1][SFOUR] * points[SFOUR]
        elif self.count[role - 1][SFOUR] > 0:
            score += points[THREE]

        score += self.count[role - 1][THREE] * points[THREE]
        score += self.count[role - 1][STHREE] * points[STHREE]
        score += self.count[role - 1][TWO] * points[TWO]
        score += self.count[role - 1][STWO] * points[STWO]

        return score

    def make_line(self, x, y, direction, role):
        # 在某个方向上制造直线
        line = [0 for _ in range(9)]
        tempx = x - 5 * direction[0]
        tempy = y - 5 * direction[1]
        for i in range(9):
            tempx += direction[0]
            tempy += direction[1]
            if 0 <= tempx <= self.width and 0 <= tempy <= self.height:
                line[i] = self.board[tempx][tempy]
            else:
                # 如果越界直接设置为对方棋子，这在棋型判断上不影响
                line[i] = 3 - role
        return line

    def involve(self, x, y, left_m, right_m, direction):
        if direction == (1, 0):
            index = 0
        elif direction == (0, 1):
            index = 1
        elif direction == (1, -1):
            index = 2
        else:
            index = 3
        tempx = x + (left_m - 5) * direction[0]
        tempy = y + (left_m - 5) * direction[0]
        for i in range(left_m, right_m + 1):
            tempx += direction[0]
            tempy += direction[1]
            self.involved[tempx][tempy][index] = 1

    def check_line(self, x, y, direction, role, count):
        left = 4
        right = 4
        line = self.make_line(x, y, direction, role)

        # 判断直线上的同类棋子的数量
        while left > 0 and line[left - 1] == role:
            left -= 1
        while right < 8 and line[right + 1] == role:
            right += 1
        left_edge = left
        right_edge = right
        # 判断空子的数量
        while left_edge > 0 and line[left_edge - 1] != 3 - role:
            left_edge -= 1
        while right_edge < 8 and line[right_edge + 1] != 3 - role:
            right_edge += 1
        pattern_length = right_edge - left_edge + 1
        if pattern_length < 5:
            self.involve(x, y, left_edge, right_edge, direction)
            return
        self.involve(x, y, left, right, direction)
        m_length = right - left + 1
        if m_length == 5:
            count[FIVE] += 1
        elif m_length == 4:
            if line[left - 1] == 0 and line[right + 1] == 0:
                count[FOUR] += 1
            else:
                count[SFOUR] += 1
        elif m_length == 3:
            if line[left - 1] == 0 and line[left - 2] == role:
                self.involve(x, y, left - 2, left - 2, direction)
                count[SFOUR] += 1
            elif line[right + 1] == 0 and line[right + 2] == role:
                self.involve(x, y, right + 2, right + 2, direction)
                count[SFOUR] += 1
            elif line[right + 1] == 0 and line[left - 1] == 0 and (line[right + 2] == 0 or line[left - 2] == 0):
                count[THREE] += 1
            else:
                count[STHREE] += 1
        elif m_length == 2:
            if line[left - 1] == 0 and line[left - 2] == role:
                self.involve(x, y, left - 2, left - 2, direction)
                if line[left - 3] == 0 and line[right + 1] == 0:
                    count[THREE] += 1
                elif line[left - 3] == 0 and line[right + 1] == 3 - role:
                    count[STHREE] += 1
                elif line[left - 3] == 3 - role:
                    count[STHREE] += 1
                elif line[left - 3] == role:
                    self.involve(x, y, left - 3, left - 3, direction)
                    count[SFOUR] += 1
            elif line[right + 1] == 0 and line[right + 2] == role:
                self.involve(x, y, right + 2, right + 2, direction)
                if line[right + 3] == 0 and line[left - 1] == 0:
                    count[THREE] += 1
                elif line[right + 3] == 0 and line[left - 1] == 3 - role:
                    count[STHREE] += 1
                elif line[right + 3] == 3 - role:
                    count[STHREE] += 1
            elif line[right + 1] == 0 and line[right + 2] == 0 and line[right + 3] == role:
                self.involve(x, y, right + 3, right + 3, direction)
                count[STHREE] += 1
            elif line[left - 1] == 0 and line[left - 2] == 0 and line[left - 3] == role:
                self.involve(x, y, left - 3, left - 3, direction)
                count[STHREE] += 1
            else:
                if line[left] == 0 and line[right] == 0:
                    count[TWO] += 1
                else:
                    count[STWO] += 1
        elif m_length == 1:
            if line[left - 1] == 0 and line[left - 2] == role and line[left - 3] == 0 and line[right + 1] == 3 - role:
                self.involve(x, y, left - 2, left - 2, direction)
                count[STWO] += 1
            elif line[right + 1] == 0 and line[right + 2] == role and line[right + 3] == 0:
                self.involve(x, y, right + 2, right + 2, direction)
                if line[left - 1] == 3 - role:
                    count[STWO] += 1
                else:
                    count[TWO] += 1
            elif line[right + 1] == 0 and line[right + 2] == 0 and line[right + 3] == role and line[right + 4] == 0:
                self.involve(x, y, right + 3, right + 3, direction)
                count[TWO] += 1
        return 0

    def get_score(self,role):
        m_score=0
        o_score=0
        if self.count[role-1][FIVE]>0:
            return points[FIVE], 0
        if self.count[2-role][FIVE]>0:
            return 0,points[FIVE]
        if self.count[role-1][SFOUR]>=2:
            self.count[role-1][FOUR]+=1
        if self.count[2-role][SFOUR]>=2:
            self.count[2-role][FOUR]+=1

        if self.count[role-1][FOUR]>0:
            return 9050,0
        if self.count[role-1][SFOUR]>0:
            return 9040,0

        if self.count[2-role][FOUR]>0:
            return 0,9030
        if self.count[2-role][SFOUR]>0:
            return 0,9020

        if self.count[role-1][THREE]>0 and self.count[2-role][SFOUR]==0:
            return 9010,0
        if self.count[2-role][THREE]>1 and self.count[role-1][THREE]==0 and self.count[role-1][STHREE]==0:
            return 0,9000

        if self.count[2-role][SFOUR]>0:
            o_score+=400

        m_score+=

    def utility(self,role):


def max_value(board, role, alpha, beta, depth, t):

    if depth>=max_depth or (time.time() - t > 10):
        return board.utility(),None
    v = float("-inf")

    action_list = board.get_actions(role)  # 获取自己可下的地方
    if len(action_list) != 0:  # 有地方可下
        action = None
        for a in action_list:
            new_board = deepcopy(board)
            new_board.update(a[0],a[1],role)
            move_v, _ = min_value(new_board, role, alpha, beta, depth + 1)
            if move_v > v:
                v = move_v
                action = a
            if v >= beta: return v, action
            alpha = max(alpha, v)
        return v, action
            # ---------------------------------

    else:  # 没有地方下
        v = board.utility()
        action = None
    return v, action


def min_value(board, role, alpha, beta, depth, t):

    if depth>=max_depth or (time.time() - t > 10):
        return board.utility(),None
    v = float("inf")

    action_list = board.get_actions(3-role)  # 获取对手可下的地方
    if len(action_list) != 0:  # 有地方可下
        action = None
        for a in action_list:
            new_board = deepcopy(board)
            new_board.update(a[0],a[1],3-role)
            move_v, _ = min_value(new_board, role, alpha, beta, depth + 1)
            if move_v < v:
                v = move_v
                action = a
            if v <= alpha: return v, action
            beta = min(beta, v)
        return v, action
            # ---------------------------------

    else:  # 没有地方下
        v = board.utility()
        action = None
    return v, action

def search(board, t):
    P=Board(board)
    alpha=float("-inf")
    beta=float("inf")
    depth=0
    role=1
    _,action=max_value(board,role,alpha,beta,depth, t)
    return action
