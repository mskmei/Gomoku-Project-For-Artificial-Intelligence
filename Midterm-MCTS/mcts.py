import numpy as np
import random
from copy import deepcopy
import pisqpipe as pp
import random as rd
from collections import defaultdict as dd
import time

width = 20
height = 20

points = [1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e7]

FIVE = 6
FOUR = 5
SFOUR = 4
THREE = 3
STHREE = 2
TWO = 1
STWO = 0

DOUBLE_THREE = 902000
SFOUR_THREE = 903000
LIVE_FOUR = 905000

directions = {
    'hori': [(1, 0), (-1, 0)],
    'ver': [(0, 1), (0, -1)],
    '24': [(1, -1), (-1, 1)],
    '13': [(1, 1), (-1, -1)]
}

maxtimes = 200  # 最大模拟次数
maxtime = 6  # 最大单次时间
max_actions_num = 10  # 剪枝容忍度
c_param = 0.1  # explore-exploit trade-off

ini_board = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]


class Node:
    def __init__(self, parent, pre_action, state):
        self.parent = parent
        self.role = state.role
        self.win = 0
        self.times = 0
        self.successors = []

        self.actions = state.get_node_actions()  # 通过局部heuristic剪枝
        # self.actions = state.get_actions()  # 子节点为所有邻近点

        self.state = state
        self.preaction = pre_action

    def is_leaf(self):
        # 每个节点在还未被探索的时候都是叶子
        childs = list(self.successors)
        if len(childs) == 0:
            return True
        return False

    def is_root(self):
        if self.parent:
            return False
        return True

    def visit_times_global(self):
        if self.parent:
            return self.parent.visit_times_global()
        return self.times

    def N(self):
        if self.is_root():
            return self.times
        parent = self.parent
        return parent.visit_times_global()

    def Q(self):
        return self.win

    # 扩展子节点，即访问该节点的每个子节点
    def expand(self):
        action = self.actions.pop()
        current_state = deepcopy(self.state)
        next_state = current_state.move(action)
        child_node = Node(parent=self, pre_action=action, state=next_state)
        self.successors.append(child_node)
        return child_node

    # 生成树的逻辑是先判断该节点是否是最终状态，如果不是再判断是否完全扩展，如果不是则继续扩展该节点的子节点，否则，
    # 从子结点中随机选择一个作为下一个要扩展节点
    def _tree_policy(self):
        current_node = self
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

    def is_terminal_node(self):
        return self.state.is_over()

    def is_fully_expanded(self):
        return len(self.actions) == 0

    # 从所有的子节点（下一状态）中选择一个最优节点
    def best_child(self):
        current_state = self.state
        # UCT算法
        choices_weights = [(c.Q() / c.times) + c_param * np.sqrt((2 * np.log(self.N()) / c.times)) for c in
                           self.successors]
        role = current_state.role
        if role == 1:  # 如果当前玩家是先手，则选取权重最大的子节点作为最优action
            return self.successors[np.argmax(choices_weights)]
        else:  # 如果当前玩家是后手，则选取权重最小的子节点（先手胜率最低的状态）作为最优action
            return self.successors[np.argmin(choices_weights)]

    # 自我对弈模拟，从子结点中随机选取action
    def rollout(self, state):
        rollout_state = state
        while not rollout_state.is_over():
            possible_moves = rollout_state.get_actions()
            action = self.rollout_policy(possible_moves)
            rollout_state = rollout_state.move(action)
        return rollout_state.game_results()

    def rollout_policy(self, possible_moves):
        return possible_moves[np.random.randint(len(possible_moves))]  # rd.choice(possible_moves)

    # 向上回溯，将该节点的胜负信息传到父节点
    def backpropagate(self, result):
        self.times += 1
        if result == 1:
            self.win += 1
        if self.parent:
            self.parent.backpropagate(result)

    # 每个节点计算best action都要自我对弈（次数是maxtimes）并记录结果，从中选出最优的子节点
    def best_action(self):
        start = time.time()
        if len(self.state.my) == 0 and len(self.state.oppo) == 0:
            return [9, 9]
        for i in range(maxtimes):
            v = self._tree_policy()
            state = deepcopy(v.state)
            reward = v.rollout(state)
            v.backpropagate(reward)
            if time.time() - start > maxtime:
                break
        winning_rate = [c.win / c.times for c in self.successors]
        best = self.successors[np.argmax(winning_rate)]
        return best.preaction


class State:
    def __init__(self, board, role, my, oppo):
        self.board = board
        self.role = role
        self.my = my
        self.oppo = oppo

    def move(self, action):
        x, y = action
        self.board[x][y] = self.role
        if self.role == 1:
            return State(self.board, 3 - self.role, self.my + [(x, y)], self.oppo)
        else:
            return State(self.board, 3 - self.role, self.my, self.oppo + [(x, y)])

    def is_over(self):
        if self.if_winner(True):
            return True
        return False


    """
    模拟终止：必赢局面；节点终止：五连或活四。
    """

    # def if_winner(self, node):
    #     directions = [(1, 0), (0, 1), (1, 1), (-1, 1)]
    #     involved = [[[0 for d in range(4)] for i in range(20)] for j in range(20)]
    #     count = [[0 for pattern in range(7)] for r in range(2)]
    #     for x, y in self.my:
    #         for d in directions:
    #             if not involved[x][y][directions.index(d)]:
    #                 self.check_line(x, y, d, count[0], 1, involved)
    #     for x, y in self.oppo:
    #         for d in directions:
    #             if not involved[x][y][directions.index(d)]:
    #                 self.check_line(x, y, d, count[1], 2, involved)
    #     if count[0][FIVE] > 0:
    #         return 1
    #     if count[1][FIVE] > 0:
    #         return 2
    #     if count[0][FOUR] > 0:
    #         return 1
    #     if count[1][FOUR] > 0:
    #         return 2
    #     if not node:
    #         if count[0][SFOUR] > 1:
    #             return 1
    #         if count[1][SFOUR] > 1:
    #             return 2
    #         if count[0][SFOUR] > 0 and count[0][THREE] > 0:
    #             return 1
    #         if count[1][SFOUR] > 0 and count[1][THREE] > 0:
    #             return 2
    #         if count[0][THREE] > 1 and count[1][SFOUR] == 0:
    #             return 1
    #         if count[1][THREE] > 1 and count[0][SFOUR] == 0:
    #             return 2
    #     return False

    """
    模拟终止：活三；节点终止：五连
    """

    def five_judge(self, x, y, role):
        direction_list = list(directions.values())
        for dir in direction_list:
            count = 1
            live = 0
            for mukau in range(len(dir)):
                i = x + dir[mukau][0]
                j = y + dir[mukau][1]
                while True:
                    if i >= width or j >= height or i < 0 or j < 0 or self.board[i][j] == 3-role:
                        break
                    elif self.board[i][j]==0:
                        live+=1
                        break
                    else:
                        count += 1
                        i += dir[mukau][0]
                        j += dir[mukau][1]
            if count >= 5:
                return 5
            # if count >=3 and live==2:
            #     return 3
        return False
    
    def if_winner(self,node):
        for x,y in self.my:
            judge=self.five_judge(x,y,1)
            # if not node and judge>=5:
            #     return 1
            if judge>=5:
                return 1
        for x,y in self.oppo:
            judge=self.five_judge(x,y,2)
            # if not node and judge>=5:
            #     return 2
            if judge>=5:
                return 2
        return False

    def is_full(self):
        flag = True
        for i in range(width):
            for j in range(height):
                if self.board[i][j] == 0:
                    flag = False
        return flag

    def game_results(self):
        winner = self.if_winner(False)
        if not winner:
            if self.is_full():
                return "heikyo"
            return False
        else:
            return winner

    def get_actions(self):
        actions = []
        for x, y in self.my + self.oppo:
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if 0 <= x + i < width and 0 <= y + j < height and not (x + i, y + j) in actions and \
                            self.board[x + i][y + j] == 0:
                        actions.append((x + i, y + j))
        return actions

    def get_node_actions(self):
        """
        选择接下来可能的落子位置，选择的位置附近一定有棋子。
        若能够形成威胁棋形，则会仅返回阻止对方威胁或构成己方威胁的落子位置。
        对位置根据周围棋形评分进行排序，只取前max_action_num个。
        优先考虑进攻，因为这是该角色的回合。
        """
        max_actions = []
        m_five = []
        o_five = []
        m_live_four = []
        o_live_four = []
        m_sleep_four = []
        involved = []
        for x, y in self.my + self.oppo:
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if 0 <= x + i < width and 0 <= y + j < height and not (x + i, y + j) in involved and \
                            self.board[x + i][y + j] == 0:
                        involved.append((x + i, y + j))
                        m_score, o_score = self.point_score(x + i, y + j)
                        action = (int(max(m_score, o_score)), x + i, y + j)
                        if m_score >= points[FIVE]:
                            m_five.append(action)
                        if o_score >= points[FIVE]:
                            o_five.append(action)
                        if m_score >= points[FOUR]:
                            m_live_four.append(action)
                        if o_score >= points[FOUR]:
                            o_live_four.append(action)
                        if m_score >= points[SFOUR]:
                            m_sleep_four.append(action)
                        if not max_actions:
                            max_actions.append(action)
                        elif len(max_actions) < max_actions_num:
                            ok = 0
                            for k in range(len(max_actions)):
                                if max_actions[k][0] < action[0]:
                                    max_actions.insert(k, action)
                                    ok = 1
                                    break
                            if not ok:
                                max_actions.append(action)
                        else:
                            for k in range(max_actions_num):
                                if max_actions[k][0] < action[0]:
                                    max_actions.pop()
                                    max_actions.insert(k, action)
                                    break
        if len(m_five) > 0:  # 构成五连威胁，直接返回
            return [(m_five[0][1], m_five[0][2])]
        if len(o_five) > 0:  # 对手构成五连威胁
            return [(o_five[0][1], o_five[0][2])]
        if len(m_live_four) > 0:  # 构成活四威胁，直接返回，优先考虑进攻
            return [(c[1], c[2]) for c in m_live_four]
        if len(o_live_four) > 0:  # 对手构成活四威胁，只考虑眠四进攻或防守
            return [(c[1], c[2]) for c in m_sleep_four + o_live_four]
        return [(c[1], c[2]) for c in max_actions]

    def point_score(self, x, y):
        """
        对一个空格子进行局部启发式评估，分别假设双方在该处落子，统计以该子为中心四个方向能构成的棋形，返回双方的评分。
        """
        involved = [[[0 for d in range(4)] for i in range(20)] for j in range(20)]
        count = [0 for chess_shape in range(7)]
        directionlist = [(1, 0), (0, 1), (1, 1), (-1, 1)]

        self.board[x][y] = self.role
        for direction in directionlist:
            self.check_line(x, y, direction, count, self.role, involved)
        m_score = self.get_point_score(count)

        self.board[x][y] = 3 - self.role
        count = [0 for chess_shape in range(7)]
        for direction in directionlist:
            self.check_line(x, y, direction, count, 3 - self.role, involved)
        o_score = self.get_point_score(count)

        self.board[x][y] = 0
        return m_score, o_score

    def get_point_score(self, count):
        """
        对下一步动作候选(x,y)进行check_line更新count后，该函数通过count返回局部启发式评估。
        若产生威胁局面，会直接返回对应的高分。由于是对下一步动作的选择，在评估的时候是站在进攻的角度评估的。
        """
        score = 0
        if count[FIVE] > 0:  # 五连
            return points[FIVE]

        if count[FOUR] > 0:  # 活四杀棋
            return points[FOUR]

        if count[SFOUR] > 1:  # 双眠四杀棋
            return points[FOUR]
        elif count[SFOUR] > 0 and count[THREE] > 0:  # 眠四加活三杀棋
            return points[FOUR]

        if count[THREE] > 1:  # 双活三杀棋
            return points[FOUR] - 1
        else:
            score += count[THREE] * points[THREE]

        score += count[SFOUR] * points[SFOUR]
        score += count[STHREE] * points[STHREE]
        score += count[TWO] * points[TWO]
        score += count[STWO] * points[STWO]

        return score

    def make_line(self, x, y, direction, role):
        """
        该函数以(x,y)为中心，direction为方向，得到连线上的9个棋子。
        若超出边界，则默认为对方的棋子，因为无法落子在check_line中等同于对方棋子。
        """
        line = [0 for _ in range(9)]
        tempx = x - 5 * direction[0]
        tempy = y - 5 * direction[1]
        for i in range(9):
            tempx += direction[0]
            tempy += direction[1]
            if 0 <= tempx < width and 0 <= tempy < height:
                line[i] = self.board[tempx][tempy]
            else:
                line[i] = 3 - role
        return line

    def involve(self, x, y, left_m, right_m, direction, involved):
        """
        该函数用于标记已经算在其他棋形中的子，utility函数在统计时不会遍历已经算过的子。
        """
        if direction == (1, 0):
            index = 0
        elif direction == (0, 1):
            index = 1
        elif direction == (1, 1):
            index = 2
        else:
            index = 3
        tempx = x + (left_m - 5) * direction[0]
        tempy = y + (left_m - 5) * direction[1]
        for i in range(left_m, right_m + 1):
            tempx += direction[0]
            tempy += direction[1]
            if tempx >= 20 or tempx < 0 or tempy >= 20 or tempy < 0:
                continue
            else:
                involved[tempx][tempy][index] = 1

    def check_line(self, x, y, direction, count, role, involved):
        """
        该函数通过"left","right","left_edge"与"right_edge"对中心子（x,y）周围的情况分类，例如：
        role=1 角色为黑子时，
        白{黑空[黑(黑)]空}白，其中小括号内为中心子，中括号内为left和right之间的子，若不是黑子则停下；
        大括号为left_edge与right_edge之间的子，若碰到白子或超过范围则停下。
        界外子默认为对手的子。
        """
        left = 4
        right = 4
        line = self.make_line(x, y, direction, role)

        while left > 0 and line[left - 1] == role:
            left -= 1
        while right < 8 and line[right + 1] == role:
            right += 1
        left_edge = left
        right_edge = right
        while left_edge > 0 and line[left_edge - 1] != 3 - role:
            left_edge -= 1
        while right_edge < 8 and line[right_edge + 1] != 3 - role:
            right_edge += 1

        pattern_length = right_edge - left_edge + 1
        if pattern_length < 5:  # 该区间已经不可能五连
            self.involve(x, y, left_edge, right_edge, direction, involved)
            return

        self.involve(x, y, left, right, direction, involved)

        m_length = right - left + 1
        if m_length >= 5:  # 五连
            count[FIVE] += 1

        elif m_length == 4:
            if line[left - 1] == 0 and line[right + 1] == 0:  # 活四
                count[FOUR] += 1

            else:  # 眠四
                count[SFOUR] += 1

        elif m_length == 3:
            if line[left - 1] == 0 and line[left - 2] == role:  # 10111，眠四
                self.involve(x, y, left - 2, left - 2, direction, involved)  # 将被空格隔开的子标记，下同
                count[SFOUR] += 1

            elif line[right + 1] == 0 and line[right + 2] == role:  # 11101，眠四
                self.involve(x, y, right + 2, right + 2, direction, involved)
                count[SFOUR] += 1

            elif line[right + 1] == 0 and line[left - 1] == 0 and \
                    (line[right + 2] == 0 or line[left - 2] == 0):  # 011100，活三
                count[THREE] += 1

            else:  # 其他均为眠三
                count[STHREE] += 1

        elif m_length == 2:
            if line[left - 1] == 0 and line[left - 2] == role:
                self.involve(x, y, left - 2, left - 2, direction, involved)
                if line[left - 3] == 0 and line[right + 1] == 0:  # 010110，活三
                    count[THREE] += 1

                elif line[left - 3] == 0 and line[right + 1] == 3 - role:  # 01011，眠三
                    count[STHREE] += 1

                elif line[left - 3] == 3 - role:  # 10110，眠三
                    count[STHREE] += 1

                elif line[left - 3] == role:  # 11011，眠四，由于对称性，只需向左判断
                    self.involve(x, y, left - 3, left - 3, direction, involved)
                    count[SFOUR] += 1

            elif line[right + 1] == 0 and line[right + 2] == role:
                self.involve(x, y, right + 2, right + 2, direction, involved)
                if line[right + 3] == 0 and line[left - 1] == 0:
                    count[THREE] += 1

                elif line[right + 3] == 0 and line[left - 1] == 3 - role:
                    count[STHREE] += 1

                elif line[right + 3] == 3 - role:
                    count[STHREE] += 1

            elif line[right + 1] == 0 and line[right + 2] == 0 and line[right + 3] == role:
                self.involve(x, y, right + 3, right + 3, direction, involved)
                count[STHREE] += 1

            elif line[left - 1] == 0 and line[left - 2] == 0 and line[left - 3] == role:
                self.involve(x, y, left - 3, left - 3, direction, involved)
                count[STHREE] += 1

            else:
                if line[left] == 0 and line[right] == 0:  # 其余情况若两边有空格，属于活二，否则为眠二
                    count[TWO] += 1

                else:
                    count[STWO] += 1

        elif m_length == 1:
            if line[left - 1] == 0 and line[left - 2] == role \
                    and line[left - 3] == 0 and line[right + 1] == 3 - role:  # 0101，眠二
                self.involve(x, y, left - 2, left - 2, direction, involved)
                count[STWO] += 1

            elif line[right + 1] == 0 and line[right + 2] == role and line[right + 3] == 0:
                self.involve(x, y, right + 2, right + 2, direction, involved)
                if line[left - 1] == 3 - role:  # 1010，眠二
                    count[STWO] += 1

                else:  # 01010，活二
                    count[TWO] += 1

            elif line[right + 1] == 0 and line[right + 2] == 0 and line[right + 3] == role:
                self.involve(x, y, right + 3, right + 3, direction, involved)
                if line[left - 1] == 0 and line[right + 4] == 0:  # 010010，活二
                    count[TWO] += 1

                else:  # 10010，眠二
                    count[STWO] += 1

            elif line[right + 1] == 0 and line[right + 2] == 0 and line[right + 3] == 0 and line[right + 4] == role:
                self.involve(x, y, right + 4, right + 4, direction, involved)
                count[STWO] += 1

        return 0

# if __name__ == "__main__":
#     axis  = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
#     board = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#              [0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#              [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
#     my=[]
#     oppo=[]
#     for x in range(20):
#         for y in range(20):
#             if board[x][y] == 1:
#                 my.append((x, y))
#             if board[x][y] == 2:
#                 oppo.append((x, y))
#     state = State(board,1,my,oppo)
#     root = Node(None, None, state)
#     print(root.actions)
