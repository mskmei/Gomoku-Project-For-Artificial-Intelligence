import pisqpipe as pp
import time
import random
import numpy as np
from pisqpipe import DEBUG_EVAL, DEBUG

pp.infotext = 'name="pbrain-alpha-beta",' \
              ' author="Keyu Mao, Jianzhi Shen",' \
              ' version="1.0", country="China",' \
              ' www="https://github.com/mskmei/Artificial-Intelligence"'

"""
initialization and parameters:
"""

MAX_BOARD = 20
board = np.zeros((MAX_BOARD, MAX_BOARD))
my = []
oppo = []

points = [1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e9]

FIVE = 6
FOUR = 5
SFOUR = 4
THREE = 3
STHREE = 2
TWO = 1
STWO = 0

DOUBLE_THREE = 8e5
SFOUR_THREE = 9e5
LIVE_FOUR = 1e6

w = 0.5

max_depth = 4

max_threat_depth = 4  # 后续算杀的最大层数

max_actions_num = 10

hash = 0

zobristHash = {}

winHash = {}

ranTable = [[[random.randint(1, 2 ** 64 - 1) for i in range(2)] for j in range(20)] for k in range(20)]

directions = {
    'hori': [(1, 0), (-1, 0)],
    'ver': [(0, 1), (0, -1)],
    '24': [(1, -1), (-1, 1)],
    '13': [(1, 1), (-1, -1)]
}

"""
pisqpipe interfaces:
"""


def brain_init():
    if pp.width < 5 or pp.height < 5:
        pp.pipeOut("ERROR size of the board")
        return
    if pp.width > MAX_BOARD or pp.height > MAX_BOARD:
        pp.pipeOut("ERROR Maximal board size is {}".format(MAX_BOARD))
        return
    pp.pipeOut("OK")


def brain_restart():
    global my, oppo, hash
    for x in range(pp.width):
        for y in range(pp.height):
            board[x][y] = 0
    my = []
    oppo = []
    hash = 0
    pp.pipeOut("OK")


def isFree(x, y):
    return 0 <= x < pp.width and 0 <= y < pp.height and board[x][y] == 0


def brain_my(x, y):
    global hash
    if isFree(x, y):
        board[x][y] = 1
        my.append((x, y))
        hash ^= ranTable[x][y][0]
    else:
        pp.pipeOut("ERROR my move [{},{}]".format(x, y))


def brain_opponents(x, y):
    global hash
    if isFree(x, y):
        board[x][y] = 2
        oppo.append((x, y))
        hash ^= ranTable[x][y][1]
    else:
        pp.pipeOut("ERROR opponents's move [{},{}]".format(x, y))


def brain_block(x, y):
    if isFree(x, y):
        board[x][y] = 3
    else:
        pp.pipeOut("ERROR winning move [{},{}]".format(x, y))


def brain_takeback(x, y):
    if 0 <= x < pp.width and 0 <= y < pp.height and board[x][y] != 0:
        board[x][y] = 0
        if (x, y) in my:
            my.remove((x, y))
        else:
            oppo.remove((x, y))
        return 0
    return 2


def brain_turn():
    global w

    # v, a = threat_search(board, 20, 20)
    # if v and a is not None:
    #     pp.do_mymove(a[1], a[2])
    #     return
    search_v, search_a = search(board, 20, 20)
    pp.do_mymove(search_a[1], search_a[2])
    return


def brain_end():
    pass


def brain_about():
    pp.pipeOut(pp.infotext)


"""
action and utility implementations:
"""


class Board:

    def __init__(self, board, width, height):
        self.board = board[0:width][0:height]
        self.width = width
        self.height = height
        self.place = [[] for role in range(2)]
        self.count = [[0 for pattern in range(7)] for role in range(2)]
        self.involved = [[[0, 0, 0, 0] for y in range(height)] for x in range(width)]
        for i in range(width):  # 储存每个选手已落子位置
            for j in range(height):
                if self.board[i][j] == 1:
                    self.place[0].append((i, j))
                elif self.board[i][j] == 2:
                    self.place[1].append((i, j))

    def adjacent(self, x, y):
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                elif not self.board[max(min(x + i, self.width - 1), 0)][max(min(y + j, self.height - 1), 0)] == 0:
                    return True
        return False

    def out_bound(self, x, y):
        return x < 0 or x >= self.width or y < 0 or y >= self.height

    def five_judge(self, x, y, role):
        direction_list = list(directions.values())
        for dir in direction_list:
            count = 1
            for mukau in range(len(dir)):
                i = x + dir[mukau][0]
                j = y + dir[mukau][1]
                while True:
                    if i >= self.width or j >= self.height or i < 0 or j < 0 or self.board[i][j] != role:
                        break
                    else:
                        count += 1
                        i += dir[mukau][0]
                        j += dir[mukau][1]
            if count >= 5:
                return True
        return False

    def if_winner(self):
        for x, y in self.place[0]:
            if self.five_judge(x, y, 1):
                return 1
        for x, y in self.place[1]:
            if self.five_judge(x, y, 2):
                return 2
        return False

    def get_actions(self, role):
        """
        选择接下来可能的落子位置，选择的位置附近一定有棋子。
        若能够形成威胁棋形，则会仅返回阻止对方威胁或构成己方威胁的落子位置。
        对位置根据周围棋形评分进行排序，只取前max_action_num个。
        优先考虑进攻，因为这是该角色的回合。
        """
        max_actions = [(0, 0, 0) for i in range(max_actions_num)]
        m_five = []
        o_five = []
        m_live_four = []
        o_live_four = []
        m_sfour_three = []
        o_sfour_three = []
        m_double_three = []
        o_double_three = []
        m_sleep_four = []
        if len(self.place[0]) == 0 and len(self.place[1]) == 0:
            return [(0, int(self.width / 2) - 1, int(self.height / 2) - 1)]
        for x in range(self.width):
            for y in range(self.height):
                if self.board[x][y] == 0 and self.adjacent(x, y):
                    m_score, o_score = self.point_score(x, y, role)
                    action = (int(max(m_score, o_score)), x, y)
                    if m_score >= points[FIVE]:
                        m_five.append(action)
                    if o_score >= points[FIVE]:
                        o_five.append(action)
                    if m_score == points[FOUR]:
                        m_live_four.append(action)
                    if o_score == points[FOUR]:
                        o_live_four.append(action)
                    if m_score == SFOUR_THREE:
                        m_sfour_three.append(action)
                    if o_score == SFOUR_THREE:
                        o_sfour_three.append(action)
                    if m_score == DOUBLE_THREE:
                        m_double_three.append(action)
                    if o_score == DOUBLE_THREE:
                        o_double_three.append(action)
                    if m_score >= points[SFOUR]:
                        m_sleep_four.append(action)
                    for i in range(max_actions_num):
                        if max_actions[i][0] < action[0]:
                            max_actions.pop()
                            max_actions.insert(i, action)
                            break
        if len(m_five) > 0:  # 构成五连威胁，直接返回
            return m_five
        if len(o_five) > 0:  # 对手构成五连威胁
            return o_five
        if len(m_live_four) or len(m_sfour_three) > 0:  # 构成活四威胁，直接返回，优先考虑进攻
            return m_live_four + m_sfour_three
        if len(o_live_four) or len(o_sfour_three) > 0:  # 对手构成活四威胁，只考虑眠四进攻或防守
            return o_live_four + o_sfour_three + m_sleep_four
        if len(m_double_three) > 0:  # 构成双活三威胁
            return m_double_three
        if len(o_double_three) > 0:  # 对手构成双活三威胁，只考虑眠四进攻或防守
            return o_double_three + m_sleep_four
        return max_actions

    def get_threat_actions(self, role):
        """
        选择构成威胁的棋子。
        优先考虑进攻，因为这是该角色的回合。
        """
        if self.if_winner():
            return []
        m_five = []
        o_five = []
        m_live_four = []
        o_live_four = []
        m_sfour_three = []
        o_sfour_three = []
        m_double_three = []
        o_double_three = []
        m_sleep_four = []
        o_sleep_four = []
        m_live_three = []
        o_live_three = []
        for x in range(self.width):
            for y in range(self.height):
                if self.board[x][y] == 0 and self.adjacent(x, y):
                    m_score, o_score = self.point_score(x, y, role)
                    action = (int(max(m_score, o_score)), x, y)
                    if m_score >= points[FIVE]:
                        m_five.append(action)
                    if o_score >= points[FIVE]:
                        o_five.append(action)
                    if m_score == points[FOUR]:
                        m_live_four.append(action)
                    if o_score == points[FOUR]:
                        o_live_four.append(action)
                    if m_score == SFOUR_THREE:
                        m_sfour_three.append(action)
                    if o_score == SFOUR_THREE:
                        o_sfour_three.append(action)
                    if m_score == DOUBLE_THREE:
                        m_double_three.append(action)
                    if o_score == DOUBLE_THREE:
                        o_double_three.append(action)
                    if DOUBLE_THREE > m_score >= points[SFOUR]:
                        m_sleep_four.append(action)
                    if points[SFOUR] > m_score >= points[THREE]:
                        m_live_three.append(action)
                    if points[SFOUR] > o_score >= points[THREE]:
                        o_live_three.append(action)
        if len(m_five) > 0:  # 构成五连威胁，直接返回
            return m_five
        if len(o_five) > 0:  # 对手构成五连威胁
            return o_five
        if len(m_live_four) or len(m_sfour_three) > 0:  # 构成活四威胁，直接返回，优先考虑进攻
            return m_live_four + m_sfour_three
        if len(o_live_four) or len(o_sfour_three) > 0:  # 对手构成活四威胁，只考虑眠四进攻或防守
            return o_live_four + o_sfour_three + m_sleep_four
        if len(m_double_three) > 0:  # 构成双活三威胁
            return m_double_three
        if len(o_double_three) > 0:  # 对手构成双活三威胁，只考虑眠四进攻或防守
            return o_double_three + m_sleep_four

        """
        1.在己方层考察活三或冲四进攻，在对方层考察对方的以攻为守或提前防守。
        """
        if role==1:
            return m_sleep_four + m_live_three
        if role==2:
            return o_sleep_four + o_live_three + m_sleep_four + m_live_three

        """
        2.在双方层均考察进攻（因为目前没有直接威胁）
        """
        # return m_sleep_four + m_live_three



    def point_score(self, x, y, role):
        """
        对一个空格子进行局部启发式评估，分别假设双方在该处落子，统计以该子为中心四个方向能构成的棋形，返回双方的评分。
        """
        self.count = [[0 for chess_shape in range(7)] for role in range(2)]
        directions = [(1, 0), (0, 1), (1, 1), (-1, 1)]

        self.board[x][y] = role
        for direction in directions:
            self.point_check_line(x, y, direction, role, self.count[role - 1])
        m_score = self.get_point_score(role)

        self.board[x][y] = 3 - role
        self.count = [[0 for chess_shape in range(7)] for role in range(2)]
        for direction in directions:
            self.point_check_line(x, y, direction, 3 - role, self.count[2 - role])
        o_score = self.get_point_score(3 - role)

        self.board[x][y] = 0
        return m_score, o_score

    def point_check_line(self, x, y, direction, role, count):
        """
        该函数与check_line原理一致，但需要额外考虑对称的pattern和一些正常对局不会出现的棋形，同时不需要考虑计入过的点
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
            return

        m_length = right - left + 1

        if m_length >= 5:  # 五连
            count[FIVE] += 1

        elif m_length == 4:
            if line[left - 1] == 0 and line[right + 1] == 0:  # 活四
                count[FOUR] += 1

            else:  # 眠四
                count[SFOUR] += 1

        elif m_length == 3:
            if line[left - 1] == 0 and line[left - 2] == role and line[left + 1] == 0 and line[
                left + 2] == role:  # 1011101，双眠四
                count[SFOUR] += 2

            elif line[left - 1] == 0 and line[left - 2] == role:  # 10111，眠四
                count[SFOUR] += 1

            elif line[right + 1] == 0 and line[right + 2] == role:  # 11101，眠四
                count[SFOUR] += 1

            elif line[right + 1] == 0 and line[left - 1] == 0 and \
                    (line[right + 2] == 0 or line[left - 2] == 0):  # 011100或001110，活三
                count[THREE] += 1

            else:  # 其他均为眠三
                count[STHREE] += 1

        elif m_length == 2:

            flag_big = 0

            flag_small = 0

            no = 0

            if line[left - 1] == 0:  # 先向左考虑

                if line[left - 2] == role and line[left - 3] == role:  # 11011，眠四
                    count[SFOUR] += 1

                elif line[left - 2] == role and line[left - 3] == 0 and line[right + 1] == 0:  # 010110，活三
                    flag_big += 1

                elif line[left - 2] == role and line[left - 3] == 0:  # 01011，眠三
                    count[STHREE] += 1

                elif line[left - 2] == role and line[left - 3] == 3 - role:  # 10110，眠三
                    flag_small += 1

                elif line[left - 2] == 0 and line[left - 3] == role:  # 10011，眠三
                    count[STHREE] += 1

                else:
                    no += 1

            if line[right + 1] == 0:  # 再向右考虑

                if line[right + 2] == role and line[right + 3] == role:
                    count[SFOUR] += 1

                elif line[right + 2] == role and line[right + 3] == 0 and line[left - 1] == 0:
                    flag_big += 1

                elif line[right + 2] == role and line[right + 3] == 0:
                    count[STHREE] += 1

                elif line[right + 2] == role and line[right + 3] == 3 - role:
                    flag_small += 1

                elif line[right + 2] == 0 and line[right + 3] == role:
                    count[STHREE] += 1

                else:
                    no += 1

            if no == 2:
                if line[right + 1] == 0 and line[left - 1] == 0:
                    count[TWO] += 1
                else:
                    count[STWO] += 1

            if flag_big == 2:
                count[THREE] += 1
                count[STHREE] += 1
            elif flag_big == 1:
                count[THREE] += 1
            elif flag_small > 0:
                count[STHREE] += 1

        elif m_length == 1:
            flag1 = 0
            flag2 = 0
            no = 0
            if line[left - 1] == 0:
                if line[left - 2] == role:
                    if line[left - 3] == role:
                        if line[left - 4] == role:
                            count[SFOUR] += 1
                        elif line[left - 4] == 0:
                            if line[right + 1] == 0:
                                flag1 += 1
                            else:
                                count[STHREE] += 1
                    elif line[left - 3] == 0:
                        if line[left - 4] == role:
                            count[STHREE] += 1
                        elif line[left - 4] == 0:
                            if line[right + 1] == 0:
                                flag2 += 1
                            else:
                                count[STWO] += 1
                elif line[left - 2] == 0:
                    if line[left - 3] == role:
                        if line[left - 4] == role:
                            count[STHREE] += 1
                        elif line[left - 4] == 0:
                            if line[right + 1] == 0:
                                flag2 += 1
                            else:
                                count[STWO] += 1
                    elif line[left - 3] == 0:
                        if line[left - 4] == role:
                            count[STWO] += 1
                else:
                    no += 1
            if line[right + 1] == 0:
                if line[right + 2] == role:
                    if line[right + 3] == role:
                        if line[right + 4] == role:
                            count[SFOUR] += 1
                        elif line[right + 4] == 0:
                            if line[left - 1] == 0:
                                flag1 += 1
                            else:
                                count[STHREE] += 1
                    elif line[right + 3] == 0:
                        if line[right + 4] == role:
                            count[STHREE] += 1
                        elif line[right + 4] == 0:
                            if line[left - 1] == 0:
                                flag2 += 1
                            else:
                                count[STWO] += 1
                elif line[right + 2] == 0:
                    if line[right + 3] == role:
                        if line[right + 4] == role:
                            count[STHREE] += 1
                        elif line[right + 4] == 0:
                            if line[left - 1] == 0:
                                flag2 += 1
                            else:
                                count[STWO] += 1
                    elif line[right + 3] == 0:
                        if line[right + 4] == role:
                            count[STWO] += 1
                else:
                    no += 1
            if no == 2:
                if line[left - 1] == 0 and line[left - 2] == role and line[right + 1] == 0 and line[right + 2] == role:
                    count[STHREE] += 1
                if line[left - 1] == 0 and line[right + 1] == 0 and line[right + 2] == role and line[right + 3] == role:
                    count[STHREE] += 1
                if line[right + 1] == 0 and line[left - 1] == 0 and line[left - 2] == role and line[left - 3] == role:
                    count[STHREE] += 1
            if flag1 > 0:
                count[THREE] += 1
            elif flag2 > 0:
                count[TWO] += 1
        return 0

    def get_point_score(self, role):
        """
        对下一步动作候选(x,y)进行check_line更新count后，该函数通过count返回局部启发式评估。
        若产生威胁局面，会直接返回对应的高分。由于是对下一步动作的选择，在评估的时候是站在进攻的角度评估的。
        """
        score = 0

        if self.count[role - 1][FIVE] > 0:  # 五连
            return points[FIVE]

        if self.count[role - 1][FOUR] > 0:  # 活四杀棋
            return LIVE_FOUR

        if self.count[role - 1][SFOUR] > 1:  # 双眠四杀棋
            return LIVE_FOUR

        if self.count[role - 1][SFOUR] > 0 and self.count[role - 1][THREE] > 0:  # 眠四加活三杀棋
            return SFOUR_THREE

        if self.count[role - 1][THREE] > 1:  # 双活三杀棋
            return DOUBLE_THREE

        score += self.count[role - 1][SFOUR] * points[SFOUR]
        score += self.count[role - 1][THREE] * points[THREE]
        score += self.count[role - 1][STHREE] * points[STHREE]
        score += self.count[role - 1][TWO] * points[TWO]
        score += self.count[role - 1][STWO] * points[STWO]

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
            if 0 <= tempx < self.width and 0 <= tempy < self.height:
                line[i] = self.board[tempx][tempy]
            else:
                line[i] = 3 - role
        return line

    def involve(self, x, y, left_m, right_m, direction):
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
            if self.out_bound(tempx, tempy):
                continue
            else:
                self.involved[tempx][tempy][index] = 1

    def check_line(self, x, y, direction, role, count):
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
            self.involve(x, y, left_edge, right_edge, direction)
            return

        self.involve(x, y, left, right, direction)

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
                self.involve(x, y, left - 2, left - 2, direction)  # 将被空格隔开的子标记，下同
                count[SFOUR] += 1

            elif line[right + 1] == 0 and line[right + 2] == role:  # 11101，眠四
                self.involve(x, y, right + 2, right + 2, direction)
                count[SFOUR] += 1

            elif line[right + 1] == 0 and line[left - 1] == 0 and \
                    (line[right + 2] == 0 or line[left - 2] == 0):  # 011100或001110，活三
                count[THREE] += 1

            else:  # 其他均为眠三
                count[STHREE] += 1

        elif m_length == 2:
            if line[left - 1] == 0 and line[left - 2] == role:

                self.involve(x, y, left - 2, left - 2, direction)

                if line[left - 3] == 0 and line[right + 1] == 0:  # 010110，活三
                    count[THREE] += 1

                elif line[left - 3] == 0 and line[right + 1] == 3 - role:  # 01011，眠三
                    count[STHREE] += 1

                elif line[left - 3] == 3 - role:  # 10110，眠三
                    count[STHREE] += 1

                elif line[left - 3] == role:  # 11011，眠四
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

                elif line[right + 3] == role:
                    self.involve(x, y, right + 3, right + 3, direction)
                    count[SFOUR] += 1

            elif line[right + 1] == 0 and line[right + 2] == 0 and line[right + 3] == role:  # 11001, 眠三
                self.involve(x, y, right + 3, right + 3, direction)
                count[STHREE] += 1

            elif line[left - 1] == 0 and line[left - 2] == 0 and line[left - 3] == role:
                self.involve(x, y, left - 3, left - 3, direction)
                count[STHREE] += 1

            else:
                if line[left] == 0 and line[right] == 0:  # 其余情况若两边有空格，属于活二，否则为眠二
                    count[TWO] += 1

                else:
                    count[STWO] += 1

        elif m_length == 1:
            if line[left - 1] == 0 and line[left - 2] == role:
                self.involve(x, y, left - 2, left - 2, direction)

            if line[left - 1] == 0 and line[left - 2] == role \
                    and line[left - 3] == 0 and line[right + 1] == 3 - role:  # 0101，眠二
                self.involve(x, y, left - 2, left - 2, direction)
                count[STWO] += 1

            elif line[right + 1] == 0 and line[right + 2] == role and line[right + 3] == 0:
                self.involve(x, y, right + 2, right + 2, direction)
                if line[left - 1] == 3 - role:  # 1010，眠二
                    count[STWO] += 1

                else:  # 01010，活二
                    count[TWO] += 1

            elif line[right + 1] == 0 and line[right + 2] == 0 and line[right + 3] == role:
                self.involve(x, y, right + 3, right + 3, direction)
                if line[left - 1] == 0 and line[right + 4] == 0:  # 010010，活二
                    count[TWO] += 1

                else:  # 10010，眠二
                    count[STWO] += 1

            elif line[right + 1] == 0 and line[right + 2] == 0 and line[right + 3] == 0 and line[right + 4] == role:
                self.involve(x, y, right + 4, right + 4, direction)
                count[STWO] += 1

        return 0

    def get_score(self, role, turn):
        """
        该函数通过check_line对count的更新，检测是否有威胁局面并返回双方统计评分。
        由于统计的是落子结果的utility，在考虑威胁局面的时候是以防守的角度考虑的。
        """

        m_score = 0
        o_score = 0

        if turn == 1:

            if self.count[role - 1][FIVE] > 0:
                return points[FIVE], 0
            if self.count[2 - role][FIVE] > 0:
                return 0, points[FIVE]

            if self.count[role - 1][SFOUR] >= 2:  # 两个眠四等价于一个活四
                self.count[role - 1][FOUR] += 1
            if self.count[2 - role][SFOUR] >= 2:
                self.count[2 - role][FOUR] += 1

            if self.count[2 - role][FOUR] > 0:  # 活四被杀
                return 0, LIVE_FOUR
            if self.count[2 - role][SFOUR] > 0:  # 眠四被杀
                return 0, LIVE_FOUR - 10

            if self.count[role - 1][FOUR] > 0:  # 活四杀棋
                return LIVE_FOUR - 20, 0

            if self.count[role - 1][THREE] > 0 and self.count[role - 1][SFOUR] > 0:  # 眠四加活三杀棋
                return SFOUR_THREE, 0

            if self.count[role - 1][THREE] > 1 and self.count[2 - role][THREE] == 0:  # 双活三杀棋（需对手没有活三）
                return DOUBLE_THREE, 0
            if self.count[2 - role][THREE] > 0 and self.count[role - 1][SFOUR] == 0:  # 活三被杀
                return 0, DOUBLE_THREE

            m_score += self.count[role - 1][SFOUR] * points[SFOUR]  # 若没有威胁局面，则正常统计
            m_score += self.count[role - 1][THREE] * points[THREE]
            m_score += self.count[role - 1][STHREE] * points[STHREE]
            m_score += self.count[role - 1][TWO] * points[TWO]
            m_score += self.count[role - 1][STWO] * points[STWO]
            o_score += self.count[2 - role][THREE] * points[THREE]
            o_score += self.count[2 - role][STHREE] * points[STHREE]
            o_score += self.count[2 - role][TWO] * points[TWO]
            o_score += self.count[2 - role][STWO] * points[STWO]

            return m_score, o_score

        if self.count[2 - role][FIVE] > 0:
            return 0, points[FIVE]
        if self.count[role - 1][FIVE] > 0:
            return points[FIVE], 0

        if self.count[role - 1][SFOUR] >= 2:  # 两个眠四等价于一个活四
            self.count[role - 1][FOUR] += 1
        if self.count[2 - role][SFOUR] >= 2:
            self.count[2 - role][FOUR] += 1

        if self.count[role - 1][FOUR] > 0:  # 活四被杀
            return LIVE_FOUR, 0
        if self.count[role - 1][SFOUR] > 0:  # 眠四被杀
            return LIVE_FOUR - 10, 0

        if self.count[2 - role][FOUR] > 0:  # 活四杀棋
            return 0, LIVE_FOUR - 20

        if self.count[2 - role][THREE] > 0 and self.count[2 - role][SFOUR] > 0:  # 眠四加活三杀棋
            return 0, SFOUR_THREE

        if self.count[2 - role][THREE] > 1 and self.count[role - 1][THREE] == 0:  # 双活三杀棋（需对手没有活三）
            return 0, DOUBLE_THREE
        if self.count[role - 1][THREE] > 0 and self.count[2 - role][SFOUR] == 0:  # 活三被杀
            return DOUBLE_THREE, 0

        m_score += self.count[role - 1][SFOUR] * points[SFOUR]  # 若没有威胁局面，则正常统计
        m_score += self.count[role - 1][THREE] * points[THREE]
        m_score += self.count[role - 1][STHREE] * points[STHREE]
        m_score += self.count[role - 1][TWO] * points[TWO]
        m_score += self.count[role - 1][STWO] * points[STWO]
        o_score += self.count[2 - role][THREE] * points[THREE]
        o_score += self.count[2 - role][STHREE] * points[STHREE]
        o_score += self.count[2 - role][TWO] * points[TWO]
        o_score += self.count[2 - role][STWO] * points[STWO]

        return m_score, o_score

    def utility(self, role, turn):
        """
        统计棋盘上所有棋形，并输出一个utility评估
        """
        self.count = [[0 for pattern in range(8)] for role in range(2)]
        self.involved = [[[0, 0, 0, 0] for y in range(self.height)] for x in range(self.width)]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for x, y in self.place[role - 1]:
            for i in range(4):
                if self.involved[x][y][i] == 0:
                    self.check_line(x, y, directions[i], role, self.count[role - 1])

        for x, y in self.place[2 - role]:
            for i in range(4):
                if self.involved[x][y][i] == 0:
                    self.check_line(x, y, directions[i], 3 - role, self.count[2 - role])

        m_score, o_score = self.get_score(role, turn)
        if m_score >= DOUBLE_THREE:
            return m_score
        if o_score >= DOUBLE_THREE:
            return -o_score
        if turn == 1:
            return w * m_score - o_score
        return m_score - w * o_score


"""
The minimax frame:
"""


def max_value(board, role, alpha, beta, depth):
    global hash
    if depth >= max_depth:
        threat_v, threat_a=threat_search(board.board,board.height,board.width)
        if threat_v:
            return points[FIVE], None
        if hash in zobristHash:
            utility = zobristHash[hash]
        else:
            utility = board.utility(role, 2)
            zobristHash[hash] = utility
        return utility, None
    v = float("-inf")

    action_list = board.get_actions(role)  # 获取自己可下的地方
    if len(action_list) != 0:  # 有地方可下
        if depth == 0 and len(action_list) == 1:
            return 0, action_list[0]
        action = None
        for i, a in enumerate(action_list):
            board.board[a[1]][a[2]] = role
            board.place[role - 1].append((a[1], a[2]))
            hash ^= ranTable[a[1]][a[2]][role - 1]

            if i == 0:
                move_v, _ = min_value(board, role, alpha, beta, depth + 1)
            else:
                move_v, _ = min_value(board, role, alpha, alpha + 1, depth + 1)  # pvc 算法
                if alpha < move_v < beta:
                    move_v, _ = min_value(board, role, move_v, beta, depth + 1)

            """
            用或不用PVC
            """
            # move_v, _ = min_value(board, role, alpha, beta, depth + 1)
            board.board[a[1]][a[2]] = 0
            board.place[role - 1].remove((a[1], a[2]))
            hash ^= ranTable[a[1]][a[2]][role - 1]
            if move_v >= DOUBLE_THREE:
                return move_v, a  # 检测到必胜情况，直接返回
            if move_v > v:
                v = move_v
                action = a
            if v >= beta:
                return v, action
            alpha = max(alpha, v)
        return v, action

    else:  # 没有地方下
        if hash in zobristHash:
            utility = zobristHash[hash]
        else:
            utility = board.utility(role, 2)
            zobristHash[hash] = utility
        return utility, None


def min_value(board, role, alpha, beta, depth):
    global hash
    if depth >= max_depth:
        if hash in zobristHash:
            utility = zobristHash[hash]
        else:
            utility = board.utility(role, 1)
            zobristHash[hash] = utility
        return utility, None
    v = float("inf")

    action_list = board.get_actions(3 - role)
    if len(action_list) != 0:
        action = None
        for i, a in enumerate(action_list):
            board.board[a[1]][a[2]] = 3 - role
            board.place[2 - role].append((a[1], a[2]))
            hash ^= ranTable[a[1]][a[2]][2 - role]
            if i == 0:
                move_v, _ = max_value(board, role, alpha, beta, depth + 1)
            else:
                move_v, _ = max_value(board, role, beta - 1, beta, depth + 1)
                if alpha < move_v < beta:
                    move_v, _ = max_value(board, role, alpha, move_v, depth + 1)
            # move_v, _ = max_value(board, role, alpha, beta, depth + 1)
            board.board[a[1]][a[2]] = 0
            board.place[2 - role].remove((a[1], a[2]))
            hash ^= ranTable[a[1]][a[2]][2 - role]
            if move_v <= -DOUBLE_THREE:
                return move_v, a
            if move_v < v:
                v = move_v
                action = a
            if v <= alpha:
                return v, action
            beta = min(beta, v)
        return v, action

    else:
        if hash in zobristHash:
            utility = zobristHash[hash]
        else:
            utility = board.utility(role, 1)
            zobristHash[hash] = utility
        return utility, None


def search(current_board, width, height):
    P = Board(current_board, width, height)
    alpha = float("-inf")
    beta = float("inf")
    depth = 0
    role = 1
    v, action = max_value(P, role, alpha, beta, depth)
    return v, action


"""
The TSS frame:
"""


def my_threat(board, role, depth, m_depth):
    global hash

    if hash in winHash:
        if winHash[hash]:
            return 1, None
    else:
        if board.if_winner() == 1:
            winHash[hash]=1
            return 1, None
        else:
            winHash[hash]=0
    if depth >= m_depth:
        return 0, None
    v = float("-inf")

    action_list = board.get_threat_actions(role)  # 获取自己构成威胁棋形的地方
    if len(action_list) != 0:  # 有地方可下

        action = None
        for a in action_list:
            board.board[a[1]][a[2]] = role
            board.place[role - 1].append((a[1], a[2]))
            hash ^= ranTable[a[1]][a[2]][role]
            move_v, _ = oppo_threat(board, role, depth + 1, m_depth)
            board.board[a[1]][a[2]] = 0
            board.place[role - 1].remove((a[1], a[2]))
            hash ^= ranTable[a[1]][a[2]][role]
            if move_v == 0:
                continue
            if move_v:
                return move_v, a  # 检测到必胜情况，直接返回
        return 0, action

    else:  # 没有地方下
        return 0, None


def oppo_threat(board, role, depth, m_depth):
    global hash

    if hash in winHash:
        if winHash[hash]:
            return 1, None
    else:
        if board.if_winner() == 1:
            winHash[hash] = 1
            return 1, None
        else:
            winHash[hash] = 0
    if depth >= m_depth:
        return 0, None
    v = float("inf")

    action_list = board.get_threat_actions(3 - role)
    if len(action_list) != 0:
        action = None
        for a in action_list:
            board.board[a[1]][a[2]] = 3 - role
            board.place[2 - role].append((a[1], a[2]))
            hash ^= ranTable[a[1]][a[2]][2 - role]
            move_v, _ = my_threat(board, role, depth + 1, m_depth)
            board.board[a[1]][a[2]] = 0
            board.place[2 - role].remove((a[1], a[2]))
            hash ^= ranTable[a[1]][a[2]][2 - role]
            if not move_v:
                return move_v, a
        return 1, action

    else:
        return 0, None


def threat_search(current_board,height,width):
    P = Board(current_board,height,width)
    role = 1
    best_action = None
    best_v = float("-inf")
    for d in range(1, max_threat_depth + 1):
        depth = 0
        v, action = my_threat(P, role, depth, d)
        if v >= DOUBLE_THREE:
            best_action = action
            best_v = v
            break
    return best_v, best_action


"""
for debug:
"""

# if DEBUG_EVAL:
#     import win32gui
#
#
#     def brain_eval(x, y):
#         # TODO check if it works as expected
#         wnd = win32gui.GetForegroundWindow()
#         dc = win32gui.GetDC(wnd)
#         rc = win32gui.GetClientRect(wnd)
#         c = str(board[x][y])
#         win32gui.ExtTextOut(dc, rc[2] - 15, 3, 0, None, c, ())
#         win32gui.ReleaseDC(wnd, dc)

######################################################################
# A possible way how to debug brains.
# To test it, just "uncomment" it (delete enclosing """)
######################################################################
# define a file for logging ...
# DEBUG_LOGFILE = "D:/pbrain-pyrandom-master/pbrain-pyrandom.log"
# ...and clear it initially
# with open(DEBUG_LOGFILE, "w") as f:
#     pass


# define a function for writing messages to the file


# define a function to get exception traceback
# def logTraceBack():
#     import traceback
#     with open(DEBUG_LOGFILE, "a") as f:
#         traceback.print_exc(file=f)
#         f.flush()
#     raise

######################################################################

# "overwrites" functions in pisqpipe module
pp.brain_init = brain_init
pp.brain_restart = brain_restart
pp.brain_my = brain_my
pp.brain_opponents = brain_opponents
pp.brain_block = brain_block
pp.brain_takeback = brain_takeback
pp.brain_turn = brain_turn
pp.brain_end = brain_end
pp.brain_about = brain_about


# if DEBUG_EVAL:
#     pp.brain_eval = brain_eval


def main():
    pp.main()


if __name__ == "__main__":
    main()
