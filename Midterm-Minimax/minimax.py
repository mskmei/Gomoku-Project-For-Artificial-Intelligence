import pisqpipe as pp
import time
import random
from pisqpipe import DEBUG_EVAL, DEBUG

pp.infotext = 'name="pbrain-alpha-beta",' \
              ' author="Keyu Mao, Jianzhi Shen",' \
              ' version="1.0", country="China",' \
              ' www="https://github.com/mskmei/Artificial-Intelligence"'

"""
initialization and parameters:
"""

MAX_BOARD = 100
board = [[0 for i in range(MAX_BOARD)] for j in range(MAX_BOARD)]
my = []
oppo = []

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

w = 0.7

max_depth = 4

max_time = 14

max_actions_num = 15

hash = 0

zobristHash = {}

ranTable = [[[random.randint(1, 2 ** 64 - 1) for i in range(2)] for j in range(20)] for k in range(20)]

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
        hash ^= ranTable[x][y][1]
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
    # try:
    #     if pp.terminateAI:
    #         return
    turn = 1
    # if len(my)==len(oppo):
    #     turn = 1
    x, y = search(board, pp.width, pp.height, turn)
    pp.do_mymove(x, y)
    # except:
    #     logTraceBack()


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
                    if m_score >= points[FOUR]:
                        m_live_four.append(action)
                    if o_score >= points[FOUR]:
                        o_live_four.append(action)
                    if self.count[role - 1][SFOUR] > 0:
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
        if len(m_live_four) > 0:  # 构成活四威胁，直接返回，优先考虑进攻
            return m_live_four
        if len(o_live_four) > 0:  # 对手构成活四威胁，只考虑眠四进攻或防守
            return o_live_four + m_sleep_four
        return max_actions

    def point_score(self, x, y, role):
        """
        对一个空格子进行局部启发式评估，分别假设双方在该处落子，统计以该子为中心四个方向能构成的棋形，返回双方的评分。
        """
        self.count = [[0 for chess_shape in range(7)] for role in range(2)]
        directions = [(1, 0), (0, 1), (1, 1), (-1, 1)]

        self.board[x][y] = role
        for direction in directions:
            self.check_line(x, y, direction, role, self.count[role - 1])
        m_score = self.get_point_score(role)

        self.board[x][y] = 3 - role
        self.count = [[0 for chess_shape in range(7)] for role in range(2)]
        for direction in directions:
            self.check_line(x, y, direction, 3 - role, self.count[2 - role])
        o_score = self.get_point_score(3 - role)

        self.board[x][y] = 0
        return m_score, o_score

    def get_point_score(self, role):
        """
        对下一步动作候选(x,y)进行check_line更新count后，该函数通过count返回局部启发式评估。
        若产生威胁局面，会直接返回对应的高分。由于是对下一步动作的选择，在评估的时候是站在进攻的角度评估的。
        """
        score = 0
        if self.count[role - 1][FIVE] > 0:  # 五连
            return points[FIVE]

        if self.count[role - 1][FOUR] > 0:  # 活四杀棋
            return points[FOUR]

        if self.count[role - 1][SFOUR] > 1:  # 双眠四杀棋
            return points[FOUR]-1
        elif self.count[role - 1][SFOUR] > 0 and self.count[role - 1][THREE] > 0:  # 眠四加眠三杀棋
            return points[FOUR]-2

        if self.count[role - 1][THREE] > 1:  # 双活三杀棋
            score += points[FOUR] - 3
        else:
            score += points[THREE]

        score+= self.count[role-1][SFOUR]*points[SFOUR]
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
        if m_length == 5:  # 五连
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
                    (line[right + 2] == 0 or line[left - 2] == 0):  # 011100，活三
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

                elif line[left - 3] == role:  # 11011，眠四，由于对称性，只需向左判断
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
                if line[left] == 0 and line[right] == 0:  # 其余情况若两边有空格，属于活二，否则为眠二
                    count[TWO] += 1

                else:
                    count[STWO] += 1

        elif m_length == 1:
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

    def get_score(self, role):
        """
        该函数通过check_line对count的更新，检测是否有威胁局面并返回双方统计评分。
        由于统计的是落子结果的utility，在考虑威胁局面的时候是以防守的角度考虑的。
        """
        m_score = 0
        o_score = 0

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

        if self.count[role - 1][THREE] > 0 and self.count[role-1][SFOUR]>0:  # 眠四加活三杀棋
            return SFOUR_THREE,0

        if self.count[role - 1][THREE] > 1 and self.count[2 - role][THREE] == 0:  # 双活三杀棋（需对手没有活三）
            return DOUBLE_THREE, 0
        if self.count[2 - role][THREE] > 0 and self.count[role - 1][SFOUR] == 0:  # 活三被杀
            return 0, DOUBLE_THREE - 10

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

    def utility(self, role):
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

        m_score, o_score = self.get_score(role)
        return w * m_score - o_score


"""
The minimax frame:
"""


def max_value(board, role, alpha, beta, depth, t):
    global hash
    if depth >= max_depth or time.time() - t >= max_time:  # 迭代加深的思路，如果时间到了，就直接返回当前的utility
        if hash in zobristHash:
            utility = zobristHash[hash]
        else:
            utility = board.utility(role)
            zobristHash[hash] = utility
        return utility, None
    v = float("-inf")

    action_list = board.get_actions(role)  # 获取自己可下的地方
    if len(action_list) != 0:  # 有地方可下
        if depth == 0 and len(action_list) == 1:
            return 0, action_list[0]
        action = None
        for a in action_list:
            board.board[a[1]][a[2]] = role
            board.place[role - 1].append((a[1], a[2]))
            hash ^= ranTable[a[1]][a[2]][role - 1]
            move_v, _ = min_value(board, role, alpha, beta, depth + 1, t)
            board.board[a[1]][a[2]] = 0
            board.place[role - 1].remove((a[1], a[2]))
            hash ^= ranTable[a[1]][a[2]][role - 1]
            if move_v >= 700000:
                return move_v, a  # 检测到必胜情况，直接返回
            if move_v > v:
                v = move_v
                action = a
            if v >= beta:
                return v, action
            alpha = max(alpha, v)
        return v, action

    else:  # 没有地方下
        v = board.utility()
        action = None
    return v, action


def min_value(board, role, alpha, beta, depth, t):
    global hash
    if depth >= max_depth or time.time() - t > max_time:
        if hash in zobristHash:
            utility = zobristHash[hash]
        else:
            utility = board.utility(role)
            zobristHash[hash] = utility
        return utility, None
    v = float("inf")

    action_list = board.get_actions(3 - role)
    if len(action_list) != 0:
        action = None
        for a in action_list:
            board.board[a[1]][a[2]] = 3 - role
            board.place[2 - role].append((a[1], a[2]))
            hash ^= ranTable[a[1]][a[2]][2 - role]
            move_v, _ = max_value(board, role, alpha, beta, depth + 1, t)
            board.board[a[1]][a[2]] = 0
            board.place[2 - role].remove((a[1], a[2]))
            hash ^= ranTable[a[1]][a[2]][2 - role]
            if move_v <= -900000:
                return move_v, a
            if move_v < v:
                v = move_v
                action = a
            if v <= alpha:
                return v, action
            beta = min(beta, v)
        return v, action

    else:
        v = board.utility()
        action = None
    return v, action


def search(current_board, width, height, turn):
    P = Board(current_board, width, height)
    alpha = float("-inf")
    beta = float("inf")
    depth = 0
    role = 1
    # if turn==1:
    _, action = max_value(P, role, alpha, beta, depth, time.time())
    # else:
    #     _, action = min_value(P, 3-role, alpha, beta, depth, time.time())
    return action[1], action[2]


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