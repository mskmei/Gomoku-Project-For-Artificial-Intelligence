import random
from copy import deepcopy
import numpy as np

ME = 0
OPPO = 1
max_actions_num = 400
epoch = 1000

"""
Features: no. of
* live four
* sleep four
* live three
* sleep three
* live two
* sleep two
of both sides
"""

weights = [1, 10, 100, 1000, 10000, 100000, 1e6, -1, -1e1, -1e2, -1e3, -1e4, -1e5, -1e6]
points = [1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e7]
FIVE = 6
FOUR = 5
SFOUR = 4
THREE = 3
STHREE = 2
TWO = 1
STWO = 0


class ENV:
    def __init__(self, board, height, width, role):
        self.board = board
        self.height = height
        self.width = width
        self.my = []
        self.oppo = []
        self.role = role

    def reset(self, role):
        """
        reset the board
        with role as the offensive
        """
        self.board = np.array([[0 for i in range(self.height)] for j in range(self.width)])
        self.my = []
        self.oppo = []
        self.role = role

    def is_full(self):
        flag = True
        for i in range(self.width):
            for j in range(self.height):
                if self.board[i][j] == 0:
                    flag = False
        return flag

    def is_end(self):
        """
        return True if the game is ended
        """
        # print(self.my)
        if len(self.my)==1 and self.my[0] == None:
            self.my = []
        if len(self.oppo)==1 and self.oppo[0] == None:
            self.oppo = []

        if self.my:
            for x, y in self.my:
                judge = self.five_judge(x, y, 1)
                if judge:
                    return 1
        if self.oppo:
            for x, y in self.oppo:
                judge = self.five_judge(x, y, 2)
                if judge:
                    return 2
        if self.is_full():
            return 3
        return False

    def five_judge(self, x, y, role):
        directions = {
            'hori': [(1, 0), (-1, 0)],
            'ver': [(0, 1), (0, -1)],
            '24': [(1, -1), (-1, 1)],
            '13': [(1, 1), (-1, -1)]
        }
        direction_list = list(directions.values())
        for dir in direction_list:
            count = 1
            for mukau in range(len(dir)):
                i = x + dir[mukau][0]
                j = y + dir[mukau][1]
                while True:
                    if i >= self.width or j >= self.height or i < 0 or j < 0 or self.board[i][j] == 0 or self.board[i][
                        j] == 3 - role:
                        break
                    else:
                        count += 1
                        i += dir[mukau][0]
                        j += dir[mukau][1]
            if count >= 5:
                return True
        return False

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

    # def get_actions(self, role):
    #     """
    #     选择接下来可能的落子位置，选择的位置附近一定有棋子。
    #     若能够形成威胁棋形，则会仅返回阻止对方威胁或构成己方威胁的落子位置。
    #     对位置根据周围棋形评分进行排序，只取前max_action_num个。
    #     优先考虑进攻，因为这是该角色的回合。
    #     """
    #     if len(self.my) == 0 and len(self.oppo) == 0:
    #         return [((int(self.width / 2) - 1, int(self.height / 2) - 1), 0)]
    #     actions = []
    #     for x in range(self.width):
    #         for y in range(self.height):
    #             if self.board[x][y] == 0 and self.adjacent(x, y):
    #                 self.make_move((x, y))
    #                 reward = -1
    #                 if self.is_end():
    #                     reward = 1e10
    #                 actions.append(((x, y), reward))
    #                 self.draw_move((x, y))
    #     return actions

    def features(self, role):
        """
        return the features
        """
        involved = [[[0 for d in range(4)] for i in range(20)] for j in range(20)]
        count = [[0 for pattern in range(7)] for r in range(2)]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        if len(self.my)==1 and self.my[0] == None:
            self.my = []
        if len(self.oppo)==1 and self.oppo[0] == None:
            self.oppo = []
        if self.my:
            for x, y in self.my:
                for direction in directions:
                    if not involved[x][y][directions.index(direction)]:
                        self.check_line(x, y, direction, count[ME], ME + 1, involved)
        if self.oppo:
            for x, y in self.oppo:
                for direction in directions:
                    if not involved[x][y][directions.index(direction)]:
                        self.check_line(x, y, direction, count[OPPO], OPPO + 1, involved)
        if role == ME:
            return count[ME] + count[OPPO]
        return count[OPPO] + count[ME]

    def make_move(self, action):
        """
        apply the change according to the action
        return reward
        """
        self.board[action] = self.role + 1
        if self.role == ME:
            self.my.append(action)
        else:
            self.oppo.append(action)
        self.role = 1 - self.role

    def draw_move(self, action):
        """
        reverse the effect of last action
        """
        self.board[action] = 0
        if self.role == ME:
            self.oppo.remove(action)
        else:
            self.my.remove(action)
        self.role = 1 - self.role

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
                line[i] = 1 - role
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

                elif line[left - 3] == role:  # 11011，眠四
                    self.involve(x, y, left - 3, left - 2, direction, involved)
                    count[SFOUR] += 1

            elif line[right + 1] == 0 and line[right + 2] == role:
                self.involve(x, y, right + 2, right + 2, direction, involved)
                if line[right + 3] == 0 and line[left - 1] == 0:
                    count[THREE] += 1

                elif line[right + 3] == 0 and line[left - 1] == 3 - role:
                    count[STHREE] += 1

                elif line[right + 3] == 3 - role:
                    count[STHREE] += 1

                elif line[right + 3] == role:
                    self.involve(x, y, right + 2, right + 3, direction, involved)
                    count[SFOUR] += 1

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

    def get_actions(self, role):
        """
        选择接下来可能的落子位置，选择的位置附近一定有棋子。
        若能够形成威胁棋形，则会仅返回阻止对方威胁或构成己方威胁的落子位置。
        对位置根据周围棋形评分进行排序，只取前max_action_num个。
        优先考虑进攻，因为这是该角色的回合。
        """
        if len(self.my) == 0 and len(self.oppo) == 0:
            return [(int(self.width / 2) - 1, int(self.height / 2) - 1)]

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
                    if 0 <= x + i < self.width and 0 <= y + j < self.height and not (x + i, y + j) in involved and \
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

        self.board[x][y] = self.role+1
        for direction in directionlist:
            self.check_line(x, y, direction, count, self.role+1, involved)
        m_score = self.get_point_score(count)

        self.board[x][y] = 2 - self.role
        count = [0 for chess_shape in range(7)]
        for direction in directionlist:
            self.check_line(x, y, direction, count, 2 - self.role, involved)
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


class ADP:
    def __init__(self, role, discount=0.9, learning_rate=0.5, eps=0.1):
        self.discount = discount
        self.learning_rate = learning_rate
        self.role = role
        self.eps = eps

    def get_Q(self, env, action):
        """
        return the estimated value
        """
        env.make_move(action)
        features = env.features(self.role)
        env.draw_move(action)
        value = 0
        for i in range(len(features)):
            value += features[i] * weights[i]
        return value

    def update(self, current_value, current_features, new_value):
        """
        update weights given a transition
        """
        for i in range(len(weights)):
            weights[i] += self.learning_rate * (new_value - current_value) * current_features[i]

    def eps_greedy(self, env):
        if random.random() < self.eps:
            actions = env.get_actions(self.role)
            random_choice = random.randint(0, len(actions) - 1)
            action, reward = actions[random_choice]
            value = self.get_Q(env, action)
            return actions[random_choice][0], value, reward
        best = None
        true_reward = 0
        Qstar = float("-inf")
        for action, reward in env.get_actions(self.role):
            value = self.get_Q(env, action)
            if value > Qstar:
                best = action
                Qstar = value
                true_reward = reward
        return best, Qstar, true_reward


if __name__ == '__main__':
    """
    train to get the fittest weights
    """
    height = 20
    width = 20
    board = np.array([[0 for i in range(height)] for j in range(width)])
    env = ENV(board, height, width, 0)
    my_agent = ADP(ME, 0.9, 0.5, 0.1)
    oppo_agent = ADP(OPPO, 0.9, 0.5, 0.1)
    for i in range(epoch):
        temp = deepcopy(weights)
        env.reset(np.random.randint(0, 2))
        my_last_feature = [0 for _ in range(14)]
        my_last_Q = 0
        my_last_reward = 0
        oppo_last_feature = [0 for _ in range(14)]
        oppo_last_Q = 0
        oppo_last_reward = 0
        while not env.is_end():
            print(weights)
            if env.role == ME:
                action, my_last_Q, my_last_reward = my_agent.eps_greedy(env)
                env.make_move(action)
                my_last_feature = env.features(ME)
                action_list = [oppo_agent.get_Q(env, a) for a, r in env.get_actions(OPPO)]
                oppo_new_Q = max(action_list)
                print(oppo_last_reward)
                print(env.board)
                print("feature", my_last_feature)
                oppo_agent.update(oppo_last_Q, oppo_last_feature, oppo_last_reward + oppo_agent.discount * oppo_new_Q)
            else:
                action, oppo_last_Q, oppo_last_reward = my_agent.eps_greedy(env)
                env.make_move(action)
                oppo_last_feature = env.features(OPPO)
                action_list = [my_agent.get_Q(env, a) for a, _ in env.get_actions(ME)]
                my_new_Q = max(action_list)
                my_agent.update(my_last_Q, my_last_feature, my_last_reward + my_agent.discount * my_new_Q)
        if env.is_end() == 1:
            my_agent.update(my_last_Q, my_last_feature, my_last_reward)
        elif env.is_end() == 2:
            oppo_agent.update(oppo_last_Q, oppo_last_feature, oppo_last_reward)
        print("OK")
        delta = 0
        for i in range(len(weights)):
            delta = max(delta, (weights[i] - temp[i]) ** 2)
        if delta < 1e-4:
            print("Over!", weights)
            break
