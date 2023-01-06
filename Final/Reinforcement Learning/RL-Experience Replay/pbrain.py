"""
This is a Python gomoku AI agent using alpha-beta pruning, a final PJ for class AI, FDU-DataScience

Author: DWB, ZZJ

Date: 2020.12
"""
import os
import pickle
import numpy as np
import pisqpipe as pp
from pisqpipe import DEBUG_EVAL
from ADP_linear import *
from CriticalNetwork import CriticNetwork, ActionNetwork
from NeuralNetwork import *

w1 = np.array([[ 2.35256599e-01, -3.26641324e-01, -5.56510077e-01,
         5.87478039e-01, -8.33381176e-03,  5.01187003e-01,
        -7.63657130e-02, -1.88420810e-01,  3.32839393e-02,
         2.90393865e-01, -6.25838422e-01,  1.91187767e-01,
         1.13711891e-01, -2.00611156e-02, -6.36103669e-01,
        -2.33794650e-01,  1.89235572e-01,  2.31087356e-01,
         6.04300895e-02, -1.62978984e-01,  3.17433614e-01,
         1.67096721e-01, -1.04038945e-01,  4.90916798e-01,
        -6.76658255e-02,  8.05722229e-02, -8.27856477e-02,
        -1.26536394e-02, -1.28011927e-01,  2.33132602e-02],
       [ 4.01009153e-01, -2.29530647e-01, -3.33998806e-01,
         2.36808766e-01,  1.68315211e-01, -3.32626034e-02,
        -4.56793208e-01, -1.82154188e-01, -3.47234960e-01,
        -3.75630927e-02, -1.88214015e-01,  4.45620471e-01,
        -1.06743426e-01,  4.98732765e-01,  5.05870301e-02,
        -2.99271105e-01,  1.43220671e-01, -1.40541112e-01,
        -4.00662380e-01, -3.22703960e-01,  3.63025639e-02,
        -5.69080616e-02, -3.33056594e-02,  3.58873856e-01,
        -2.36747498e-01, -1.12783140e-01, -7.28081499e-02,
         5.98622450e-02, -1.75521788e-01, -2.72126427e-01],
       [ 4.80830523e-01, -2.93541482e-01, -3.10133160e-01,
        -1.66735828e-01, -4.93070940e-01,  1.09859034e-01,
        -2.31438594e-01, -5.36813775e-02, -1.43141858e-01,
        -1.17091818e-01, -5.24001341e-01, -1.42998466e-01,
        -1.21089028e-01, -3.35711734e-01, -2.66451004e-01,
         1.99510289e-01, -2.92246661e-01,  2.23389136e-01,
         2.80860327e-01,  2.74219948e-01,  3.05195592e-01,
         3.23644444e-01,  4.89892089e-02,  4.29274763e-01,
         2.83255113e-01,  2.04430150e-01, -5.16702474e-01,
        -6.31982186e-02, -1.30641226e-01,  4.07376808e-01],
       [-4.88197283e-01,  5.37392740e-01,  5.56796635e-01,
        -4.27565333e-01, -2.46748178e-02, -7.47972316e-02,
        -3.16695183e-02, -5.93246009e-01, -3.79739813e-01,
         1.03859213e-01,  6.49472695e-01, -1.10185395e-02,
         4.84947830e-02, -3.59803876e-01, -5.06084065e-02,
        -7.43466482e-02, -1.94756465e-01, -2.35715724e-01,
         1.28572708e-01, -8.40766066e-01, -8.56909316e-02,
         2.46845607e-01,  7.79750565e-01, -5.12967193e-01,
        -2.14841114e-01,  1.89307853e-01,  6.15187644e-01,
        -1.82497606e-01, -7.20162728e-02,  9.34819004e-02],
       [-9.64460760e-03,  3.07874506e-01, -3.00051945e-01,
        -3.26354359e-01,  2.76028546e-01, -2.66552494e-02,
         4.70745598e-01,  5.71176732e-02, -2.69413212e-01,
         1.57272694e-01, -5.98021870e-02,  5.58791527e-02,
        -4.10636882e-01, -7.40650246e-02, -5.20165069e-02,
         2.08379879e-01,  4.49214301e-01, -2.29319998e-01,
         1.31365064e-01, -4.57823321e-01,  1.13385686e-01,
         7.08454898e-02,  4.18395609e-01, -2.14262634e-01,
         3.42978095e-01, -1.79651108e-01,  2.77530888e-01,
         2.57552098e-01,  2.11341273e-01,  1.70983662e-01],
       [-4.26952704e-01,  2.85097243e-01,  4.75781646e-01,
         9.59021241e-02,  3.49597272e-01, -3.82951757e-01,
         5.55339213e-01, -8.52657442e-01,  3.29984233e-01,
        -2.62517657e-01,  1.27546253e-01,  6.51489786e-02,
         4.53985952e-01,  3.42298989e-01, -4.29891702e-02,
        -4.96428944e-01,  1.75581588e-01, -4.71120657e-01,
         3.99571151e-01, -1.01876578e-01, -4.62254899e-01,
        -2.66260460e-01,  7.22921684e-01, -4.66942482e-01,
        -3.10912308e-01, -2.76391644e-01,  9.15436729e-02,
        -4.96733250e-01,  6.04891663e-01, -1.93501168e-01],
       [ 6.97915068e-01, -4.09667087e-01, -6.51854441e-01,
         1.33901219e-01, -3.55118594e-01,  1.38086161e-01,
        -3.59712467e-01,  8.38949125e-01, -6.02533698e-02,
        -3.35962907e-01, -4.35543079e-01,  7.35908006e-02,
        -3.77953205e-01,  6.83277816e-03, -5.04686723e-01,
         1.02278178e-01, -4.89735766e-01, -3.49833898e-01,
        -7.35169056e-01,  5.74729435e-01,  2.06096307e-01,
         3.07495774e-01,  2.02598033e-01,  7.77858184e-03,
         1.13339173e-01,  3.80278012e-01,  1.66734992e-01,
         2.35537772e-01, -2.88326100e-01,  1.09407808e-01],
       [ 2.81126728e-01, -2.49335677e-01, -1.21368789e-01,
        -2.73440654e-01,  2.58389360e-01,  5.04532805e-01,
        -4.56228617e-01,  6.04635023e-01, -2.19201514e-01,
         4.67541937e-01,  2.43317529e-02,  7.97015847e-02,
        -2.30466043e-02, -1.97962612e-01, -2.85561866e-01,
         1.31499223e-01,  1.51304446e-01,  4.76008635e-02,
        -3.08712885e-01,  2.41172245e-01,  4.70811201e-02,
         5.56108170e-02, -2.03069865e-01,  2.24532823e-02,
         2.27582597e-01,  4.18191486e-01, -6.09048513e-01,
         5.60759369e-01, -6.09757762e-01,  5.98439336e-01],
       [-2.42138397e-01,  3.04281248e-01, -2.12592538e-02,
        -2.77681138e-01,  1.20568422e-01, -5.86500228e-01,
         3.77385538e-01, -2.24589750e-01, -3.47010188e-01,
        -2.98889958e-02,  4.19912717e-02, -1.01277655e-01,
        -2.86302703e-02,  2.65735664e-01,  2.10284701e-01,
        -3.04596941e-01,  4.39105651e-01,  3.56878206e-01,
        -8.45063730e-02,  1.52091201e-02,  1.86687988e-01,
        -1.02530670e-01,  3.85795761e-01, -1.35830390e-01,
         4.01933904e-01, -1.47339179e-01,  6.96631563e-01,
        -3.69290098e-01,  1.31978448e-01,  2.46335190e-01],
       [-9.29250237e-02,  3.53917900e-01,  1.73886853e-01,
        -7.31272466e-01,  1.97867639e-01, -4.05345228e-01,
         5.34323003e-01, -6.55318921e-01, -2.27427636e-01,
        -6.87822400e-02,  1.97937954e-01, -2.25845251e-01,
        -4.24495515e-01,  1.94645607e-01, -1.26397621e-01,
        -1.22830299e-02, -1.46404193e-01, -4.02075308e-01,
         2.10921367e-01, -1.36387611e-01,  4.35520496e-01,
        -2.07195710e-01,  6.72321898e-01, -3.92760199e-01,
         3.46255503e-01, -4.82102554e-01,  7.39502705e-01,
        -4.92612088e-02, -1.55825101e-01, -4.20502386e-01],
       [ 5.61596485e-01, -5.62154140e-01,  2.33125895e-02,
         7.15120831e-01,  2.46636885e-01, -6.02924462e-02,
        -7.53321035e-01, -2.17966336e-01,  1.46467607e-01,
        -2.85254415e-01, -5.51500380e-01,  6.34124056e-01,
         1.21377824e-01,  8.12464845e-02,  1.24618966e-02,
         4.35816533e-01,  1.25389924e-01, -6.26446113e-03,
         1.58356597e-02,  5.63830178e-01, -4.43190911e-01,
        -4.14806992e-01,  1.52227867e-01, -9.07345896e-02,
         5.05390957e-02, -1.34442785e-01, -5.08518514e-01,
         3.50441119e-03, -6.42580580e-01, -5.38046806e-02],
       [-2.08516047e-01,  3.31778300e-01,  1.92344778e-01,
         3.15838333e-01,  1.40572526e-02,  3.36458557e-01,
        -2.57513723e-01, -1.93948965e-02,  8.06226062e-02,
        -3.60712112e-01,  5.54276541e-01, -1.68840723e-01,
         6.41759803e-02,  9.58753870e-02,  5.14009546e-01,
        -5.41203967e-01, -1.94600879e-01,  3.13806106e-01,
         2.12660288e-01, -3.96047556e-01,  5.00077197e-01,
         4.53524800e-01,  6.20022209e-01, -3.05910174e-01,
         3.13830239e-02, -2.64142002e-01, -1.79660542e-01,
        -1.40320566e-01, -4.01272684e-02, -6.32808143e-01],
       [ 4.16826244e-01, -2.96594023e-01, -1.97128981e-01,
        -2.01887133e-01, -5.21300363e-01,  4.57643853e-01,
        -3.52472722e-01,  4.99291513e-01, -1.44925859e-01,
        -2.62181724e-01, -6.74068870e-01,  5.50003313e-01,
        -2.58980316e-01,  2.64471716e-03, -6.43485452e-01,
         5.03707646e-01, -1.47919772e-01,  4.02076742e-01,
        -3.31983671e-01,  4.87151976e-01,  1.11327276e-01,
         3.01886582e-01, -3.01288461e-01,  5.84860536e-01,
        -1.84217359e-01,  3.66530974e-01, -3.17183130e-01,
         4.18525793e-01, -1.75239480e-01, -5.52018994e-02],
       [ 4.68065560e-01, -4.60668306e-01, -5.85119151e-01,
         7.31304260e-01,  4.93507626e-02,  4.17218068e-01,
        -5.84988008e-01,  6.59039766e-01, -5.37788167e-02,
         2.94861988e-01, -1.18981254e-01, -5.90480417e-02,
        -5.09281744e-02,  1.46241700e-01, -5.83078761e-01,
         2.34701910e-01, -4.58978401e-01, -4.86388510e-01,
         2.81945630e-01, -2.09359916e-01,  2.38151073e-01,
         5.90568323e-02,  1.74889125e-01,  4.36283563e-01,
         1.66260557e-01,  3.24786485e-02, -4.46426513e-01,
        -1.85805743e-01, -7.03120180e-01,  4.24638713e-01],
       [ 5.91118487e-01, -5.95422636e-01, -4.19675197e-02,
         4.22623547e-01, -1.06579809e-01, -2.85812965e-01,
         1.31200612e-01,  5.98892997e-01, -1.72660298e-01,
         2.86366166e-01, -7.62567983e-01,  6.89466959e-01,
        -7.65909786e-02, -6.66236818e-02, -5.80026651e-01,
         3.71742733e-01,  7.01852351e-02,  3.42635302e-05,
        -3.29045472e-01,  5.16455775e-01, -4.38279555e-01,
         3.08253951e-01, -4.73558331e-01, -1.34255998e-01,
         9.57442040e-02, -1.61712263e-01, -1.22201540e-01,
        -2.13209162e-01, -5.11350927e-02,  4.98427316e-01],
       [-2.04007604e-01, -3.40899686e-01,  3.94494240e-01,
        -1.44454665e-01, -3.73692499e-01, -4.20751915e-01,
        -4.72394921e-01, -6.61206983e-02, -1.02512660e-01,
         3.00189761e-01,  1.77568514e-01,  5.60731734e-01,
        -2.11695694e-01,  3.47668362e-01, -4.52654377e-01,
         2.39924174e-02,  1.09439391e-01,  3.64609973e-01,
        -2.26986552e-01,  9.66811863e-02, -1.45322777e-01,
         7.05231588e-02,  8.74452744e-02,  2.97120387e-01,
         1.53876484e-01, -3.45240467e-01, -3.41754550e-01,
         4.52980354e-01, -3.98346205e-01,  1.98890264e-01]])

b1 = np.array([[ 0.07702417],
       [-0.00549476],
       [ 0.02493517],
       [ 0.01255032],
       [ 0.0007357 ],
       [-0.08691322],
       [ 0.06195102],
       [-0.02272155],
       [ 0.02156512],
       [ 0.06951428],
       [ 0.02530123],
       [ 0.01778077],
       [-0.00313016],
       [ 0.03868342],
       [-0.08186005],
       [ 0.00524765]])

w2 = np.array([[-0.95410892, -0.49115897, -0.89979883,  2.40374874,  0.68348946,
         2.55004264, -1.88119123, -1.33700257,  1.28931455,  1.9889636 ,
        -1.4818944 ,  1.23112225, -2.37857184, -1.65239514, -1.86338896,
        -0.50818887]])

b2 = np.array([[0.97469278]])



# pp.infotext = infotext
MAX_BOARD = 100
board = [[0 for i in range(MAX_BOARD)] for j in range(MAX_BOARD)]
estimator = None
oppo_move = None

WORK_FOLDER = r"D:\lesson\3-\人工智能\Final Project\Final\nn"
CRITIC_NETWORK_SAVEPATH = WORK_FOLDER + r'\weights'

# f = open("C:/Users/Little_Zhang/Desktop/人工智能/finalpj/ADP-MCTS/text.txt", 'w')

action_network_me = ActionNetwork(objective=1, EPSILON=0.)
critic_network = CriticNetwork(params=[7 * 4 + 2, 16, 1])
critic_network.layers[0].w = w1
critic_network.layers[0].b = b1
critic_network.layers[1].w = w2
critic_network.layers[1].b = b2
# if os.path.exists(CRITIC_NETWORK_SAVEPATH):
#     critic_network.layers = pickle.load(open(CRITIC_NETWORK_SAVEPATH, 'rb'))




def get_candidate(role):
    if estimator:
        moves = estimator.get_actions(role)  # 返回临近的点
        mv_values = []
        for move in moves:
            estimator.make_move(move)
            mv_values.append(critic_network.forward(estimator))
            estimator.draw_move(move)
            # x, y = action
            # # 能不能不用deepcopy
            # env_next = env.deepcopy()
            # env_next[x][y] = role
            # values.append(critic_network.forward(env_next))
        # actions = [i[0] for i in moves]
        return moves, mv_values
    else:
        pass


def brain_init():
    global estimator, oppo_move
    estimator = None
    oppo_move = None
    # f.write("init\n")
    if pp.width < 5 or pp.height < 5:
        pp.pipeOut("ERROR size of the board")
        return
    if pp.width > MAX_BOARD or pp.height > MAX_BOARD:
        pp.pipeOut("ERROR Maximal board size is {}".format(MAX_BOARD))
        return
    pp.pipeOut("OK")


def brain_restart():
    global estimator, oppo_move
    estimator = None
    oppo_move = None
    for x in range(pp.width):
        for y in range(pp.height):
            board[x][y] = 0
    pp.pipeOut("OK")


def isFree(x, y):
    return x >= 0 and y >= 0 and x < pp.width and y < pp.height and board[x][y] == 0


def brain_my(x, y):
    if isFree(x, y):
        board[x][y] = 1
    else:
        pp.pipeOut("ERROR my move [{},{}]".format(x, y))


def brain_opponents(x, y):
    global oppo_move, vegetable
    # f.write('oppo_turn\n')
    if isFree(x, y):
        board[x][y] = 2
        oppo_move = (x, y)
    else:
        pp.pipeOut("ERROR opponents's move [{},{}]".format(x, y))


def brain_block(x, y):
    if isFree(x, y):
        board[x][y] = 3
    else:
        pp.pipeOut("ERROR winning move [{},{}]".format(x, y))


def brain_takeback(x, y):
    if x >= 0 and y >= 0 and x < pp.width and y < pp.height and board[x][y] != 0:
        board[x][y] = 0
        return 0
    return 2


def brain_turn():
    global estimator, oppo_move
    try:
        # f.write('my_turn\n')
        if pp.terminateAI:
            return

        if estimator is None:
            estimator = ENV(np.array(board), 20, 20, 0)
            for i in range(20):
                for j in range(20):
                    if estimator.board[i,j]==2:
                        estimator.oppo.append((i,j))
            # f.write('estimator_init\n')
            # 第一步假定对方很强
            # candidates = [None]
        else:
            # candidates = estimator.get_candidates(brunch=8)
            # f.write('estimator_update\n')
            estimator.make_move(oppo_move)

        # f.write(str(candidates)+"\n")
        # move_t = oppo_move
        # f.write(str(move_t) + "\n")

        # f.write('try to get candidate\n')
        moves, values = get_candidate(0)
        # f.write('try to choose move\n')
        move, value = action_network_me.forward(moves, values)
        # f.write(str(move)+'\n')
        x, y = move
        pp.do_mymove(x, y)
        # f.write('finish:do_mymove\n')
        estimator.make_move((x, y))
        # f.write('estimator_update2\n')
    except:
        pass
    # logTraceBack()


def brain_end():
    pass


def brain_about():
    pp.pipeOut(pp.infotext)


if DEBUG_EVAL:
    import win32gui


    def brain_eval(x, y):
        # TODO check if it works as expected
        wnd = win32gui.GetForegroundWindow()
        dc = win32gui.GetDC(wnd)
        rc = win32gui.GetClientRect(wnd)
        c = str(board[x][y])
        win32gui.ExtTextOut(dc, rc[2] - 15, 3, 0, None, c, ())
        win32gui.ReleaseDC(wnd, dc)

######################################################################
# A possible way how to debug brains.
# To test it, just "uncomment" it (delete enclosing """)
######################################################################
"""
# define a file for logging ...
DEBUG_LOGFILE = "C:/Users/Little_Zhang/Desktop/人工智能/finalpj/Log"
# ...and clear it initially
with open(DEBUG_LOGFILE, "w") as f:
    pass


# define a function for writing messages to the file
def logDebug(msg):
    with open(DEBUG_LOGFILE, "a") as f:
        f.write(msg + "\n")
        f.flush()


# define a function to get exception traceback
def logTraceBack():
    import traceback
    with open(DEBUG_LOGFILE, "a") as f:
        traceback.print_exc(file=f)
        f.flush()
    raise


# use logDebug wherever
# use try-except (with logTraceBack in except branch) to get exception info
# an example of problematic function
# def brain_turn():
# # 	logDebug("some message 1")
# # 	try:
# # 		logDebug("some message 2")
# # 		1. / 0. # some code raising an exception
# # 		logDebug("some message 3") # not logged, as it is after error
# # 	except:
# # 		logTraceBack()
"""
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
if DEBUG_EVAL:
    pp.brain_eval = brain_eval


def main():
    pp.main()


if __name__ == "__main__":
    main()
