"""
This is a Python gomoku AI agent using alpha-beta pruning, a final PJ for class AI, FDU-DataScience

Author: DWB, ZZJ

Date: 2020.12
"""
import numpy as np
import os
import pickle
import pisqpipe as pp
from pisqpipe import DEBUG_EVAL
from ADP_linear import *
from CriticalNetwork import CriticNetwork, Action_Choose

w1 = np.array([[ 2.35272356e-01, -3.26694613e-01, -5.56543187e-01,
         5.87512277e-01, -8.34733979e-03,  5.01183306e-01,
        -7.63988229e-02, -1.88386572e-01,  3.32848026e-02,
         2.90367517e-01, -6.25871532e-01,  1.91222005e-01,
         1.13715390e-01, -2.00654467e-02, -6.36136779e-01,
        -2.33760412e-01,  1.89235613e-01,  2.31089669e-01,
         6.03969796e-02, -1.62944745e-01,  3.17433614e-01,
         1.67095551e-01, -1.04072055e-01,  4.90951036e-01,
        -6.76658255e-02,  8.05722229e-02, -8.28187576e-02,
        -1.26194013e-02, -1.28045036e-01,  2.33474983e-02],
       [ 4.01022592e-01, -2.29575378e-01, -3.34024450e-01,
         2.36832104e-01,  1.68302329e-01, -3.32582379e-02,
        -4.56818852e-01, -1.82130851e-01, -3.47232022e-01,
        -3.75759786e-02, -1.88239659e-01,  4.45643808e-01,
        -1.06740042e-01,  4.98726744e-01,  5.05613859e-02,
        -2.99247768e-01,  1.43220685e-01, -1.40538725e-01,
        -4.00688024e-01, -3.22680623e-01,  3.63025639e-02,
        -5.69089182e-02, -3.33313036e-02,  3.58897193e-01,
        -2.36747498e-01, -1.12783140e-01, -7.28337941e-02,
         5.98855822e-02, -1.75547433e-01, -2.72103090e-01],
       [ 4.80850490e-01, -2.93576308e-01, -3.10165783e-01,
        -1.66705681e-01, -4.93076320e-01,  1.09864616e-01,
        -2.31471217e-01, -5.36512298e-02, -1.43136060e-01,
        -1.17102592e-01, -5.24033964e-01, -1.42968318e-01,
        -1.21085450e-01, -3.35711101e-01, -2.66483626e-01,
         1.99540437e-01, -2.92246622e-01,  2.23391383e-01,
         2.80827704e-01,  2.74250096e-01,  3.05195592e-01,
         3.23642979e-01,  4.89565862e-02,  4.29304911e-01,
         2.83255113e-01,  2.04430150e-01, -5.16735097e-01,
        -6.31680709e-02, -1.30673848e-01,  4.07406956e-01],
       [-4.88186117e-01,  5.37444933e-01,  5.56836971e-01,
        -4.27587697e-01, -2.46597266e-02, -7.47944465e-02,
        -3.16291825e-02, -5.93268373e-01, -3.79733124e-01,
         1.03877013e-01,  6.49513031e-01, -1.10409039e-02,
         4.84930215e-02, -3.59793316e-01, -5.05680707e-02,
        -7.43690126e-02, -1.94756471e-01, -2.35716689e-01,
         1.28613044e-01, -8.40788430e-01, -8.56909316e-02,
         2.46845902e-01,  7.79790901e-01, -5.12989558e-01,
        -2.14841114e-01,  1.89307853e-01,  6.15227980e-01,
        -1.82519970e-01, -7.19759371e-02,  9.34595361e-02],
       [-9.72542148e-03,  3.07891836e-01, -3.00019142e-01,
        -3.26392979e-01,  2.76033768e-01, -2.66742084e-02,
         4.70778401e-01,  5.70790538e-02, -2.69422300e-01,
         1.57272399e-01, -5.97693838e-02,  5.58405333e-02,
        -4.10643224e-01, -7.40687299e-02, -5.19837037e-02,
         2.08341260e-01,  4.49214282e-01, -2.29323944e-01,
         1.31397867e-01, -4.57861940e-01,  1.13385686e-01,
         7.08457390e-02,  4.18428412e-01, -2.14301253e-01,
         3.42978095e-01, -1.79651108e-01,  2.77563692e-01,
         2.57513479e-01,  2.11374077e-01,  1.70945042e-01],
       [-4.26946673e-01,  2.85165615e-01,  4.75823672e-01,
         9.58746359e-02,  3.49609772e-01, -3.82941184e-01,
         5.55381240e-01, -8.52684930e-01,  3.29978848e-01,
        -2.62482436e-01,  1.27588279e-01,  6.51214903e-02,
         4.53981508e-01,  3.42302616e-01, -4.29471441e-02,
        -4.96456432e-01,  1.75581486e-01, -4.71122416e-01,
         3.99613177e-01, -1.01904066e-01, -4.62254899e-01,
        -2.66258546e-01,  7.22963710e-01, -4.66969970e-01,
        -3.10912308e-01, -2.76391644e-01,  9.15856990e-02,
        -4.96760738e-01,  6.04933689e-01, -1.93528656e-01],
       [ 6.97915260e-01, -4.09701329e-01, -6.51885785e-01,
         1.33924858e-01, -3.55126060e-01,  1.38087519e-01,
        -3.59743811e-01,  8.38972764e-01, -6.02528295e-02,
        -3.35973535e-01, -4.35574423e-01,  7.36144395e-02,
        -3.77950390e-01,  6.82794900e-03, -5.04718067e-01,
         1.02301817e-01, -4.89735748e-01, -3.49831829e-01,
        -7.35200399e-01,  5.74753074e-01,  2.06096307e-01,
         3.07494900e-01,  2.02566689e-01,  7.80222072e-03,
         1.13339173e-01,  3.80278012e-01,  1.66703648e-01,
         2.35561411e-01, -2.88357444e-01,  1.09431447e-01],
       [ 2.81095030e-01, -2.49440755e-01, -1.21410131e-01,
        -2.73418532e-01,  2.58363252e-01,  5.04513728e-01,
        -4.56269959e-01,  6.04657144e-01, -2.19207609e-01,
         4.67494570e-01,  2.42904112e-02,  7.97237058e-02,
        -2.30460691e-02, -1.97974624e-01, -2.85603208e-01,
         1.31521345e-01,  1.51304486e-01,  4.76019706e-02,
        -3.08754227e-01,  2.41194366e-01,  4.70811201e-02,
         5.56085112e-02, -2.03111206e-01,  2.24754034e-02,
         2.27582597e-01,  4.18191486e-01, -6.09089855e-01,
         5.60781491e-01, -6.09799103e-01,  5.98461457e-01],
       [-2.42148487e-01,  3.04365399e-01, -2.12076456e-02,
        -2.77719056e-01,  1.20587465e-01, -5.86492574e-01,
         3.77437146e-01, -2.24627668e-01, -3.47003231e-01,
        -2.98571330e-02,  4.20428798e-02, -1.01315573e-01,
        -2.86342978e-02,  2.65741450e-01,  2.10336309e-01,
        -3.04634859e-01,  4.39105595e-01,  3.56875324e-01,
        -8.44547648e-02,  1.51712023e-02,  1.86687988e-01,
        -1.02529306e-01,  3.85847369e-01, -1.35868308e-01,
         4.01933904e-01, -1.47339179e-01,  6.96683171e-01,
        -3.69328015e-01,  1.32030056e-01,  2.46297272e-01],
       [-9.29494748e-02,  3.53962729e-01,  1.73928368e-01,
        -7.31296928e-01,  1.97877921e-01, -4.05343766e-01,
         5.34364518e-01, -6.55343383e-01, -2.27427054e-01,
        -6.87668737e-02,  1.97979469e-01, -2.25869713e-01,
        -4.24496863e-01,  1.94645877e-01, -1.26356106e-01,
        -1.23074919e-02, -1.46404268e-01, -4.02076411e-01,
         2.10962882e-01, -1.36412073e-01,  4.35520496e-01,
        -2.07194718e-01,  6.72363413e-01, -3.92784661e-01,
         3.46255503e-01, -4.82102554e-01,  7.39544219e-01,
        -4.92856708e-02, -1.55783587e-01, -4.20526848e-01],
       [ 5.61601285e-01, -5.62182973e-01,  2.32816408e-02,
         7.15142957e-01,  2.46626060e-01, -6.02885619e-02,
        -7.53351984e-01, -2.17944210e-01,  1.46466624e-01,
        -2.85261266e-01, -5.51531328e-01,  6.34146182e-01,
         1.21379817e-01,  8.12417525e-02,  1.24309479e-02,
         4.35838658e-01,  1.25389937e-01, -6.26324619e-03,
         1.58047109e-02,  5.63852304e-01, -4.43190911e-01,
        -4.14807213e-01,  1.52196919e-01, -9.07124638e-02,
         5.05390957e-02, -1.34442785e-01, -5.08549463e-01,
         3.52653697e-03, -6.42611529e-01, -5.37825548e-02],
       [-2.08558378e-01,  3.31824739e-01,  1.92391676e-01,
         3.15804412e-01,  1.40705241e-02,  3.36446015e-01,
        -2.57466824e-01, -1.94288173e-02,  8.06148856e-02,
        -3.60693468e-01,  5.54323439e-01, -1.68874644e-01,
         6.41678723e-02,  9.58778144e-02,  5.14056445e-01,
        -5.41237887e-01, -1.94600930e-01,  3.13800962e-01,
         2.12707187e-01, -3.96081477e-01,  5.00077197e-01,
         4.53525482e-01,  6.20069107e-01, -3.05944095e-01,
         3.13830239e-02, -2.64142002e-01, -1.79613644e-01,
        -1.40354486e-01, -4.00803699e-02, -6.32842064e-01],
       [ 4.16831362e-01, -2.96623865e-01, -1.97155991e-01,
        -2.01868572e-01, -5.21305308e-01,  4.57642960e-01,
        -3.52499731e-01,  4.99310075e-01, -1.44923400e-01,
        -2.62190797e-01, -6.74095880e-01,  5.50021874e-01,
        -2.58978145e-01,  2.64277280e-03, -6.43512461e-01,
         5.03726207e-01, -1.47919735e-01,  4.02077861e-01,
        -3.32010680e-01,  4.87170537e-01,  1.11327276e-01,
         3.01885840e-01, -3.01315470e-01,  5.84879097e-01,
        -1.84217359e-01,  3.66530974e-01, -3.17210139e-01,
         4.18544354e-01, -1.75266489e-01, -5.51833381e-02],
       [ 4.68037870e-01, -4.60744488e-01, -5.85158439e-01,
         7.31327094e-01,  4.93315407e-02,  4.17202853e-01,
        -5.85027296e-01,  6.59062599e-01, -5.37845941e-02,
         2.94822426e-01, -1.19020542e-01, -5.90252082e-02,
        -5.09258943e-02,  1.46228060e-01, -5.83118049e-01,
         2.34724744e-01, -4.58978362e-01, -4.86386307e-01,
         2.81906341e-01, -2.09337082e-01,  2.38151073e-01,
         5.90556883e-02,  1.74849837e-01,  4.36306397e-01,
         1.66260557e-01,  3.24786485e-02, -4.46465802e-01,
        -1.85782910e-01, -7.03159469e-01,  4.24661547e-01],
       [ 5.91120251e-01, -5.95454593e-01, -4.20026274e-02,
         4.22642523e-01, -1.06587893e-01, -2.85810867e-01,
         1.31165505e-01,  5.98911973e-01, -1.72657639e-01,
         2.86351064e-01, -7.62603090e-01,  6.89485935e-01,
        -7.65880479e-02, -6.66277693e-02, -5.80061759e-01,
         3.71761709e-01,  7.01852764e-02,  3.64048765e-05,
        -3.29080579e-01,  5.16474751e-01, -4.38279555e-01,
         3.08253025e-01, -4.73593438e-01, -1.34237022e-01,
         9.57442040e-02, -1.61712263e-01, -1.22236648e-01,
        -2.13190186e-01, -5.11702004e-02,  4.98446292e-01],
       [-2.03954334e-01, -3.40904074e-01,  3.94473945e-01,
        -1.44432323e-01, -3.73695056e-01, -4.20738496e-01,
        -4.72415216e-01, -6.60983571e-02, -1.02506683e-01,
         3.00191353e-01,  1.77548219e-01,  5.60754075e-01,
        -2.11691352e-01,  3.47671533e-01, -4.52674672e-01,
         2.40147586e-02,  1.09439402e-01,  3.64612373e-01,
        -2.27006847e-01,  9.67035275e-02, -1.45322777e-01,
         7.05232077e-02,  8.74249796e-02,  2.97142729e-01,
         1.53876484e-01, -3.45240467e-01, -3.41774845e-01,
         4.53002695e-01, -3.98366499e-01,  1.98912605e-01]])

b1 = np.array([[ 0.0770253 ],
       [-0.00549706],
       [ 0.02493269],
       [ 0.01256829],
       [ 0.00072988],
       [-0.08689869],
       [ 0.06194331],
       [-0.02274077],
       [ 0.02157881],
       [ 0.06953133],
       [ 0.02529241],
       [ 0.01779375],
       [-0.00313861],
       [ 0.03866696],
       [-0.08187618],
       [ 0.0052497 ]])

w2 = np.array([[-0.95425174, -0.49124339, -0.89994461,  2.40414387,  0.68374852,
         2.55043626, -1.8813916 , -1.3371781 ,  1.28965398,  1.98935087,
        -1.482087  ,  1.2314682 , -2.37878371, -1.65258483, -1.86358964,
        -0.50827936]])

b2 = np.array([[0.97488382]])


# pp.infotext = infotext
MAX_BOARD = 100
board = [[0 for i in range(MAX_BOARD)] for j in range(MAX_BOARD)]
estimator = None
oppo_move = None

WORK_FOLDER = r"D:\lesson\3-\人工智能\Final Project\Final\nn"
WEIGHTS_PATH = WORK_FOLDER + r'\weigts'


action_network_me = Action_Choose(objective=1, threshold=0.)
critic_network = CriticNetwork(params=[7 * 4 + 2, 16, 1])
critic_network.layers[0].w = w1
critic_network.layers[0].b = b1
critic_network.layers[1].w = w2
critic_network.layers[1].b = b2

# if os.path.exists(WEIGHTS_PATH):
#     critic_network.layers = pickle.load(open(WEIGHTS_PATH, 'rb'))


def get_move(role):
    if estimator:
        moves = estimator.get_actions(role)  # 返回临近的点
        mv_values = []
        for move in moves:
            estimator.make_move(move)
            mv_values.append(critic_network.forward(estimator))
            estimator.draw_move(move)
        return moves, mv_values
    else:
        pass


def brain_init():
    global estimator, oppo_move
    estimator = None
    oppo_move = None
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
        else:
            estimator.make_move(oppo_move)



        moves, values = get_move(0)
        move, value = action_network_me.forward(moves, values)
        x, y = move
        pp.do_mymove(x, y)
        estimator.make_move((x, y))
    except:
        pass


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
