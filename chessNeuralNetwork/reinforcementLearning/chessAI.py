import numpy as np
import chess
import chess.svg
from cairosvg import svg2png
import matplotlib.pyplot as plt
from params import *
from utils.calcDeffence import DeffenceAlgorithm
from utils.calcAttack import AttackAlgorithm

import matplotlib.image as mpimg
from time import sleep

# board = chess.Board('8/5k2/8/8/8/3KR3/8/8 w - - 0 0')

class ChessGame():

    history_moves = []
    is_white_move = True

    def __init__(self) :
        self.board = chess.Board(START_FEN)

        # plt.ion()
        # fig, axs = plt.subplots(1,sharex=True)
        # ax = axs
        # display.start(board.fen())

        self.default_board = np.array(default_board)
        self.deffenceCalc = DeffenceAlgorithm()
        self.attackCalc = AttackAlgorithm()
    # while not board.is_checkmate():
        
            
    def createX(self):
        X = []
        for row in self.default_board:
            for cell in row:

                if cell.isupper():
                    val = FIGURE_VALUES_FOR_NET_MY[cell.lower()]
                elif cell == '.':
                    # new_val = 0.01 * weight_of_field_white[idx_digit][idx_letter]
                    val = 0
                else:
                    val = FIGURE_VALUES_FOR_NET_ENEMY[cell.lower()]

                X.append([val])
            
        

        return X

    def move(self, currPos, targetPos, board):

        currentPointer = (LETTER_MAP.index(currPos[0].upper()), DIGIT_MAP.index(currPos[1]))

        currentVal = board[currentPointer[1]][currentPointer[0]]
        
        targetPointer = (LETTER_MAP.index(targetPos[0].upper()), DIGIT_MAP.index((targetPos[1])))
        targetVal = board[targetPointer[1]][targetPointer[0]]

        board[currentPointer[1]][currentPointer[0]] = '.'
        board[targetPointer[1]][targetPointer[0]] = currentVal

        return board

    def generate_clear_board_points(self): 
        new_board = {}

        for digit in DIGIT_MAP:
            new_board[digit] = {}
            for letter in LETTER_MAP:
                new_board[digit][letter] = {
                    'attack_figure_value': None,
                    'deffence_figure_value': None,
                    'attack_value': [0],
                    'defence_value': [0],
                    'figure_attack': None,
                    'figure_deffence': None,
                }
        return  new_board

    def generate_board_points_and_clear_board_field(self, is_white_move, board_points, enemy_figure_indexes, my_figure_indexes): 
        board = default_board
        if is_white_move:

            for idx_digit, digit in enumerate(DIGIT_MAP):
                for idx_letter, letter in enumerate(LETTER_MAP):
                    cell = board[idx_digit][idx_letter]
                    new_att_val = -1
                    new_def_val = -1
                    if cell.isupper():
                        new_att_val = -1
                        new_def_val = FIGURE_VALUES_TO_ATTACK[cell.lower()]
                        board_points[digit][letter]['figure_deffence'] = cell
                        self.deffenceCalc.generateValueForField(cell.lower(), (idx_digit, idx_letter), is_white_move, board_points, my_figure_indexes)
                    elif cell == '.':
                        # new_val = 0.01 * weight_of_field_white[idx_digit][idx_letter]
                        new_att_val = 0
                        new_def_val = 0
                    else:
                        new_att_val = FIGURE_VALUES_TO_ATTACK[cell.lower()]
                        new_def_val = -1
                        board_points[digit][letter]['figure_attack'] = cell
                        self.attackCalc.generateValueForField(cell.lower(), (idx_digit, idx_letter), is_white_move, board_points, enemy_figure_indexes)
                    board_points[digit][letter]['attack_figure_value'] = new_att_val
                    board_points[digit][letter]['deffence_figure_value'] = new_def_val


        else:
            for idx_digit, digit in enumerate(DIGIT_MAP):
                for idx_letter, letter in enumerate(LETTER_MAP):
                    cell = board[idx_digit][idx_letter]
                    new_att_val = -1
                    new_def_val = -1
                    if cell.islower():
                        new_att_val = -1
                        new_def_val = FIGURE_VALUES_TO_ATTACK[cell.lower()]
                        board_points[digit][letter]['figure_deffence'] = cell
                        self.deffenceCalc.generateValueForField(cell.lower(), (idx_digit, idx_letter), is_white_move, board_points, my_figure_indexes)
                    elif cell == '.':
                        # new_val = 0.01 * weight_of_field_black[idx_digit][idx_letter]
                        new_att_val = 0
                        new_def_val = 0
                    else:
                        new_att_val = FIGURE_VALUES_TO_ATTACK[cell.lower()]
                        new_def_val = -1
                        board_points[digit][letter]['figure_attack'] = cell
                        self.attackCalc.generateValueForField(cell.lower(), (idx_digit, idx_letter), is_white_move, board_points, enemy_figure_indexes)
                    board_points[digit][letter]['attack_figure_value'] = new_att_val
                    board_points[digit][letter]['deffence_figure_value'] = new_def_val


        return board_points

    # print(board)
        # if is_white_move:
        #     default_board = white_board
        # else:
        #     default_board = black_board
    
    def playMe(self, possible_move):

        if self.is_white_move:
            print('WIN BLACK')
            return False
        else:
            print('WIN WHITE')
            return True
        uniq_figures = np.array(list(FIGURE_VALUES_ENEMY.keys()))
        if is_white_move:
            enemy_figures = np.char.lower(uniq_figures)
            my_figures = np.char.upper(uniq_figures)
        else:
            enemy_figures = np.char.upper(uniq_figures)
            my_figures = np.char.lower(uniq_figures)

        enemy_figure_indexes = []
        my_figure_indexes = []

        for figure in enemy_figures:
            t_enemy = np.where(default_board == figure)
            for idx, item in enumerate(t_enemy[0]): 
                enemy_figure_indexes.append([t_enemy[0][idx], t_enemy[1][idx]])
        for figure in my_figures:
            t_my = np.where(default_board == figure)
            for idx, item in enumerate(t_my[0]): 
                my_figure_indexes.append([t_my[0][idx], t_my[1][idx]])
                
        legal_moves = list(self.board.legal_moves)
        most_quality_legal_move = 0
        move_value = -100
        values_for_target_field = {}
        # print(enemy_figure_indexes)
        # print('is_checkmate=' + board.is_checkmate)

        clear_board_points = self.generate_clear_board_points()
        generated_board_points = self.generate_board_points_and_clear_board_field(is_white_move, clear_board_points, enemy_figure_indexes, my_figure_indexes)
        # print(generated_board_points)
        move_values = []

        for legal_move in legal_moves:
            legal_move = str(legal_move)

            attack_figure_val = generated_board_points[legal_move[3]][legal_move[2].upper()]['attack_figure_value']
            deff_val = generated_board_points[legal_move[3]][legal_move[2].upper()]['defence_value']
            attack_val = generated_board_points[legal_move[3]][legal_move[2].upper()]['attack_value']
            figure_def = generated_board_points[legal_move[1]][legal_move[0].upper()]['figure_deffence']

            targetY = DIGIT_MAP.index(legal_move[3])
            targetX = LETTER_MAP.index(legal_move[2].upper())

            # possible_moves[targetY][targetX] = figure_def
            # print('figure_def=' + str(figure_def))

            figure_def_Val = FIGURE_VALUES_ENEMY[figure_def.lower()]
        # =======================================================
            # curr_val = attack_figure_val - (np.mean(attack_val)) - (figure_def_Val)
            # curr_val = attack_figure_val - attack_calc[targetY][targetX] - figure_def_Val
            curr_val = attack_figure_val - (np.mean(attack_val)) - (figure_def_Val)
            # if is_white_move:
            # else:
                #     curr_val = calc_def_minus_att[targetY][targetX] - figure_def_Val

            # =======================================================
                
            move_values[targetY][targetX].append(curr_val)

            if curr_val not in values_for_target_field.keys():
                values_for_target_field[curr_val] = {}

            legalMoveDict = {
                'def_figure': figure_def,
                'legal_move': legal_move,
            }
            values_for_target_field[curr_val][figure_def_Val] = legalMoveDict

            if curr_val > move_value:
                move_value = curr_val
            # print('legal_move = ' + legal_move + 'val=' + str(generated_board_points[legal_move[3]][legal_move[2].upper()]))

        most_val_moves = values_for_target_field[move_value]

        min_figure = np.min(list(most_val_moves.keys()))
        # min_figure = most_val_moves[min_figure]
        print( 'most_val_moves', most_val_moves)
        print( 'most_val_moves', min_figure)


        # legal_random_move = np.random.choice(most_val_moves)
        legal_random_move = most_val_moves[min_figure]['legal_move']

        legal_move_split = (legal_random_move[0] + legal_random_move[1], legal_random_move[2] + legal_random_move[3])

        most_quality_legal_move = legal_random_move

        legal_move_split = (legal_random_move[0] + legal_random_move[1], legal_random_move[2] + legal_random_move[3])

        default_board = self.move(legal_move_split[0], legal_move_split[1], default_board)
        self.board.push_san(most_quality_legal_move )

        is_white_move = not is_white_move
        


    def auto_play_chess(self):
        is_white_move = True
        values_for_target_field = {}
        last_moves = []
        last_moves_black = []
        self.history_moves = []

        last_used_figures = []
        last_used_figures_black = []
        for _ in range(NR_MOVES):
            uniq_figures = np.array(list(FIGURE_VALUES_ENEMY.keys()))
            if is_white_move:
                enemy_figures = np.char.lower(uniq_figures)
                my_figures = np.char.upper(uniq_figures)
            else:
                enemy_figures = np.char.upper(uniq_figures)
                my_figures = np.char.lower(uniq_figures)

            enemy_figure_indexes = []
            my_figure_indexes = []

            for figure in enemy_figures:
                t_enemy = np.where(self.default_board == figure)
                for idx, item in enumerate(t_enemy[0]): 
                    enemy_figure_indexes.append([t_enemy[0][idx], t_enemy[1][idx]])
            for figure in my_figures:
                t_my = np.where(self.default_board == figure)
                for idx, item in enumerate(t_my[0]): 
                    my_figure_indexes.append([t_my[0][idx], t_my[1][idx]])
                    
            legal_moves = list(self.board.legal_moves)
            if len(list(legal_moves)) == 0:
                if is_white_move:
                    print('WIN BLACK')
                    return False
                else:
                    print('WIN WHITE')
                    return True

            self.history_moves.append(self.createX())
            
            # print(enemy_figure_indexes)
            # print('is_checkmate=' + board.is_checkmate)

            clear_board_points = self.generate_clear_board_points()
            generated_board_points = self.generate_board_points_and_clear_board_field(is_white_move, clear_board_points, enemy_figure_indexes, my_figure_indexes)
            # print(generated_board_points)

            most_quality_legal_move = 0
            move_value = -100
            values_for_target_field = {}

            # quality_me = getQualitySetup(is_white_move)
            # quality_enemy = getQualitySetup(not is_white_move)

            attack_figure = []
            deffence_figure = []
            attack = []
            deffence = []
            move_values = []

            for row in generated_board_points.keys():
                attFigArr = []
                deffFigArr = []
                deffArr = []
                attackArr = []
                move_valuesArr = []
                # attack_figure.append([])
                for col in generated_board_points[row].keys():
                    # print(generated_board_points[row][col])
                    attFigArr.append(generated_board_points[row][col]['attack_figure_value'])
                    deffFigArr.append(generated_board_points[row][col]['deffence_figure_value'])
                    deffArr.append(round(np.sum(generated_board_points[row][col]['defence_value']), 3))
                    attackArr.append(round(np.sum(generated_board_points[row][col]['attack_value']), 3))
                    move_valuesArr.append([])
                attack_figure.append(attFigArr)
                deffence_figure.append(deffFigArr)
                deffence.append(deffArr)
                attack.append(attackArr)
                move_values.append(move_valuesArr)
                # attack.put(values=attArr)
                # deffence.put(values=deffArr)
                # # attack.put

            deffence = np.array(deffence)
            attack = np.array(attack)
            attack_figure = np.array(attack_figure)
            deffence_figure = np.array(deffence_figure)


            print('is_white_move=' + str(is_white_move))
            # print('============== attack_figure ===================')
            # print (attack_figure, sep=' ')
            # print('============== deffence_figure ===================')
            # print (deffence_figure, sep=' ')
            print('============== attack ===================')
            print (attack, sep=' ')
            print('============== deffence ===================')
            print (deffence, sep=' ')
            
            attack_calc = attack_figure + attack
            deffence_calc = deffence_figure + deffence
            print('============== attack_figure + attack ===================')
            print (attack_calc)
            print('============== deffence_figure + deffence ===================')
            print (deffence_calc)

            calc_def_minus_att =  deffence_calc - attack_calc 
            print('============== deffence_calc - attack_calc ===================')
            print (calc_def_minus_att)
            
            calc_att_minusd_def = attack_calc - deffence_calc
            print('============== attack_calc - deffence_calc ===================')
            print (calc_att_minusd_def)        

            print('count legal_moves=' + str(len(list(legal_moves))))
            possible_moves = np.array(['o' for _ in range(64)])

            possible_moves = possible_moves.reshape(8,8)
            # print(default_board, end='\n\n')
            # print(board, end='\n\n')
            # print(generated_board_points, end='\n\n')

            # print(legal_moves )

            for legal_move in legal_moves:
                legal_move = str(legal_move)

                attack_figure_val = generated_board_points[legal_move[3]][legal_move[2].upper()]['attack_figure_value']
                deff_val = generated_board_points[legal_move[3]][legal_move[2].upper()]['defence_value']
                attack_val = generated_board_points[legal_move[3]][legal_move[2].upper()]['attack_value']
                figure_def = generated_board_points[legal_move[1]][legal_move[0].upper()]['figure_deffence']

                targetY = DIGIT_MAP.index(legal_move[3])
                targetX = LETTER_MAP.index(legal_move[2].upper())

                possible_moves[targetY][targetX] = figure_def
                # print('figure_def=' + str(figure_def))

                figure_def_Val = FIGURE_VALUES_ENEMY[figure_def.lower()]
            # =======================================================
                # curr_val = attack_figure_val - (np.mean(attack_val)) - (figure_def_Val)
                # curr_val = attack_figure_val - attack_calc[targetY][targetX] - figure_def_Val
                if is_white_move:
                    curr_val = attack_figure_val - (np.mean(attack_val)) - (figure_def_Val)
                else:
                    curr_val = calc_def_minus_att[targetY][targetX] - figure_def_Val

            # =======================================================
                
                move_values[targetY][targetX].append(curr_val)

                if curr_val not in values_for_target_field.keys():
                    values_for_target_field[curr_val] = {}

                legalMoveDict = {
                    'def_figure': figure_def,
                    'legal_move': legal_move,
                }
                values_for_target_field[curr_val][figure_def_Val] = legalMoveDict

                if curr_val > move_value:
                    move_value = curr_val
                # print('legal_move = ' + legal_move + 'val=' + str(generated_board_points[legal_move[3]][legal_move[2].upper()]))
            is_checkmate = self.board.is_checkmate()
            print('==========================================possible_moves ==========================================' )
            print(possible_moves )
            print('is_checkmate=' + str(is_checkmate) )
            print( values_for_target_field)

            # for TEST 
            # if not is_white_move:
            #     values_for_target_field = {}
            #     values_for_target_field[move_value] = {}
            #     legalMoveDict = {
            #             'def_figure': 'k',
            #             'legal_move': 'e8d8',
            #         }
            #     values_for_target_field[move_value][0.9] = legalMoveDict

            print (move_values, sep=' ')

            print( values_for_target_field)

            most_val_moves = values_for_target_field[move_value]

            min_figure = np.min(list(most_val_moves.keys()))
            # min_figure = most_val_moves[min_figure]
            print( 'most_val_moves', most_val_moves)
            print( 'most_val_moves', min_figure)
            print( 'most_val_moves.keys()', most_val_moves.keys())
            print( 'min_figure', min_figure)
            print( 'min_figure', most_val_moves[min_figure])


            # legal_random_move = np.random.choice(most_val_moves)
            legal_random_move = most_val_moves[min_figure]

            legal_move_split = (legal_random_move['legal_move'][0] + legal_random_move['legal_move'][1], legal_random_move['legal_move'][2] + legal_random_move['legal_move'][3])

            most_quality_legal_move = legal_random_move

            av_moves = []
            if is_white_move and last_moves.count(most_quality_legal_move) > 3:
                for key in values_for_target_field.keys():
                    v = values_for_target_field[key]
                    for innerK in v.keys():

                        vk = v[innerK]['legal_move']
                        if vk not in last_moves:
                            av_moves.append(v)

                if len(av_moves) > 0:
                    legal_random_move_rand = np.random.choice(av_moves)
                    legal_random_move = legal_random_move_rand[list(legal_random_move_rand.keys())[0]]
                else :
                    legal_random_move = most_quality_legal_move

                legal_move_split = (legal_random_move['legal_move'][0] + legal_random_move['legal_move'][1], legal_random_move['legal_move'][2] + legal_random_move['legal_move'][3])

                most_quality_legal_move = legal_random_move
                print(legal_random_move)


            elif not is_white_move and last_moves_black.count(most_quality_legal_move) > 3:
                for key in values_for_target_field.keys():
                    v = values_for_target_field[key]
                    for innerK in v.keys():

                        vk = v[innerK]['legal_move']
                        if vk not in last_moves:
                            av_moves.append(v)

                if len(av_moves) > 0:
                    legal_random_move_rand = np.random.choice(av_moves)
                    legal_random_move = legal_random_move_rand[list(legal_random_move_rand.keys())[0]]
                else :
                    legal_random_move = most_quality_legal_move

                legal_move_split = (legal_random_move['legal_move'][0] + legal_random_move['legal_move'][1], legal_random_move['legal_move'][2] + legal_random_move['legal_move'][3])

                most_quality_legal_move = legal_random_move
                print(legal_random_move)
            # print(last_moves)

            if is_white_move:
                last_moves.append(most_quality_legal_move)
                if len(last_moves) > 30:

                    last_moves = last_moves[len(last_moves) - 30:]
                
            else :
                last_moves_black.append(most_quality_legal_move)
                if len(last_moves_black) > 30:

                    last_moves_black = last_moves_black[len(last_moves_black) - 30:]

            DEF_FIGURE = most_quality_legal_move['def_figure']

            column_move_difference = np.abs(LETTER_MAP.index(legal_move_split[1][0].upper()) - LETTER_MAP.index(legal_move_split[0][0].upper()))

            print('column_move_difference=' + str(column_move_difference))
            print('move_value=' + str(move_value))
            print('i=' + str(_))
            # print('quality_me=' + str(quality_me))
            # print('quality_enemy=' + str(quality_enemy))
            # print('move_value=' + str(move_value))
            # print('is_white_move=' + str(is_white_move))
            print(legal_move_split[0] + ' -> ' + legal_move_split[1])

            self.default_board = self.move(legal_move_split[0], legal_move_split[1], default_board)
            self.board.push_san(most_quality_legal_move['legal_move'] )

            print(self.board, end='\n\n')
            # print(default_board, end='\n\n')
            # print('DEF_FIGURE', DEF_FIGURE)


                # print('new_figure', self.default_board)

                # return True
            self.initPromotionFigureAndCastling(legal_move_split, DEF_FIGURE, is_white_move, column_move_difference)

            is_white_move = not is_white_move
   
            # display.update(board.fen())
            # ax.clear()
            # # ax.plot(predictedX, predictedY)

            # svg_text = chess.svg.board(
            #     board,
            #     # fill=dict.fromkeys(board.attacks(chess.E4), "#cc0000cc"),
            #     # arrows=[chess.svg.Arrow(chess.E4, chess.F6, color="#0000cccc")],
            #     # squares=chess.SquareSet(chess.BB_DARK_SQUARES & chess.BB_FILE_B),
            #     size=350,
            # ) 

            # with open('example-board.svg', 'w') as f:
            #     f.write(svg_text)

            # svg2png(bytestring=svg_text, write_to='board.png')
            # img = mpimg.imread('board.png')
            # imgplot = ax.imshow(img)

            # plt.pause(0.02)
            # sleep(1)
        return False
    def initPromotionFigureAndCastling(self, legal_move_split, DEF_FIGURE, is_white_move, column_move_difference):
        targetRow = legal_move_split[1][1]
        targetCol = legal_move_split[1][0]
        if (DEF_FIGURE == 'P' and is_white_move and targetRow == '8') or (DEF_FIGURE == 'p' and not is_white_move and targetRow == '1'):

            digits = DIGIT_MAP_NOT_REV
            # digits.reverse()

            cell_nr = (digits.index(targetRow)) * 8 + LETTER_MAP.index(targetCol.upper())
            # print('DIGIT_MAP', DIGIT_MAP)
            # print('digits', digits)
            # print('cell_nr', cell_nr)
            # print('new_figure', new_figure)
            new_figure = str(game.board.piece_at(cell_nr))

            if is_white_move :
                self.default_board[DIGIT_MAP.index(targetRow)][LETTER_MAP.index(targetCol.upper())] = new_figure.upper()
            else :
                self.default_board[DIGIT_MAP.index(targetRow)][LETTER_MAP.index(targetCol.upper())] = new_figure.lower()
        
        if column_move_difference > 1:
            if (DEF_FIGURE == 'K' and is_white_move and targetRow == '1' and targetCol == 'g' ) :
                if self.default_board[DIGIT_MAP.index('1')][LETTER_MAP.index('H')] == 'R':
                    self.move('h1', 'f1', default_board)
            elif (DEF_FIGURE == 'K' and is_white_move and targetRow == '1' and targetCol == 'c' ):
                if self.default_board[DIGIT_MAP.index('1')][LETTER_MAP.index('A')] == 'R':
                    self.move('a1', 'd1', default_board)
            elif (DEF_FIGURE == 'k' and not is_white_move and targetRow == '8' and targetCol == 'g' ) :
                if self.default_board[DIGIT_MAP.index('8')][LETTER_MAP.index('H')] == 'r':
                    self.move('h8', 'f8', default_board)
            elif (DEF_FIGURE == 'k' and not is_white_move and targetRow == '8' and targetCol == 'c' ):
                if self.default_board[DIGIT_MAP.index('8')][LETTER_MAP.index('A')] == 'r':
                    self.move('a8', 'd8', default_board)

# {'8': {'A': {'attack_value': 0.3, 'defence_value': []}, 'B': {'attack_value': 0.25, 'defence_value': ['r']}, 'C': {'attack_value': 0.25, 'defence_value': ['q']}, 'D': {'attack_value': 0.6, 'defence_value': []}, 'E': {'attack_value': 0.9, 'defence_value': ['q']}, 'F': {'attack_value': 0.25, 'defence_value': []}, 'G': {'attack_value': 0.25, 'defence_value': ['r']}, 'H': {'attack_value': 0.3, 'defence_value': []}}, '7': {'A': {'attack_value': 0.1, 'defence_value': ['r']}, 'B': {'attack_value': 0.1, 'defence_value': ['b']}, 'C': {'attack_value': 0.1, 'defence_value': ['q']}, 'D': {'attack_value': 0.1, 'defence_value': ['b', 'q']}, 'E': {'attack_value': 0.1, 'defence_value': ['q', 'b']}, 'F': {'attack_value': 0.1, 'defence_value': []}, 'G': {'attack_value': 0.1, 'defence_value': ['b']}, 'H': {'attack_value': 0.1, 'defence_value': ['r']}}, '6': {'A': {'attack_value': 0.01, 'defence_value': ['r', 'n', 'b', 'r', 'b']}, 'B': {'attack_value': 0.01, 'defence_value': ['q']}, 'C': {'attack_value': 0.01, 'defence_value': ['n']}, 'D': {'attack_value': 0.01, 'defence_value': ['q', 'b', 'q']}, 'E': {'attack_value': 0.01, 'defence_value': ['b']}, 'F': {'attack_value': 0.01, 'defence_value': ['q', 'n', 'p']}, 'G': {'attack_value': 0.01, 'defence_value': []}, 'H': {'attack_value': 0.01, 'defence_value': ['b', 'n', 'r', 'b', 'r']}}, '5': {'A': {'attack_value': 0.01, 'defence_value': ['r', 'q', 'r']}, 'B': {'attack_value': 0.01, 'defence_value': ['b']}, 'C': {'attack_value': 0.01, 'defence_value': ['b']}, 'D': {'attack_value': 0.01, 'defence_value': ['q', 'q']}, 'E': {'attack_value': 0.01, 'defence_value': []}, 'F': {'attack_value': 0.01, 'defence_value': ['b']}, 'G': {'attack_value': 0.01, 'defence_value': ['q', 'b']}, 'H': {'attack_value': 0.01, 'defence_value': ['r', 'q', 'r']}}, '4': {'A': {'attack_value': 0.01, 'defence_value': ['r', 'r', 'q']}, 'B': {'attack_value': 0.01, 'defence_value': ['b']}, 'C': {'attack_value': 0.01, 'defence_value': ['b']}, 'D': {'attack_value': 0.01, 'defence_value': ['q', 'q']}, 'E': {'attack_value': 0.01, 'defence_value': []}, 'F': {'attack_value': 0.01, 'defence_value': ['b']}, 'G': {'attack_value': 0.01, 'defence_value': ['b', 'q']}, 'H': {'attack_value': 0.01, 'defence_value': ['q', 'r', 'r']}}, '3': {'A': {'attack_value': 0.01, 'defence_value': ['r', 'b', 'r', 'n', 'b']}, 'B': {'attack_value': 0.01, 'defence_value': ['q']}, 'C': {'attack_value': 0.01, 'defence_value': ['n']}, 'D': {'attack_value': 0.01, 'defence_value': ['q', 'q', 'b']}, 'E': {'attack_value': 0.01, 'defence_value': ['b']}, 'F': {'attack_value': 0.01, 'defence_value': ['p', 'q', 'n']}, 'G': {'attack_value': 0.01, 'defence_value': []}, 'H': {'attack_value': 0.01, 'defence_value': ['b', 'r', 'b', 'n', 'r']}}, '2': {'A': {'attack_value': -1, 'defence_value': ['r', 'r']}, 'B': {'attack_value': -1, 'defence_value': ['b']}, 'C': {'attack_value': -1, 'defence_value': ['q']}, 'D': {'attack_value': -1, 'defence_value': ['q', 'n', 'b', 'q']}, 'E': {'attack_value': -1, 'defence_value': ['q', 'b', 'n']}, 'F': {'attack_value': -1, 'defence_value': []}, 'G': {'attack_value': -1, 'defence_value': ['b']}, 'H': {'attack_value': -1, 'defence_value': ['r', 'r']}}, '1': {'A': {'attack_value': -1, 'defence_value': ['r', 'q', 'r']}, 'B': {'attack_value': -1, 'defence_value': ['r', 'q', 'r']}, 'C': {'attack_value': -1, 'defence_value': ['r', 'q', 'r']}, 'D': {'attack_value': -1, 'defence_value': ['q', 'r', 'r']}, 'E': {'attack_value': -1, 'defence_value': ['r', 'q', 'r']}, 'F': {'attack_value': -1, 'defence_value': ['p', 'r', 'q', 'r']}, 'G': {'attack_value': -1, 'defence_value': ['r', 'q', 'r']}, 'H': {'attack_value': -1, 'defence_value': ['r', 'r', 'q']}}}

        
# board.is_checkmate()
# board.find_move(('h1'), ('h3

# svg2png(bytestring=svg_text, write_to='example-board.png')
# sleep(100)
    

game = ChessGame()

game.auto_play_chess()
print(game.default_board, end='\n\n')

# print(game.board.piece_at(16))
# for el in game.board.pieces():
#     # print(game.board.piece_map()[el])
#     print(el)