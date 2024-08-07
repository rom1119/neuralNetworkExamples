import numpy as np
import chess
import chess.svg
from cairosvg import svg2png
import matplotlib.pyplot as plt
from params import *
from utils.calcDeffence import DeffenceAlgorithm
from utils.calcAttack import AttackAlgorithm
import random

import matplotlib.image as mpimg
from time import sleep

# board = chess.Board('8/5k2/8/8/8/3KR3/8/8 w - - 0 0')

def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)

class ChessGame():

    last_moves_black = []

    history_boards = []
    selected_moves = []
    current_board_for_y = []
    is_white_move = True
    show_ui = True
    nr_moves_white = 0

    game_move_nr = 0

    def __init__(self, show_ui=False) :
        self.board = chess.Board()
        self.show_ui = show_ui

        if show_ui:
            plt.ion()
            fig, axs = plt.subplots(1,sharex=True)
            self.ax = axs
        # display.start(board.fen())

        self.default_board = np.array(default_board)
        self.deffenceCalc = DeffenceAlgorithm()
        self.attackCalc = AttackAlgorithm()
    # while not board.is_checkmate():
        
            
    def tryShowUI(self):

        if self.show_ui:
            self.ax.clear()
            svg_text = chess.svg.board(
                self.board,
                # fill=dict.fromkeys(board.attacks(chess.E4), "#cc0000cc"),
                # arrows=[chess.svg.Arrow(chess.E4, chess.F6, color="#0000cccc")],
                # squares=chess.SquareSet(chess.BB_DARK_SQUARES & chess.BB_FILE_B),
                size=350,
            ) 

            # with open('example-board.svg', 'w') as f:
            #     f.write(svg_text)

            svg2png(bytestring=svg_text, write_to='board.png')
            img = mpimg.imread('board.png')
            imgplot = self.ax.imshow(img)
            plt.pause(0.002)

    def selectMove(self, legal_move_split, most_quality_legal_move, legal_random_move_idx):
        targetY = DIGIT_MAP.index(legal_move_split[1][1])
        targetX = LETTER_MAP.index(legal_move_split[1][0].upper())
        self.current_board_for_y[targetY][targetX] = 3
        # print(f"current_board_for_y {self.current_board_for_y}")
        self.current_board_for_y = self.current_board_for_y.reshape(64)
        self.current_board_for_y = np.append(self.current_board_for_y, most_quality_legal_move['figure_def_val'])
        self.current_board_for_y = np.append(self.current_board_for_y, legal_random_move_idx / 10)
        # print(f"current_board_for_y AF {self.current_board_for_y}")
        # print(f"most_quality_legal_move AF {most_quality_legal_move['curr_val']}")
        a = self.current_board_for_y
        self.selected_moves.append(a)

    def revertMove(self, predict):
        Y = np.array(predict[0]).tolist()

        predict_y_def_figure = trunc(np.round(Y[len(Y) - 2], 1), 1)
        predict_y_move_idx = np.round(Y[len(Y) - 1], 1) * 10

        del Y[len(Y) - 1]
        # print(f"predict AF DEL= {predict}")
        del Y[len(Y) - 1]

        Y = np.array(Y)
        outputY = np.round(Y.reshape(8,8),2)

        outputY = trunc(outputY, 1)
        return (outputY, predict_y_def_figure, predict_y_move_idx)

    def createX(self):
        X = []
        for row in self.default_board:
            r = []
            for cell in row:
                if cell.isupper():
                    val = FIGURE_VALUES_FOR_NET_MY[cell.lower()]
                elif cell == '.':
                    # new_val = 0.01 * weight_of_field_white[idx_digit][idx_letter]
                    val = 0
                else:
                    val = FIGURE_VALUES_FOR_NET_ENEMY[cell.lower()]
                r.append(val)

            X.append(r)
        X = np.array(X)

        # li = np.array(X)
        # res = 0
        # for ids, x in enumerate(li):
        #     idx = ids + 1
        #     v = (0.000001 * idx) + (x + (idx ** 2)) 
        #     # print(f"idx={idx} x={x} (0.0001 * idx)={(0.000001 * idx)} v={v}")
        #     res += v
        # print(f"x")
        # print(f"{X}")
        x_resh = X.reshape(64)
        
        return x_resh

    def move(self, currPos, targetPos, board):

        currentPointer = (LETTER_MAP.index(currPos[0].upper()), DIGIT_MAP.index(currPos[1]))

        currentVal = self.default_board[currentPointer[1]][currentPointer[0]]
        
        targetPointer = (LETTER_MAP.index(targetPos[0].upper()), DIGIT_MAP.index((targetPos[1])))
        targetVal = self.default_board[targetPointer[1]][targetPointer[0]]

        self.default_board[currentPointer[1]][currentPointer[0]] = '.'
        self.default_board[targetPointer[1]][targetPointer[0]] = currentVal

        return self.default_board

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

    def generate_board_points_and_clear_board_field(self, board_points, enemy_figure_indexes, my_figure_indexes): 
        board = self.default_board
        if self.is_white_move:

            for idx_digit, digit in enumerate(DIGIT_MAP):
                for idx_letter, letter in enumerate(LETTER_MAP):
                    cell = board[idx_digit][idx_letter]
                    new_att_val = -1
                    new_def_val = -1
                    if cell.isupper():
                        new_att_val = -1
                        new_def_val = FIGURE_VALUES_TO_ATTACK[cell.lower()]
                        board_points[digit][letter]['figure_deffence'] = cell
                        self.deffenceCalc.generateValueForField(cell.lower(), (idx_digit, idx_letter), self.is_white_move, board_points, my_figure_indexes)
                    elif cell == '.':
                        # new_val = 0.01 * weight_of_field_white[idx_digit][idx_letter]
                        new_att_val = 0
                        new_def_val = 0
                    else:
                        new_att_val = FIGURE_VALUES_TO_ATTACK[cell.lower()]
                        new_def_val = -1
                        board_points[digit][letter]['figure_attack'] = cell
                        self.attackCalc.generateValueForField(cell.lower(), (idx_digit, idx_letter), self.is_white_move, board_points, enemy_figure_indexes)
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
                        self.deffenceCalc.generateValueForField(cell.lower(), (idx_digit, idx_letter), self.is_white_move, board_points, my_figure_indexes)
                    elif cell == '.':
                        # new_val = 0.01 * weight_of_field_black[idx_digit][idx_letter]
                        new_att_val = 0
                        new_def_val = 0
                    else:
                        new_att_val = FIGURE_VALUES_TO_ATTACK[cell.lower()]
                        new_def_val = -1
                        board_points[digit][letter]['figure_attack'] = cell
                        self.attackCalc.generateValueForField(cell.lower(), (idx_digit, idx_letter), self.is_white_move, board_points, enemy_figure_indexes)
                    board_points[digit][letter]['attack_figure_value'] = new_att_val
                    board_points[digit][letter]['deffence_figure_value'] = new_def_val


        return board_points

    # print(board)
        # if is_white_move:
        #     default_board = white_board
        # else:
        #     default_board = black_board
    
    def play_me(self, possible_move):

        legal_moves = list(self.board.legal_moves)
        if len(list(legal_moves)) == 0:
            return False
        generated_board_points, possible_moves, move_value, values_for_target_field = self.prepare_matrix_points(legal_moves)
        
        # most_quality_legal_move, legal_move_split, DEF_FIGURE = self.generate_and_choose_move(possible_moves, [], self.last_moves_black, values_for_target_field, move_value)
        
        generated_ai_possible_move = self.revertMove(possible_move)

        field_val_map, figure_def_val, figure_idx = (generated_ai_possible_move[0], generated_ai_possible_move[1], generated_ai_possible_move[2])
        # field_val = float(round(field_val, 2))
        # figure_val = float(round(figure_val, 2))
        # figure_idx = int(abs(round(figure_idx, 0)))
        if figure_def_val == 0.2:
            figure_def_val = 0.25
        print(f'!!!!game_move_nr {self.game_move_nr}', end='\n\n')
        print(f'!!!!white_move_nr {len(self.selected_moves)}',end='\n\n')
        # print(f'!!!!possible_move {possible_move}',end='\n\n')
        print(f'!!!!figure_def_val {figure_def_val} figure_idx {figure_idx}',end='\n\n')
        # print(f'!!!!field_val_map {field_val_map.tolist()}',end='\n\n')
        # print(f'!!!!values_for_target_field {values_for_target_field}',end='\n\n')

        field_val = 9999
        # while True:
        field_val_map[field_val_map < 0] = 0
        tmp_map = field_val_map.reshape(64)

        ind = np.argpartition(tmp_map, -15)[-15:]
        top5 = sorted(np.unique(tmp_map[ind]), reverse=True)
        selected_coords = None
        for most_val in top5:
            if not selected_coords is None: break

            Y_coords, X_cords = np.where(field_val_map == most_val)
            print(f"Y_coords {Y_coords}, X_cords{X_cords} - most_val {most_val}")
            for idx_cord, cord_y in enumerate(Y_coords):
                if self.current_board_for_y[cord_y][X_cords[idx_cord]] > 0:
                    field_val = self.current_board_for_y[cord_y][X_cords[idx_cord]]
                    selected_coords = f"cord_y {cord_y} cord_x {X_cords[idx_cord]}"
                    break

        # print(f'!!!!field_val_map {field_val_map}',end='\n\n')

        # print(f"current_board_for_y {self.current_board_for_y}")
        # print(f"selected field_val {field_val} selected_coords {selected_coords}")
        
        if field_val in values_for_target_field:
            most_val_moves = values_for_target_field[field_val]
            print(f'!!!!GREAT field_val {field_val}')
        else :
            most_val_moves = values_for_target_field[np.max(list(values_for_target_field.keys()))]

        # print(f'!!!!most_val_moves {most_val_moves}',end='\n\n')

            

        if self.is_white_move:
            # if self.nr_moves_white % 5 == 0:
            if figure_def_val in most_val_moves:
                min_figure = figure_def_val
                print(f'!!!!GREAT figure_def_val {figure_def_val}')

                # min_figure = np.random.choice(list(most_val_moves.keys()))
            else:
                min_figure = np.min(list(most_val_moves.keys()))
            
            self.nr_moves_white = self.nr_moves_white + 1


        legal_random_move_list = most_val_moves[min_figure]
        # print(f'!!!!legal_random_move_list {legal_random_move_list}')

        if figure_idx > len(legal_random_move_list) - 1:
            legal_random_move_idx = len(legal_random_move_list) - 1
        elif figure_idx < 0:
            legal_random_move_idx = 0
        else:
            legal_random_move_idx = figure_idx
            print(f'!!!!GREAT figure_idx {figure_idx}/{len(legal_random_move_list)}')
            # legal_random_move_idx = random.randint(0, len(legal_random_move_list) - 1)
        # print( 'len(legal_random_move_list)', len(legal_random_move_list))
        # print( 'legal_random_move_idx', legal_random_move_idx)

        legal_random_move = legal_random_move_list[int(legal_random_move_idx)]
        legal_move_split = (legal_random_move['legal_move'][0] + legal_random_move['legal_move'][1], legal_random_move['legal_move'][2] + legal_random_move['legal_move'][3])
        most_quality_legal_move = legal_random_move

        self.selectMove(legal_move_split, most_quality_legal_move, legal_random_move_idx)

        # print('last_moves', last_moves)


        column_move_difference = np.abs(LETTER_MAP.index(legal_move_split[1][0].upper()) - LETTER_MAP.index(legal_move_split[0][0].upper()))

        print('i=' + str(self.game_move_nr))
        # print('quality_me=' + str(quality_me))

        print(legal_move_split[0] + ' -> ' + legal_move_split[1])

        self.default_board = self.move(legal_move_split[0], legal_move_split[1], self.default_board)
        self.board.push_san(most_quality_legal_move['legal_move'] )

        print(self.board, end='\n\n')
        # print(default_board, end='\n\n')
        # print('DEF_FIGURE', DEF_FIGURE)
        DEF_FIGURE = most_quality_legal_move['def_figure']
        self.initPromotionFigureAndCastling(legal_move_split, DEF_FIGURE, column_move_difference)

        self.is_white_move = not self.is_white_move
        self.game_move_nr = self.game_move_nr + 1

        # display.update(board.fen())
        # ax.plot(predictedX, predictedY)
        self.tryShowUI()

            # sleep(1)
        return True
    
    def play_enemy(self):
                
        legal_moves = list(self.board.legal_moves)
        if len(list(legal_moves)) == 0:
            return False

        generated_board_points, possible_moves, move_value, values_for_target_field = self.prepare_matrix_points(legal_moves)
        
        most_quality_legal_move, legal_move_split, DEF_FIGURE, legal_random_move_idx = self.generate_and_choose_move(possible_moves, [], self.last_moves_black, values_for_target_field, move_value)
        
        # print(default_board, end='\n\n')
        # print(board, end='\n\n')


        column_move_difference = np.abs(LETTER_MAP.index(legal_move_split[1][0].upper()) - LETTER_MAP.index(legal_move_split[0][0].upper()))

        # print('column_move_difference=' + str(column_move_difference))
        # print('move_value=' + str(move_value))
        # print('i=' + str(self.game_move_nr))
        # print('quality_me=' + str(quality_me))
        # print('quality_enemy=' + str(quality_enemy))
        # print('move_value=' + str(move_value))
        # print('is_white_move=' + str(is_white_move))
        print(legal_move_split[0] + ' -> ' + legal_move_split[1])

        self.default_board = self.move(legal_move_split[0], legal_move_split[1], self.default_board)
        self.board.push_san(most_quality_legal_move['legal_move'] )

        print(self.board, end='\n\n')
        # print(default_board, end='\n\n')
        # print('DEF_FIGURE', DEF_FIGURE)


            # print('new_figure', self.default_board)

            # return True
        self.initPromotionFigureAndCastling(legal_move_split, DEF_FIGURE, column_move_difference)

        self.is_white_move = not self.is_white_move
        self.game_move_nr = self.game_move_nr + 1

        # display.update(board.fen())
        # ax.plot(predictedX, predictedY)
        self.tryShowUI()

            # sleep(1)
        return True
        


    def auto_play_chess(self):
        self.is_white_move = True
        values_for_target_field = {}
        last_moves = []
        last_moves_black = []
        self.history_boards = []
        self.selected_moves = []

        last_used_figures = []
        last_used_figures_black = []
        for _ in range(NR_MOVES):
            legal_moves = list(self.board.legal_moves)
            if len(list(legal_moves)) == 0:
                print('i=' + str(_))

                print(self.board, end='\n\n')
                if self.is_white_move:
                    print('WIN BLACK')
                    return False
                else:
                    print('WIN WHITE')
                    return True


            generated_board_points, possible_moves, move_value, values_for_target_field = self.prepare_matrix_points(legal_moves)
            
            most_quality_legal_move, legal_move_split, DEF_FIGURE, legal_random_move_idx = self.generate_and_choose_move(possible_moves, last_moves, last_moves_black, values_for_target_field, move_value)
            
            # print(default_board, end='\n\n')
            # print(board, end='\n\n')

            if self.is_white_move:
                # cell_nr = self.generate_cell_nr(legal_move_split[1][1], legal_move_split[1][0].upper()
                # self.current_board_for_y.append(most_quality_legal_move['figure_def_val'])
                # self.current_board_for_y.append(legal_random_move_idx)
                self.selectMove(legal_move_split, most_quality_legal_move, legal_random_move_idx)




            column_move_difference = np.abs(LETTER_MAP.index(legal_move_split[1][0].upper()) - LETTER_MAP.index(legal_move_split[0][0].upper()))

            # print('column_move_difference=' + str(column_move_difference))
            # print('move_value=' + str(move_value))
            # print('i=' + str(_))
            # print('quality_me=' + str(quality_me))
            # print('quality_enemy=' + str(quality_enemy))
            # print('move_value=' + str(move_value))
            # print('is_white_move=' + str(is_white_move))
            # print(legal_move_split[0] + ' -> ' + legal_move_split[1])

            self.default_board = self.move(legal_move_split[0], legal_move_split[1], default_board)
            self.board.push_san(most_quality_legal_move['legal_move'] )

            # print(default_board, end='\n\n')
            # print('DEF_FIGURE', DEF_FIGURE)

            self.initPromotionFigureAndCastling(legal_move_split, DEF_FIGURE, column_move_difference)

            self.is_white_move = not self.is_white_move
   
            # display.update(board.fen())
            # ax.plot(predictedX, predictedY)
            self.tryShowUI()

            # sleep(1)
        print('i=' + str(_))

        print(self.board, end='\n\n')

        return False
    
    def prepare_matrix_points(self, legal_moves):

        uniq_figures = np.array(list(FIGURE_VALUES_ENEMY.keys()))
        if self.is_white_move:
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

        clear_board_points = self.generate_clear_board_points()
        generated_board_points = self.generate_board_points_and_clear_board_field(clear_board_points, enemy_figure_indexes, my_figure_indexes)
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


        # print('is_white_move=' + str(self.is_white_move))
        # print('============== attack_figure ===================')
        # print (attack_figure, sep=' ')
        # print('============== deffence_figure ===================')
        # print (deffence_figure, sep=' ')
        # print('============== attack ===================')
        # print (attack, sep=' ')
        # print('============== deffence ===================')
        # print (deffence, sep=' ')
        
        attack_calc = attack_figure + attack
        deffence_calc = deffence_figure + deffence
        # print('============== attack_figure + attack ===================')
        # print (attack_calc)
        # print('============== deffence_figure + deffence ===================')
        # print (deffence_calc)

        calc_def_minus_att =  deffence_calc - attack_calc 
        # print('============== deffence_calc - attack_calc ===================')
        # print (calc_def_minus_att)
        
        calc_att_minusd_def = attack_calc - deffence_calc
        # print('============== attack_calc - deffence_calc ===================')
        # print (calc_att_minusd_def)  
        
        calc_white = attack_figure - attack
        # calc_white_t = - attack_figure + attack
        # print('============== attack_figure - attack ===================')
        # print (calc_white)        
        # print (calc_white_t)        

        # print('count legal_moves=' + str(len(list(legal_moves))))

        possible_moves = np.array(['o' for _ in range(64)])
        possible_moves = possible_moves.reshape(8,8)
        values_for_target_field = {}
        self.current_board_for_y = np.zeros(64).reshape(8,8)

        poss_moves = []
        # print(f'generated_board_points {generated_board_points} ')
        for legal_move in legal_moves:
            legal_move = str(legal_move)
            if len(legal_move) == 5 and not legal_move[4].upper() == 'Q':
                continue
            poss_moves.append(legal_move)
            # attack_figure_val = generated_board_points[legal_move[3]][legal_move[2].upper()]['attack_figure_value']
            # attack_val = generated_board_points[legal_move[3]][legal_move[2].upper()]['attack_value']
            # print(f'legal_move {legal_move} ')

            figure_def = generated_board_points[legal_move[1]][legal_move[0].upper()]['figure_deffence']

            targetY = DIGIT_MAP.index(legal_move[3])
            targetX = LETTER_MAP.index(legal_move[2].upper())

            figure_def_Val = FIGURE_VALUES_ENEMY[figure_def.lower()]
        # =======================================================
            # curr_val = attack_figure_val - (np.mean(attack_val)) - (figure_def_Val)
            # curr_val = attack_figure_val - attack_calc[targetY][targetX] - figure_def_Val
            # print(f"calc_white[targetY][targetX] {calc_white[targetY][targetX]}")

            curr_val = calc_white[targetY][targetX] - figure_def_Val + 3
            self.current_board_for_y[targetY][targetX] = curr_val
            # if self.is_white_move:
            # else:
            #     curr_val = calc_def_minus_att[targetY][targetX] - figure_def_Val

        # =======================================================
            curr_val = round(curr_val, 3)
            move_values[targetY][targetX].append(curr_val)

            if curr_val not in values_for_target_field.keys():
                values_for_target_field[curr_val] = {}
                
            if figure_def_Val not in values_for_target_field[curr_val].keys():
                values_for_target_field[curr_val][figure_def_Val] = []

            legalMoveDict = {
                'def_figure': figure_def,
                'legal_move': legal_move,
                'figure_def_val': figure_def_Val,
                'curr_val': curr_val,
            }
            values_for_target_field[curr_val][figure_def_Val].append(legalMoveDict)

            if curr_val > move_value:
                move_value = curr_val
        
        # print(f'possible_moves {poss_moves}')
        return generated_board_points, possible_moves, move_value, values_for_target_field


    def generate_and_choose_move(self, possible_moves, last_moves, last_moves_black, values_for_target_field, move_value):


        is_checkmate = self.board.is_checkmate()
        # print('==========================================possible_moves ==========================================' )
        # print(possible_moves )
        # print('is_checkmate=' + str(is_checkmate) )
        # print( values_for_target_field, end='\n\n')

        # for TEST 
        # if not is_white_move:
        #     values_for_target_field = {}
        #     values_for_target_field[move_value] = {}
        #     legalMoveDict = {
        #             'def_figure': 'k',
        #             'legal_move': 'e8d8',
        #         }
        #     values_for_target_field[move_value][0.9] = legalMoveDict
        # for TEST END
        # print (move_values, sep=' ')

        # print( values_for_target_field, end='\n\n')

        most_val_moves = values_for_target_field[move_value]

        if self.is_white_move:

            if self.nr_moves_white % 5 == 0:
                min_figure = np.random.choice(list(most_val_moves.keys()))
            else:
                min_figure = np.min(list(most_val_moves.keys()))
            
            self.nr_moves_white = self.nr_moves_white + 1
        else:
            min_figure = np.min(list(most_val_moves.keys()))
        # min_figure = most_val_moves[min_figure]
        # print( 'most_val_moves', most_val_moves, end='\n\n')
        # print( 'most_val_moves', min_figure)
        # print( 'most_val_moves.keys()', most_val_moves.keys())
        # print( 'min_figure', min_figure)
        # print( 'min_figure', most_val_moves[min_figure])


        # legal_random_move = np.random.choice(most_val_moves)
        legal_random_move_list = most_val_moves[min_figure]
        legal_random_move_idx = random.randint(0, len(legal_random_move_list) - 1)
        # print( 'len(legal_random_move_list)', len(legal_random_move_list))
        # print( 'legal_random_move_idx', legal_random_move_idx)

        legal_random_move = legal_random_move_list[legal_random_move_idx]

        legal_move_split = (legal_random_move['legal_move'][0] + legal_random_move['legal_move'][1], legal_random_move['legal_move'][2] + legal_random_move['legal_move'][3])

        most_quality_legal_move = legal_random_move

        most_quality_legal_move, legal_random_move_idx, legal_move_split = self.generate_second_quality_move(last_moves, legal_random_move, most_quality_legal_move, legal_move_split, last_moves_black, values_for_target_field, legal_random_move_idx)
        # print(last_moves)

        DEF_FIGURE = most_quality_legal_move['def_figure']

        # print('last_moves', last_moves)

        if self.is_white_move:
            self.history_boards.append(self.createX())

            last_moves.append(most_quality_legal_move['legal_move'])
            if len(last_moves) > 30:

                last_moves = last_moves[len(last_moves) - 30:]
            
        else :
            last_moves_black.append(most_quality_legal_move['legal_move'])
            if len(last_moves_black) > 30:

                last_moves_black = last_moves_black[len(last_moves_black) - 30:]

        
        return (most_quality_legal_move, legal_move_split, DEF_FIGURE, legal_random_move_idx)

    
    def generate_second_quality_move(self, last_moves, legal_random_move, most_quality_legal_move, legal_move_split, last_moves_black,  values_for_target_field, legal_random_move_idx):
        if self.is_white_move and last_moves.count(legal_random_move['legal_move']) > 3:
                move_from_loop = False
                for key in sorted(values_for_target_field.keys(), reverse=True):
                    
                    v = values_for_target_field[key]
                    for innerK in sorted(v.keys()):

                        vk = v[innerK]
                        for i, item in enumerate(vk):

                            if item['legal_move'] not in last_moves:
                                legal_random_move_idx = i
                                legal_random_move = item
                                move_from_loop = True
                                break
                        if move_from_loop:
                            break

                    if move_from_loop:
                        break

                if not move_from_loop:
                    legal_random_move = most_quality_legal_move

                legal_move_split = (legal_random_move['legal_move'][0] + legal_random_move['legal_move'][1], legal_random_move['legal_move'][2] + legal_random_move['legal_move'][3])

                most_quality_legal_move = legal_random_move
                print(legal_random_move)


        elif not self.is_white_move and last_moves_black.count(legal_random_move['legal_move']) > 3:
            move_from_loop = False
            for key in sorted(values_for_target_field.keys(), reverse=True):
                
                v = values_for_target_field[key]
                for innerK in sorted(v.keys()):

                    vk = v[innerK]
                    for i, item in enumerate(vk):

                        if item['legal_move'] not in last_moves_black:
                            legal_random_move_idx = i
                            legal_random_move = item
                            move_from_loop = True
                            break
                    if move_from_loop:
                        break

                if move_from_loop:
                    break
            

            if not move_from_loop:
                legal_random_move = most_quality_legal_move

            legal_move_split = (legal_random_move['legal_move'][0] + legal_random_move['legal_move'][1], legal_random_move['legal_move'][2] + legal_random_move['legal_move'][3])

            most_quality_legal_move = legal_random_move

        return (most_quality_legal_move, legal_random_move_idx, legal_move_split)

    def initPromotionFigureAndCastling(self, legal_move_split, DEF_FIGURE, column_move_difference):
        targetRow = legal_move_split[1][1]
        targetCol = legal_move_split[1][0]
        if (DEF_FIGURE == 'P' and self.is_white_move and targetRow == '8') or (DEF_FIGURE == 'p' and not self.is_white_move and targetRow == '1'):

            cell_nr = self.generate_cell_nr(targetRow, targetCol)
            # print('DIGIT_MAP', DIGIT_MAP)
            # print('digits', digits)
            # print('cell_nr', cell_nr)
            # print('new_figure', new_figure)
            new_figure = str(self.board.piece_at(cell_nr))

            if self.is_white_move :
                self.default_board[DIGIT_MAP.index(targetRow)][LETTER_MAP.index(targetCol.upper())] = new_figure.upper()
            else :
                self.default_board[DIGIT_MAP.index(targetRow)][LETTER_MAP.index(targetCol.upper())] = new_figure.lower()
        
        if column_move_difference > 1:
            if (DEF_FIGURE == 'K' and self.is_white_move and targetRow == '1' and targetCol == 'g' ) :
                if self.default_board[DIGIT_MAP.index('1')][LETTER_MAP.index('H')] == 'R':
                    self.move('h1', 'f1', default_board)
            elif (DEF_FIGURE == 'K' and self.is_white_move and targetRow == '1' and targetCol == 'c' ):
                if self.default_board[DIGIT_MAP.index('1')][LETTER_MAP.index('A')] == 'R':
                    self.move('a1', 'd1', default_board)
            elif (DEF_FIGURE == 'k' and not self.is_white_move and targetRow == '8' and targetCol == 'g' ) :
                if self.default_board[DIGIT_MAP.index('8')][LETTER_MAP.index('H')] == 'r':
                    self.move('h8', 'f8', default_board)
            elif (DEF_FIGURE == 'k' and not self.is_white_move and targetRow == '8' and targetCol == 'c' ):
                if self.default_board[DIGIT_MAP.index('8')][LETTER_MAP.index('A')] == 'r':
                    self.move('a8', 'd8', default_board)

    def generate_cell_nr(self, targetRow, targetCol):
        return (DIGIT_MAP_NOT_REV.index(targetRow)) * 8 + LETTER_MAP.index(targetCol.upper())

# {'8': {'A': {'attack_value': 0.3, 'defence_value': []}, 'B': {'attack_value': 0.25, 'defence_value': ['r']}, 'C': {'attack_value': 0.25, 'defence_value': ['q']}, 'D': {'attack_value': 0.6, 'defence_value': []}, 'E': {'attack_value': 0.9, 'defence_value': ['q']}, 'F': {'attack_value': 0.25, 'defence_value': []}, 'G': {'attack_value': 0.25, 'defence_value': ['r']}, 'H': {'attack_value': 0.3, 'defence_value': []}}, '7': {'A': {'attack_value': 0.1, 'defence_value': ['r']}, 'B': {'attack_value': 0.1, 'defence_value': ['b']}, 'C': {'attack_value': 0.1, 'defence_value': ['q']}, 'D': {'attack_value': 0.1, 'defence_value': ['b', 'q']}, 'E': {'attack_value': 0.1, 'defence_value': ['q', 'b']}, 'F': {'attack_value': 0.1, 'defence_value': []}, 'G': {'attack_value': 0.1, 'defence_value': ['b']}, 'H': {'attack_value': 0.1, 'defence_value': ['r']}}, '6': {'A': {'attack_value': 0.01, 'defence_value': ['r', 'n', 'b', 'r', 'b']}, 'B': {'attack_value': 0.01, 'defence_value': ['q']}, 'C': {'attack_value': 0.01, 'defence_value': ['n']}, 'D': {'attack_value': 0.01, 'defence_value': ['q', 'b', 'q']}, 'E': {'attack_value': 0.01, 'defence_value': ['b']}, 'F': {'attack_value': 0.01, 'defence_value': ['q', 'n', 'p']}, 'G': {'attack_value': 0.01, 'defence_value': []}, 'H': {'attack_value': 0.01, 'defence_value': ['b', 'n', 'r', 'b', 'r']}}, '5': {'A': {'attack_value': 0.01, 'defence_value': ['r', 'q', 'r']}, 'B': {'attack_value': 0.01, 'defence_value': ['b']}, 'C': {'attack_value': 0.01, 'defence_value': ['b']}, 'D': {'attack_value': 0.01, 'defence_value': ['q', 'q']}, 'E': {'attack_value': 0.01, 'defence_value': []}, 'F': {'attack_value': 0.01, 'defence_value': ['b']}, 'G': {'attack_value': 0.01, 'defence_value': ['q', 'b']}, 'H': {'attack_value': 0.01, 'defence_value': ['r', 'q', 'r']}}, '4': {'A': {'attack_value': 0.01, 'defence_value': ['r', 'r', 'q']}, 'B': {'attack_value': 0.01, 'defence_value': ['b']}, 'C': {'attack_value': 0.01, 'defence_value': ['b']}, 'D': {'attack_value': 0.01, 'defence_value': ['q', 'q']}, 'E': {'attack_value': 0.01, 'defence_value': []}, 'F': {'attack_value': 0.01, 'defence_value': ['b']}, 'G': {'attack_value': 0.01, 'defence_value': ['b', 'q']}, 'H': {'attack_value': 0.01, 'defence_value': ['q', 'r', 'r']}}, '3': {'A': {'attack_value': 0.01, 'defence_value': ['r', 'b', 'r', 'n', 'b']}, 'B': {'attack_value': 0.01, 'defence_value': ['q']}, 'C': {'attack_value': 0.01, 'defence_value': ['n']}, 'D': {'attack_value': 0.01, 'defence_value': ['q', 'q', 'b']}, 'E': {'attack_value': 0.01, 'defence_value': ['b']}, 'F': {'attack_value': 0.01, 'defence_value': ['p', 'q', 'n']}, 'G': {'attack_value': 0.01, 'defence_value': []}, 'H': {'attack_value': 0.01, 'defence_value': ['b', 'r', 'b', 'n', 'r']}}, '2': {'A': {'attack_value': -1, 'defence_value': ['r', 'r']}, 'B': {'attack_value': -1, 'defence_value': ['b']}, 'C': {'attack_value': -1, 'defence_value': ['q']}, 'D': {'attack_value': -1, 'defence_value': ['q', 'n', 'b', 'q']}, 'E': {'attack_value': -1, 'defence_value': ['q', 'b', 'n']}, 'F': {'attack_value': -1, 'defence_value': []}, 'G': {'attack_value': -1, 'defence_value': ['b']}, 'H': {'attack_value': -1, 'defence_value': ['r', 'r']}}, '1': {'A': {'attack_value': -1, 'defence_value': ['r', 'q', 'r']}, 'B': {'attack_value': -1, 'defence_value': ['r', 'q', 'r']}, 'C': {'attack_value': -1, 'defence_value': ['r', 'q', 'r']}, 'D': {'attack_value': -1, 'defence_value': ['q', 'r', 'r']}, 'E': {'attack_value': -1, 'defence_value': ['r', 'q', 'r']}, 'F': {'attack_value': -1, 'defence_value': ['p', 'r', 'q', 'r']}, 'G': {'attack_value': -1, 'defence_value': ['r', 'q', 'r']}, 'H': {'attack_value': -1, 'defence_value': ['r', 'r', 'q']}}}

        
# board.is_checkmate()
# board.find_move(('h1'), ('h3

# svg2png(bytestring=svg_text, write_to='example-board.png')
# sleep(100)
    

# game = ChessGame(True)

# game.auto_play_chess()
# print(np.array(game.default_board), end='\n\n')

# print(game.board.piece_at(16))
# for el in game.board.pieces():
#     # print(game.board.piece_map()[el])
#     print(el)