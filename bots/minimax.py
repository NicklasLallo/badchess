import chess
import chess.pgn
import numpy as np
import time
import random
from bots.simple import chessBot

def minimax(board, eval_fun, this_prio_fun=max, next_prio_fun=min, depth=2, alfabeta=None):
    
    if depth == 0:
        return eval_fun(board), None

    moves = list(board.legal_moves)
    random.shuffle(moves)
    best_value = -this_prio_fun(-99999,99999)
    best_move = None
    if alfabeta is None:
        alfabeta = -best_value
    for move in moves:
        board.push(move)
        move_value, _ = minimax(
            board,
            eval_fun, 
            next_prio_fun,
            this_prio_fun,
            depth-1,
            best_value
        )
        board.pop()
        new_value = this_prio_fun(best_value, move_value)
        if best_value != new_value:
            best_value = new_value
            best_move = move
            if new_value == this_prio_fun(new_value, alfabeta):
                break
    return best_value, best_move

def naive_eval_fun(board, is_white):
    piece_values = [0, 1, 3, 3, 5, 8, 1000]
    value = 0
    for square in range(64):
        if not board.piece_type_at(square) is None:
            v = piece_values[board.piece_type_at(square)]
            if board.color_at(square) != is_white:
                v = -v
            value = value + v
    return value

class naiveMinimaxBot(chessBot):
    def __init__(self):
        self.value = 0

    def makeMove(self, board, moves):
        my_color = board.turn
        self.value, move = minimax(board, lambda x: naive_eval_fun(x, my_color))
        return [move]

    def evalPos(self, board):
        return self.value