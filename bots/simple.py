import chess
import chess.pgn
import numpy as np
import time
import random

class chessBot(object):
    def makeMove(self, board, moves, verbose):
        pass
    def evalPos(self, board):
        return random.random()

class aggroBot(chessBot):

    def makeMove(self, board, moves, verbose):
        zeroing  = []
        others   = []
        checks   = []
        captures = []
        for move in moves:
            if board.is_capture(move):
                captures.append(move)
            elif board.is_zeroing(move): # favours captures and pawn moves
                zeroing.append(move)
            elif self.checking(board, move):
                checks.append(move)
            else:
                others.append(move)

        random.shuffle(captures)
        random.shuffle(zeroing)
        random.shuffle(checks)
        random.shuffle(others)
        move_types = [captures, zeroing, checks, others]
        probabilites = [0.5, 0.30, 0.15, 0.05]
        move_choice = np.random.choice([0,1,2,3],4,False,probabilites)
        move_types = [move_types[choice] for choice in move_choice]
        return list(np.concatenate(move_types))

    def checking(self, board, move):
        board.push(move)
        check = board.is_check()
        board.pop()
        return check

    def evalPos(self, board):
        super().evalPos(self, board)

class randomBot(chessBot):

    def makeMove(self, board, moves, verbose):
        moves = list(moves)
        random.shuffle(moves)
        return moves

    def evalPos(self, board):
        super().evalPos(self, board)

# A bot that favours moves that start a piece type of low rank
# Goes pawn -> knight -> bishop -> rook -> queen -> king
class lowRankBot(chessBot):

    def makeMove(self, board, moves, verbose):
        moves = list(moves)
        random.shuffle(moves)

        def pieceTypeOfMove(move):
            return board.piece_at(move.from_square).piece_type
        moves.sort(key = pieceTypeOfMove)

        return moves

    def evalPos(self, board):
        super().evalPos(self, board)

class jaqueBot(chessBot):
    def __init__(self):
        self.piece_values = [0, 1, 3, 3, 5, 8, 1000]

    def makeMove(self, board, moves, verbose):
        moves = list(moves)
        boards = []
        for move in moves:
            move_piece = board.piece_type_at(move.from_square)
            move_piece_value = self.piece_values[move_piece]
            if board.is_capture(move):
                freebie = not board.is_attacked_by(not board.turn, move.to_square)
                if freebie:
                    if verbose:
                        print('I made move {} because I could capture without retaliation'.format(move))
                    return [move]
        for move in moves:
            move_piece = board.piece_type_at(move.from_square)
            move_piece_value = self.piece_values[move_piece]
            if board.is_capture(move):

                if board.is_en_passant(move):
                    eigth = -8 if board.turn else 8
                    capture_square = move.to_square + eigth
                    capture_piece = board.piece_type_at(capture_square)
                else:
                    capture_piece = board.piece_type_at(move.to_square)
                to_value = self.piece_values[capture_piece]
                if move_piece_value < to_value:
                    if verbose:
                        print('I made move {} because the piece I attacked with was worth less than the one it took'.format(move))
                    return [move]
        for move in moves:
            move_piece = board.piece_type_at(move.from_square)
            move_piece_value = self.piece_values[move_piece]
            if len(board.attackers(board.turn, move.to_square)) > 1 and len(board.attackers(board.turn, move.to_square)) > 0:
                lowest_enemy_value = 1001
                for enemy in board.attackers(not board.turn, move.to_square):
                    if self.piece_values[board.piece_type_at(enemy)] < lowest_enemy_value:
                        lowest_enemy_value = self.piece_values[board.piece_type_at(enemy)]
                if lowest_enemy_value > move_piece_value:
                    if verbose:
                        print('I made move {} because it baits a more valuable piece to attack'.format(move))
                    return [move]
        random.shuffle(moves)
        if verbose:
            print('I randomly chose move {}'.format(moves[0]))
        return moves

    def evalPos(self, board):
        super().evalPos(self, board)
