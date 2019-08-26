import chess
import chess.pgn
import numpy as np
import time
import random

class chessBot(object):
    def makeMove(self, board, moves):
        pass
    def evalPos(self, board):
        return random.random()

class aggroBot(chessBot):

    def makeMove(self, board, moves):
        zeroing = []
        others  = []
        checks  = []
        for move in moves:
            if board.is_zeroing(move): # favours captures and pawn moves
                zeroing.append(move)
            elif self.checking(board, move):
                checks.append(move)
            else:
                others.append(move)

        random.shuffle(zeroing)
        random.shuffle(checks)
        random.shuffle(others)
        return zeroing + checks + others

    def checking(self, board, move):
        board.push(move)
        check = board.is_check()
        board.pop()
        return check

    def evalPos(self, board):
        super().evalPos(self, board)

class randomBot(chessBot):

    def makeMove(self, board, moves):
        moves = list(moves)
        random.shuffle(moves)
        return moves

    def evalPos(self, board):
        super().evalPos(self, board)

# A bot that favours moves that start a piece type of low rank
# Goes pawn -> knight -> bishop -> rook -> queen -> king
class lowRankBot(chessBot):

    def makeMove(self, board, moves):
        moves = list(moves)
        random.shuffle(moves)
        
        def pieceTypeOfMove(move):
            return board.piece_at(move.from_square).piece_type
        moves.sort(key = pieceTypeOfMove)

        return moves
    
    def evalPos(self, board):
        super().evalPos(self, board)
