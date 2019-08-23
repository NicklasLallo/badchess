import chess
import chess.pgn
import numpy as np
import time
import random

class chessMaster:

    def __init__(self, agentA, agentB, chessVariant='Standard'):

        if not chessVariant == 'Standard':
            pass
        else:
            self.board = chess.Board()

        # gamelog = chess.pgn.Game()
        # gamelog.headers["Event"] = "Bot Championships Alpha"

        with open('previousGame.pgn', 'w') as f:
            active = True
            while(not self.board.is_game_over()):
                if active:
                    movelist = agentA.makeMove(self.board, self.board.legal_moves)
                else:
                    movelist = agentB.makeMove(self.board, self.board.legal_moves)
                active = not active


                #todo logic
                for move in movelist:
                    if move in self.board.legal_moves:
                        self.board.push(move)
                        break


                # gamelog.add_variation(self.board.peek())
                # f.write(self.board.peek().uci() + '\n')

            # f.write(str(gamelog))
            gamelog = chess.pgn.Game.from_board(self.board)
            gamelog.headers["Event"] = "Bot Champtionships Alpha"
            f.write(str(gamelog))
            # f.write(self.board.result())
            f.close()


    def output(self):
        return str(self.board)+'\n\n'+self.board.result()


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

if __name__ == "__main__":

    bot2 = aggroBot()
    bot1 = randomBot()

    game = chessMaster(bot1, bot2)

    print(game.output())
