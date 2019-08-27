import chess
import chess.pgn
import numpy as np
import time
import random
from bots import simple, minimax
import operator
import chessUtils

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

    def winner(self):
        if self.board.result()=='1/2-1/2':
            return (0.5,0.5)
        elif self.board.result()=='1-0':
            return (1,0)
        else:
            return (0,1)

def sampleGames(agentA, agentB, chessVariant='Standard'):
    results=(0,0)
    sampleSize=100
    prefix = "Playing "+str(sampleSize)+" games: "
    chessUtils.printProgressBar(0, sampleSize, prefix, suffix = 'Complete', length = 20)
    for i in range(0,sampleSize):
        game = chessMaster(agentA, agentB, chessVariant)
        results = tuple(map(operator.add, results, game.winner()))
        # Update Progress Bar
        chessUtils.printProgressBar(i + 1, sampleSize, prefix, suffix = 'Complete', length = 20)
    print(type(agentA).__name__ +": "+str(results[0])+"%")
    print(type(agentB).__name__ +": "+str(results[1])+"%")


if __name__ == "__main__":

    # Current bots:
    # simple: randomBot, aggroBot, lowRankBot

    bot1 = simple.aggroBot()
    bot2 = simple.lowRankBot()
    bot3 = minimax.naiveMinimaxBot()

    game = chessMaster(bot1, bot3)

    print(game.output())
    sampleGames(bot1, bot3)
    sampleGames(bot2, bot3)
