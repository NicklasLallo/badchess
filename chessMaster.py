import chess
import chess.pgn
import numpy as np
import time
import random
import bots.simple as simple

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


if __name__ == "__main__":

    bot1 = simple.randomBot()
    bot2 = simple.lowRankBot()

    game = chessMaster(bot1, bot2)

    print(game.output())
