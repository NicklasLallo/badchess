import asyncio
import chess
import chess.pgn
import chess.engine
import numpy as np
from time import sleep
import random
from bots.simple import chessBot


class stockfish(chessBot):
    def __init__(self, time=0.100, depth=None):
        self.engine = chess.engine.SimpleEngine.popen_uci("/usr/local/bin/stockfish")
        sleep(0.10) # Give the stockfish engine time to start properly
        self.time = time
        self.depth=depth
        # self.engine = await chess.engine.SimpleEngine.popen_uci("/usr/local/bin/stockfish")
        # asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())

    def quit(self):
        self.engine.quit()
        del self

    def makeMove(self, board, moves, verbose):
        result = self.engine.play(board, chess.engine.Limit(time=self.time, depth=self.depth))
        # board.push(result.move)
        return [result.move]

    def evalPos(self, board): # points from white's point of view
        info = self.engine.analyse(board, chess.engine.Limit(time=self.time))
        score = info["score"].white().score(mate_score=2000)
        return score

class stochfish(stockfish):
    # Stochastic stockfish
    # non determenistic stockfish for generating training data
    # Plays like stockfish but makes random moves sometimes
    # favoures random moves early and then gets better and better
    # the longer the game goes

    def __init__(self, time=0.100, depth=None, noise=0.400, noise_decay=0.05):
        super().__init__(time, depth)
        self.noise = noise
        self.noise_decay = noise_decay

    def quit(self):
        super().quit()

    def evalPos(self, board):
        return super().evalPos(board)

    def makeMove(self, board, moves, verbose):
        goodfish = random.random()
        if goodfish < self.noise:
            # make random move
            moves = list(moves)
            random.shuffle(moves)
            # After a random move the noise can decay faster since we are now
            # set on a non deterministic path
            self.noise = self.noise * (1 - 2 * self.noise_decay)
        else:
            result = self.engine.play(board, chess.engine.Limit(time=self.time, depth=self.depth))
            moves = [result.move]
            # Make noise decay at the rate of noise_decay
            # Good moves still decay the noise but slower
            self.noise = self.noise * (1 - self.noise_decay)
        return moves


