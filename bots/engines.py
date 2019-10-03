import asyncio
import chess
import chess.pgn
import chess.engine
import numpy as np
import time
import random
from bots.simple import chessBot


class stockfish(chessBot):
    def __init__(self, time=0.100):
        self.engine = chess.engine.SimpleEngine.popen_uci("/usr/local/bin/stockfish")
        self.time = time
        # self.engine = await chess.engine.SimpleEngine.popen_uci("/usr/local/bin/stockfish")
        # asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())

    def quit(self):
        self.engine.quit()

    def makeMove(self, board, moves, verbose):
        result = self.engine.play(board, chess.engine.Limit(time=self.time))
        # board.push(result.move)
        return [result.move]

    def evalPos(self, board):
        info = self.engine.analyse(board, chess.engine.Limit(time=0.200))
        return int(info["score"])
