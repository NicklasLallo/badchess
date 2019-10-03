import chess
import chess.pgn
import numpy as np
import time
import random
from bots import simple, minimax#, engines
import operator
import chessUtils
import json
import pandas
from ast import literal_eval as make_tuple
import neural
import multiprocessing
from queue import Empty as EmptyException
import signal
import pickle

class chessMaster:

    def __init__(self, agentA, agentB, log=True, chessVariant='Standard', nineSixty=False, verbose=False):

        if chessVariant == 'Standard':
            self.board = chess.Board(chess960=nineSixty)
        elif chessVariant == 'Suicide':
            self.board = chess.variant.SuicideBoard(chess960=nineSixty)
        elif chessVariant == 'Giveaway':
            self.board = chess.variant.GiveawayBoard(chess960=nineSixty)
        elif chessVariant == 'Atomic':
            self.board = chess.variant.Atomic(chess960=nineSixty)
        elif chessVariant == 'King of the Hill':
            self.board = chess.variant.KingOfTheHillBoard(chess960=nineSixty)
        elif chessVariant == 'Racing Kings':
            self.board = chess.variant.RacingKingsBoard(chess960=nineSixty)
        elif chessVariant == 'Horde':
            self.board = chess.variant.HordeBoard(chess960=nineSixty)
        elif chessVariant == 'Three-check':
            self.board = chess.variant.ThreeCheckBoard(chess960=nineSixty)
        elif chessVariant == 'Crazyhouse':
            self.board = chess.variant.CrazyhouseBoard(chess960=nineSixty)
        else:
            self.board = chess.Board(chess960=nineSixty)
            print('ChessVariant missmatch')

        # gamelog = chess.pgn.Game()
        # gamelog.headers["Event"] = "Bot Championships Alpha"

        self.play(agentA, agentB, verbose)
        if log:
            with open('previousGame.pgn', 'w') as f:
                gamelog = chess.pgn.Game.from_board(self.board)
                gamelog.headers["Event"] = "Bot Championships Alpha"
                gamelog.headers["White"] = str(type(agentA).__name__)
                gamelog.headers["Black"] = str(type(agentB).__name__)
                f.write(str(gamelog))
                f.close()

    def play(self, agentA, agentB, verbose):
        active = True
        while(not self.board.is_game_over()):
            if verbose:
                print(str(self.board))
            if active:
                movelist = agentA.makeMove(self.board, self.board.legal_moves, verbose)
            else:
                movelist = agentB.makeMove(self.board, self.board.legal_moves, verbose)
            active = not active

            #todo logic
            if len(movelist) == 0:
                print("something wrong here")
            # elif i > 500:
            # print("long game: ", i)
            for move in movelist:
                if move in self.board.legal_moves:
                    self.board.push(move)
                    if verbose:
                        print("move made:", str(move))
                    break

    def output(self):
        return str(self.board)+'\n\n'+self.board.result()

    def winner(self):
        if self.board.result()=='1/2-1/2':
            return (0,0,1)
        elif self.board.result()=='1-0':
            return (1,0,0)
        else:
            return (0,1,0)

def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def zaveGamez( boards, file_name):
    pickle.dump(boards, open(file_name, 'wb'))

def zloadGamez( boards, file_name):
    return pickle.load(open(file_name, 'rb'))

def playSingleGames(agentA, agentB, num_games, workers=2, chessVariant='Standard', display_progress=False, log=True, save=False):
    games = []
    prefix = "Playing "+str(num_games)+" games: "
    chessUtils.printProgressBar(0, num_games, prefix, suffix = 'Complete', length = 20)
    try:
        for i in range(num_games):
            chessUtils.printProgressBar(i+1, num_games, prefix, suffix = 'Complete', length = 20)
            games.append(chessMaster(agentA, agentB, log, chessVariant))
    except KeyboardInterrupt:
        pass
    if save:
        zaveGamez(games, 'games.pickle')
    return games

def playMultipleGames(agentA, agentB, num_games, workers=2, chessVariant='Standard', display_progress=False, log=False, save=False):
    pool = multiprocessing.Pool(workers, init_worker)
    try:
        processes = [pool.apply_async(chessMaster, (agentA, agentB, log, chessVariant)) for _ in range(num_games)]
        if display_progress:
            prefix = "Playing "+str(num_games)+" games: "
            chessUtils.printProgressBar(0, num_games, prefix, suffix = 'Complete', length = 20)
        games = []
        for i, process in enumerate(processes):
            # print("process ",i)
            games.append(process.get())
            if display_progress:
                chessUtils.printProgressBar(i+1, num_games, prefix, suffix = 'Complete', length = 20)
    except KeyboardInterrupt:
        pool.close()
        pool.terminate()
        pool.join()
        raise KeyboardInterrupt
    else:
        pool.close()
        pool.join()
    if save:
        zaveGamez(games, 'games.pickle')
    return games

def sampleGames(agentA, agentB, chessVariant='Standard', workers=2, parallel=True, sampleSize=100):
    results=(0,0,0)
    prefix = "Playing "+str(sampleSize)+" games: "
    if not parallel:
        games = playSingleGames(agentA, agentB, sampleSize, workers, chessVariant, False)
    else:
        games = playMultipleGames(agentA, agentB, sampleSize, workers, chessVariant, False)
    for game in games:
        results = tuple(map(operator.add, results, game.winner()))
    saveToJSON(agentA, agentB, resultA=results)
    print(type(agentA).__name__ +": "+str(results[0])+"%")
    print(type(agentB).__name__ +": "+str(results[1])+"%")
    print("draws: "+str(results[2])+"%")

def saveToJSON(agentA, agentB, datafile='database.json', resultA=(0,0,0)):
    # with open(datafile, 'r+') as f:
    df = pandas.read_json(datafile)
    aName = type(agentA).__name__
    bName = type(agentB).__name__

    if not aName in df:
        df.insert(0, aName, "(0,0,0)")
    if not bName in df:
        df.insert(0, bName, "(0,0,0)")

    labels = df.index.union(df.columns)
    df = df.reindex(index=labels, columns=labels, fill_value="(0,0,0)")

    currentA = make_tuple(df[aName][bName])
    currentB = make_tuple(df[bName][aName])

    currentA = tuple(map(operator.add, resultA, currentA))
    resultB  = (resultA[1], resultA[0], resultA[2])
    currentB = tuple(map(operator.add, resultB, currentB))

    df[aName][bName] = str(currentA)
    df[bName][aName] = str(currentB)

    df.to_json(datafile)

if __name__ == "__main__":

    # Current bots:
    # simple: randomBot, aggroBot, lowRankBot
    # minimax: naiveMinimaxBot, arrogantBot, suicideBot, suicideMirrorBot

    bot0 = simple.randomBot()
    bot1 = simple.aggroBot()
    bot2 = simple.lowRankBot()
    bot3 = minimax.naiveMinimaxBot()
    bot4 = simple.jaqueBot()

    nbot1 = neural.NeuralBoardValueBot(model="bad_neural_net.pt", gpu=False)
    nbot2 = neural.NeuralMoveInstructionBot(model="instruction_neural_net.pt", gpu=True)
    nbvbMedium = neural.NeuralBoardValueBot(model="not_as_bad_neural_net.pt", gpu=True)
    nbvbLarge = neural.NeuralBoardValueBot(model="not_as_bad_neural_net_large.pt", gpu=True)
    nnibExtraLarge = neural.NeuralMoveInstructionBot(model="instruction_neural_net_extralarge.pt", gpu=True)

    "shlockfish = engines.stockfish(time=0.01)"

    playSingleGames(bot0, bot0, 100000, log=False, save=True)

    #game = chessMaster(nnibExtraLarge, bot0, verbose=True)
    # for i in range(100000000):
    #     game = chessMaster(bot1, bot3)
    #     if game.winner() == (1,0,0):
    #         break
    #     print("\r{}".format(i), end="")
    #print(game.output())
    #sampleGames(nbvbMedium, bot0, workers=2, parallel=False)
    # sampleGames(minimax.arrogantBot(), simple.randomBot())
    # sampleGames(simple.randomBot(), bot3)
    # sampleGames(simple.randomBot(), bot1)
