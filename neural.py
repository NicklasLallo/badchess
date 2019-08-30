import torch
import torch.nn as nn
import torch.utils.data as data
import chess
import chessUtils
import numpy as np
import time
import random
from bots.simple import chessBot
from bots.minimax import suicideBot, minimax
from chessMaster import chessMaster

def matchesToTensor(boards):
    '''
    Takes a list of Board and returns a Tensor of all boardstates and a Tensor with results
    '''
    num_boardstates = sum([len(board.move_stack) for board in boards])
    boardstatesTensor = torch.zeros([num_boardstates, 65])
    resultsTensor = torch.zeros([num_boardstates])
    current_board = 0
    for i, board in enumerate(boards):
        r = board.result()
        if r == "0-1":
            r = 0.0
        elif r == "1-0":
            r = 1.0
        else:
            r = 0.5
        while len(board.move_stack) > 0:
            boardstatesTensor[current_board, 0] = float(board.turn)
            for square in range(0,64):
                if not board.piece_type_at(square) is None:
                    piece = board.piece_type_at(square)/6
                    if not board.color_at(square):
                        piece = -piece
                    boardstatesTensor[current_board,square+1] = piece
            resultsTensor[current_board] = r
            current_board += 1
            board.pop()
    return boardstatesTensor, resultsTensor

model = nn.Sequential(
    nn.Linear(65, 65),
    nn.LeakyReLU(),
    nn.Linear(65, 1)
)

class NeuralBot(chessBot):
    def __init__(self, model=None, gpu=False):
        self.model = model
        self.gpu = gpu
        if self.model is None:
            self.model = nn.Sequential(
                nn.Linear(65, 65),
                nn.LeakyReLU(),
                nn.Linear(65, 1)
            )
        elif isinstance(self.model,str):
            self.model = torch.load(model)
        if self.gpu:
            self.model.cuda()

    def evalPos(self, board):
        board = board.copy()
        m = board.pop()
        board.clear_stack()
        board.push(m)
        boardTensor, _ = matchesToTensor([board])
        boardTensor = boardTensor[0].unsqueeze(0)
        if self.gpu:
            boardTensor = boardTensor.cuda()
        value = self.model(boardTensor)
        value = value.item()
        if not board.turn:
            value = 1-value
        return value

    def decideMove(self, board, moves):
        moves = list(moves)
        boards = []
        for move in moves:
            b = board.copy(stack=False)
            b.push(move)
            boards.append(b)
        boardsTensor, _ = matchesToTensor(boards)
        if self.gpu:
            boardsTensor = boardsTensor.cuda()
        value = self.model(boardsTensor).view(-1)
        if not board.turn:
            value = 1-value
        index = value.argmax().item()
        return [moves[index]]
        
    def makeMove(self, board, moves):
        return self.decideMove(board, moves)
        #value, move = minimax(board, self.evalPos, depth=1)
        #return [move]


if __name__ == "__main__":
    LOAD_FILE = None #"bad_neural_net.pt"

    EPOCHS = 1000
    GAMES = 100
    BATCH_SIZE = 1000
    PLAYER = NeuralBot(model=LOAD_FILE, gpu=torch.cuda.is_available())
    OPPONENT = suicideBot()
    optimizer = torch.optim.Adam(PLAYER.model.parameters())
    loss_fun = nn.MSELoss()
    for epoch in range(EPOCHS):
        new_games = []
        chessUtils.printProgressBar(0, GAMES, "Playing games", suffix = 'Complete', length = 20)
        for game in range(GAMES):
            with torch.no_grad():
                game1 = chessMaster(PLAYER,OPPONENT).board
                game2 = chessMaster(OPPONENT,PLAYER).board
                new_games.append(game1)
                new_games.append(game2)
                chessUtils.printProgressBar(game + 1, GAMES, "Playing games", suffix = 'Complete', length = 20)
        matchesTensor, resultsTensor = matchesToTensor(new_games)
        dataset = data.TensorDataset(matchesTensor, resultsTensor)
        dataloader = data.DataLoader(dataset, BATCH_SIZE, True)
        epoch_loss = 0
        for i, (inputs, labels) in enumerate(dataloader):
            if PLAYER.gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
            preds = PLAYER.model(inputs)
            loss = loss_fun(preds, labels)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print("\rEpoch {} [{}/{}] - Loss {}".format(epoch, i, len(dataloader), loss.item()),end="")
        torch.save(PLAYER.model, "bad_neural_net.pt")
        print("\rEpoch {} - Loss {}".format(epoch, epoch_loss))