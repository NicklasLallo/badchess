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
# from chessMaster import chessMaster
import chessMaster

def boardstateToTensor(board):
    boardstateTensor = torch.zeros([65])
    boardstateTensor[0] = float(board.turn)
    for square in range(0,64):
        if not board.piece_type_at(square) is None:
            piece = board.piece_type_at(square)/6
            if not board.color_at(square):
                piece = -piece
            boardstateTensor[square+1] = piece
    return boardstateTensor

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
            boardstatesTensor[current_board,:] = boardstateToTensor(board)
            resultsTensor[current_board] = r
            current_board += 1
            board.pop()
    return boardstatesTensor, resultsTensor



class NeuralBot(chessBot):
    # Abstract class for NN bots that take a 65 dimensional boardstate as input
    def __init__(self, model=None, gpu=False, out_size=1):
        self.model = model
        self.gpu = gpu
        if self.model is None:
            self.model = nn.Sequential(
                nn.Linear(65, 65),
                nn.LeakyReLU(),
                nn.Linear(65, out_size)
            )
        elif isinstance(self.model,str):
            self.model = torch.load(model, map_location='cpu')
        if self.gpu:
            self.model.cuda()

    def evalPos(self, board):
        pass
    def makeMove(self, board, moves):
        pass

class NeuralBoardValueBot(NeuralBot):
    def __init__(self, model=None, gpu=False):
        super().__init__(model, gpu, 1)

    def evalPos(self, board):
        boardTensor = boardstateToTensor(board).unsqueeze(0)
        if self.gpu:
            boardTensor = boardTensor.cuda()
        value = self.model(boardTensor)
        value = value.item()
        if not board.turn:
            value = 1-value
        return value

    def makeMove(self, board, moves):
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
        if not board.turn: # add Not to attempt to win instead of lose
            value = 1-value
        index = value.argmax().item()
        return [moves[index]]

class NeuralMoveCategorizerBot(NeuralBot):
    def __init__(self, model=None, gpu=False):
        super().__init__(model, gpu, None) #TODO: Change 'None' for #pieces+#move_categories 
    
    def makeMove(self, board, moves):
        boardTensor = boardstateToTensor(board).unsqueeze()
        pieces_and_move_types = self.model(boardTensor)
        #TODO: extract network output and 
        return None

    def evalPos(self, board):
        return 0.5

if __name__ == "__main__":
    LOAD_FILE = "bad_neural_net.pt" # None #"bad_neural_net.pt"

    EPOCHS = 100
    GAMES = 100
    BATCH_SIZE = 1000
    PLAYER = NeuralBoardValueBot(model=LOAD_FILE, gpu=False)
    GPU = torch.cuda.is_available()
    OPPONENT = suicideBot()
    MAX_TRAIN_GAMES = 850

    optimizer = torch.optim.Adam(PLAYER.model.parameters())
    loss_fun = nn.MSELoss()
    games = []
    for epoch in range(EPOCHS):
        new_games = chessMaster.playMultipleGames(PLAYER, PLAYER, GAMES, workers=4, display_progress=True)
        new_games = [game.board for game in new_games]
        games = new_games + games
        games = games[:min(MAX_TRAIN_GAMES, len(new_games))]
        matchesTensor, resultsTensor = matchesToTensor(games)
        dataset = data.TensorDataset(matchesTensor, resultsTensor)
        dataloader = data.DataLoader(dataset, BATCH_SIZE, True)
        epoch_loss = 0
        if GPU:
            PLAYER.model.cuda()
        for i, (inputs, labels) in enumerate(dataloader):
            if GPU:
                inputs, labels = inputs.cuda(), labels.cuda()
            preds = PLAYER.model(inputs).view(-1)
            loss = loss_fun(preds, labels)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print("\rEpoch {} [{}/{}] - Loss {}".format(epoch, i, len(dataloader), loss.item()),end="")
        PLAYER.model.cpu()
        torch.save(PLAYER.model, "bad_neural_net.pt")
        print("\rEpoch {} - Loss {}".format(epoch, epoch_loss))