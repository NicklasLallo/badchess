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

def whiteWinnerLabeller(board, outcome):
    if outcome == "0-1":
        outcome = 0.0
    elif outcome == "1-0":
        outcome = 1.0
    else:
        outcome = 0.5
    return [outcome]

def matchesToTensor(boards, label_fun, out_size):
    '''
    Takes a list of Board and returns a Tensor of all boardstates and a Tensor with labels
    label_fun takes a board and the outcome of the game and returns a label for that board
    The label is a list of expected outputs for the neurons of the final layer
    out_size is the size of the label
    '''
    num_boardstates = sum([len(board.move_stack) for board in boards])
    boardstatesTensor = torch.zeros([num_boardstates, 65])
    resultsTensor = torch.zeros([num_boardstates, out_size])
    current_board = 0
    for i, board in enumerate(boards):
        board = board.copy()
        outcome = board.result()
        while len(board.move_stack) > 0:
            boardstatesTensor[current_board,:] = boardstateToTensor(board)
            resultsTensor[current_board, :] = torch.Tensor(label_fun(board, outcome))
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
        boardsTensor, _ = matchesToTensor(boards, whiteWinnerLabeller, 1)
        if self.gpu:
            boardsTensor = boardsTensor.cuda()
        value = self.model(boardsTensor).view(-1)
        if not board.turn: # add Not to attempt to win instead of lose
            value = 1-value
        print(value)
        torch.seed() #torch.manual_seed(torch.Generator().seed())
        index = torch.multinomial(value, 1).item()
        print(index)
        return [moves[index]]

class NeuralMoveCategorizerBot(NeuralBot):
    def __init__(self, model=None, gpu=False):
        super().__init__(model, gpu, 18) #18 is 9 piece categories and 9 move categories
    
    def makeMove(self, board, moves):
        # moves = None
        boardTensor = boardstateToTensor(board).unsqueeze()
        pieces_and_move_types = self.model(boardTensor)
        pieces = pieces_and_move_types.numpy()[0,8]
        moves = pieces_and_move_types.numpy()[8,17]
        selectedMoves = pickMoveFromCategory(pieces, moves, board)
        return selectedMoves

    def evalPos(self, board):
        return 0.5

def labelMove(board):
    move = board.pop()
    pieceFile = chess.square_file(move.from_square)
    pieceType = board.piece_type_at(move.from_square)
    if pieceType == 3:
        piece = 0
    elif pieceType == 2:
        piece = 1
    elif pieceType == 4:
        piece = 2
    if pieceType == 1 and pieceFile < 2:
        piece = 3
    if pieceType == 1 and 1 < pieceFile < 4:
        piece = 4
    if pieceType == 1 and 3 < pieceFile < 6:
        piece = 5
    if pieceType == 1 and 5 < pieceFile < 8:
        piece = 6
    elif pieceType == 6:
        piece = 7
    elif pieceType == 5:
        piece = 8
    pieces = [0,0,0,0,0,0,0,0,0]
    pieces[piece] = 1

    moves = [0,0,0,0,0,0,0,0,0]
    def checking(board, move):
        board.push(move)
        check = board.is_check()
        board.pop()
        return check
    if board.is_castling(move):
        moves[0] = 1
    elif board.is_capture(move) and chess.square_rank(move.from_square) < chess.square_rank(move.to_square):
        moves[1] = 1
    elif board.is_capture(move) and chess.square_rank(move.from_square) > chess.square_rank(move.to_square):
        moves[2] = 1
    elif board.is_capture(move) and chess.square_rank(move.from_square) == chess.square_rank(move.to_square):
        moves[3] = 1
    if checking(board, move):
        moves[4] = 1
    if chess.square_rank(move.from_square) < chess.square_rank(move.to_square): # forward move:
        moves[5] = 1
    if chess.square_rank(move.from_square) > chess.square_rank(move.to_square):
        moves[6] = 1
    if chess.square_file(move.from_square) > chess.square_file(move.to_square):
        moves[7] = 1
    if chess.square_file(move.from_square) < chess.square_file(move.to_square):
        moves[8] = 1

    board.push(move)
    return pieces+moves

def pickMoveFromCategory(pieceList, moveList, board):
    # input two numpy arrays [1x9] in dimensions
    # TODO optimize with recursion instead of complete matrix multiplication
    # pieceList
    # maxPiece = max(pieceList)
    # # indexGen = (i for i,val in enumerate(pieceList) if val == maxPiece) # gets all the max pieces
    # # indexPiece = next(ind for ind, i in enumerate(pieceList)) if i == maxPiece) # Gets first matching piece index
    # indexPiece = pieceList.index(maxPiece)

    # maxMove = max(moveList)
    # indexMove = moveList.index(maxMove)

    # movePossible = checkIfMove(indexMove, indexPiece, board)

    # if not movePossible:
    #     pickMoveFromCategory(pieceList[indexPiece]

    moveArray = np.multiply(pieceList, moveList)
    bestMove = lambda moveArray : numpy.unravel_index(indices=numpy.argmax(moveArray), shape=moveArray.shape())

    selectedMoves, movePossible = checkIfMove(bestMove(moveArray), board)
    while(not movePossible):
        moveArray[bestMove[0], bestMove[1]] = 0
        selectedMoves, movePossible = checkIfMove(bestMove(moveArray), board)
    return selectedMoves

def checkIfMove(move, board):
    # pieces:
    # bishop
    # knight
    # rook
    # A,B lane pawn
    # C, D lane pawn
    # E, F lane pawn
    # G, H lane pawn
    # King
    # Queen

    # move types:
    # castling
    # forward capture (y-axis)
    # backwards capture
    # horizontal capture
    # checking move
    # forward move
    # backward move
    # left move
    # right move

    # Step 1, sort moves that matches piece type
    def pieceMatch(square):
        pieceType = board.piece_type_at(square)
        PieceRank = chess.square_rank(square)
        pieceFile = chess.square_file(square)
        if move[0] == 0:
            match = pieceMatch == 3 # Bishop
        elif move[0] == 1:
            match = pieceMatch == 2 # Knight
        elif move[0] == 2:
            match = pieceMatch == 4 # Rook
        elif move[0] == 3:
            match = pieceMatch == 1 and PieceFile < 2
        elif move[0] == 4:
            match = pieceMatch == 1 and 1 < PieceFile < 4
        elif move[0] == 5:
            match = pieceMatch == 1 and 3 < PieceFile < 6
        elif move[0] == 6:
            match = pieceMatch == 1 and 5 < PieceFile < 8
        elif move[0] == 7:
            match = pieceMatch == 6
        elif move[0] == 8:
            match = pieceMatch == 5
        else:
            print("move matrix dimension missmatch")
        return match

    moves = [move for move in board.legal_moves() if pieceMatch(move.from_square)]
    # Step 2, sort moves that matches move type
    def checking(board, move):
        board.push(move)
        check = board.is_check()
        board.pop()
        return check

    def moveMatch(move):
        if move[1] == 0:
            match = board.is_castling(move)
        elif move[1] == 1:
            match = board.is_capture(move) and chess.square_rank(move.from_square) < chess.square_rank(move.to_square) # forward capture (y-axis)
        elif move[1] == 2:
            match = board.is_capture(move) and chess.square_rank(move.from_square) > chess.square_rank(move.to_square) # backwards capture
        elif move[1] == 3:
            match = board.is_capture(move) and chess.square_rank(move.from_square) == chess.square_rank(move.to_square) # horizontal capture
        elif move[1] == 4:
            match = checking(board, move)
        elif move[1] == 5:
            match = chess.square_rank(move.from_square) < chess.square_rank(move.to_square) # forward move
        elif move[1] == 6:
            match = chess.square_rank(move.from_square) > chess.square_rank(move.to_square)
        elif move[1] == 7:
            match = chess.square_file(move.from_square) > chess.square_file(move.to_square)
        elif move[1] == 8:
            match = chess.square_file(move.from_square) < chess.square_file(move.to_square)
        return match

    moves = [move for move in moves if moveMatch(move)]

    if len(moves) == 0:
        return (_, False)
    else:
        # board.push(random.choice(moves))
        return (moves.shuffle(), True)

if __name__ == "__main__":
    LOAD_FILE = "bad_neural_net.pt" # None #"bad_neural_net.pt"

    EPOCHS = 100
    GAMES = 10
    BATCH_SIZE = 1000
    PLAYER = NeuralBoardValueBot(model=LOAD_FILE, gpu=False)
    GPU = torch.cuda.is_available()
    OPPONENT = suicideBot()
    MAX_TRAIN_GAMES = 850

    optimizer = torch.optim.Adam(PLAYER.model.parameters())
    loss_fun = nn.MSELoss()
    games = []
    for epoch in range(EPOCHS):
        new_games = chessMaster.playMultipleGames(PLAYER, PLAYER, GAMES, workers=2, display_progress=True)
        new_games = [game.board for game in new_games]
        if not new_games[0].is_game_over():
            print("dfsdf")
            exit()
        games = new_games + games
        for game in games:
            print(game, game.result(), game.is_fivefold_repetition(), len(game.move_stack), "\n\n")
            print(labelMove(game))
        games = games[:min(MAX_TRAIN_GAMES, len(new_games))]
        matchesTensor, resultsTensor = matchesToTensor(games, whiteWinnerLabeller, 1)
        dataset = data.TensorDataset(matchesTensor, resultsTensor)
        dataloader = data.DataLoader(dataset, BATCH_SIZE, True)
        epoch_loss = 0
        if GPU:
            PLAYER.model.cuda()
        for i, (inputs, labels) in enumerate(dataloader):
            if GPU:
                inputs, labels = inputs.cuda(), labels.cuda()
            preds = PLAYER.model(inputs)
            loss = loss_fun(preds, labels)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print("\rEpoch {} [{}/{}] - Loss {}".format(epoch, i, len(dataloader), loss.item()),end="")
        PLAYER.model.cpu()
        torch.save(PLAYER.model, "bad_neural_net.pt")
        print("\rEpoch {} - Loss {}".format(epoch, epoch_loss))
