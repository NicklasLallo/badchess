import torch
import torch.nn as nn
import torch.utils.data as data
import chess
import chessUtils as cu
import numpy as np
import time
import random
from bots.simple import chessBot, randomBot, aggroBot
from bots.minimax import suicideBot, naiveMinimaxBot, minimax
from bots.engines import stockfish, stochfish
# from chessMaster import chessMaster
import chessMaster
import operator

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

def whiteWinnerLabeller(board, outcome, discount, stockfish=None):
    if not stockfish:
        if outcome == "0-1":
            outcome = 0.0
        elif outcome == "1-0":
            outcome = 1.0
        else:
            outcome = 0.5
    else: # if stockfish!
        outcome = float(stockfish.evalPos(board))
    return [outcome]

def matchesToTensor(boards, label_fun, out_size, only_winner=False, stockfish=None, discounted=False):
    '''
    Takes a list of Board and returns a Tensor of all boardstates and a Tensor with labels
    label_fun takes a board and the outcome of the game and returns a label for that board
    The label is a list of expected outputs for the neurons of the final layer
    out_size is the size of the label

    The outputs can also be labeled by the stockfish engine instead of the game output if one is provided.
    Technically any bot that has .evalPos() implemented should be useable to labeling

    discount is optional
    '''
    num_boardstates = sum([len(board.move_stack) for board in boards])
    boardstatesTensor = torch.zeros([num_boardstates, 65])
    resultsTensor = torch.zeros([num_boardstates, out_size])
    current_board = 0
    for i, board in enumerate(boards):
        discount = 1
        board = board.copy()
        outcome = board.result()
        while len(board.move_stack) > 0:
            if only_winner:
                # ignore draws
                if outcome == "1/2-1/2":
                    continue
            boardstatesTensor[current_board,:] = boardstateToTensor(board)
            print("\rLabeling. Game [{}/{}] with length: {}".format(i+1,len(boards),len(board.move_stack)),end="")
            resultsTensor[current_board, :] = torch.Tensor(label_fun(board, outcome, discount, stockfish))
            current_board += 1
            board.pop()
            if only_winner:
                # top move in stack should have been the winning move
                # so we don't need to check what side was the winning side
                if len(board.move_stack) > 0:
                    # don't pop if the stack had an odd size (white won)
                    board.pop() # skip over the moves of the losing side
                    if discounted:
                        discount *= 0.9 # make it still decay at the same speed as otherwise
            if discounted:
                discount *= 0.9
    print("")
    return boardstatesTensor, resultsTensor



class NeuralBot(chessBot):
    # Abstract class for NN bots that take a 65 dimensional boardstate as input
    def __init__(self, model=None, gpu=False, out_size=1):
        self.model = model
        self.gpu = gpu
        if self.model is None:
            # Default model
            self.model = nn.Sequential(
                nn.Linear(65, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 64),
                nn.LeakyReLU(),
                nn.Linear(64, out_size),
                nn.Sigmoid()
            )
        elif isinstance(self.model,str):
            self.model = torch.load(model)
        if self.gpu:
            self.model.cuda()

    def evalPos(self, board):
        pass
    def makeMove(self, board, moves, verbose):
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

    def makeMove(self, board, moves, verbose=False):
        # print("beg")
        moves = list(moves)
        boards = []
        for move in moves:
            b = board.copy(stack=False)
            b.push(move)
            boards.append(b)
        boardsTensor, _ = matchesToTensor(boards, whiteWinnerLabeller, 1)
        if self.gpu:
            boardsTensor = boardsTensor.cuda()
            # self.model.cuda()
        value = torch.softmax(self.model(boardsTensor).view(-1), 0)
        if not board.turn: # add Not to attempt to win instead of lose
            value = 1-value
        # torch.seed() #torch.manual_seed(torch.Generator().seed())
        index = torch.multinomial(value+1e-8, 1).item()
        return [moves[index]]

class NeuralMoveInstructionBot(NeuralBot):
    def __init__(self, model=None, gpu=False):
        super().__init__(model, gpu, 18) #18 is 9 piece categories and 9 move categories

    def makeMove(self, board, moves, verbose):
        # moves = None
        boardTensor = boardstateToTensor(board).unsqueeze(0)
        if self.gpu:
            boardTensor = boardTensor.cuda()
        pieces_and_move_types = self.model(boardTensor).detach().cpu().numpy()
        #if not board.turn:
            # pieces_and_move_types = [1-elem for elem in pieces_and_move_types]
            # pieces_and_move_types = np.subtract(pieces_and_move_types, 1)
        pieces = pieces_and_move_types[:,0:9]
        moves = pieces_and_move_types[:,9:18]
        selectedMoves = pickMoveFromCategory(pieces, moves, board, verbose)
        return selectedMoves

    def evalPos(self, board):
        return 0.5

class NeuralMoveInstructionBotv2(NeuralBot):
    def __init__(self, model=None, gpu=False):
        super().__init__(model, gpu, 81) # 81, each combination of 9 piece categories and 9 move categories

    def makeMove(self, board, moves, verbose):
        boardTensor = boardstateToTensor(board).unsqueeze(0)
        if self.gpu:
            boardTensor = boardTensor.cuda()
        numpy_moves = self.model(boardTensor).detach().cpu().numpy()
        candidate_moves = numpy_moves.argsort() # has the indexes of the sorted moves lowest -> highest
        candidate_moves = np.flip(candidate_moves) # highest -> lowest
        candidate_moves = np.squeeze(candidate_moves) # remove extra useless dimension
        for candidate_move in candidate_moves:
            # print(candidate_move)
            move = (int(candidate_move / 9), int(candidate_move % 9))
            # fist value should be piece type, second move type
            selectedMoves, movePossible = checkIfMove(move, board)
            if movePossible:
                break
        if verbose:
            print("Selected moves: " + selectedMoves)
        return selectedMoves

    def evalPos(self, board):
        return 0.5

def move_to_cat(board, move):
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

    return pieces+moves

def instructionLabel18(board, outcome, discount, stockfish=None): # for 18 long output
    if not stockfish:
        assert outcome != "1/2-1/2", "NeuralMoveInstructionBot should not be trained on draws without stockfish!"
    else: # if stockfish!
        pre_move_score = stockfish.evalPos(board)
    move = board.pop()
    outputArray = move_to_cat(board, move)
    if not stockfish:
        # If active player is white and lost
        if outcome == "0-1" and board.turn:
            outputArray = [1 - elem for elem in outputArray]
        # Or if active player is black and lost
        elif outcome == "1-0" and not board.turn:
            outputArray = [1 - elem for elem in outputArray]
        # Otherwise the player who is active won

        # Discount early moves to matter less than later moves
        outputArray = [(elem - 0.5)*discount+0.5 for elem in outputArray]
    else: # if stockfish!
        post_move_score = stockfish.evalPos(board) # fixed point of view
        if not board.turn: # not because this is after we pop the move stack
            score = post_move_score - pre_move_score # maximize score for white
        else:
            score = pre_move_score - post_move_score # minimize score for white
        # scale the output rewards based on the score difference the move makes
        # as well as based on the discount
        outputArray = [((0.5+score)*elem-0.5)*discount+0.5 for elem in outputArray]
    # Push the move to reset the board again
    board.push(move)
    return outputArray

def convert_81(input_array):
    # convert from 18 to 81
    allcombos = []
    for pieceT in input_array[0:9]:
        for moveT in input_array[9:18]:
            allcombos.append(moveT*pieceT)
            # [piece1 * moves,
            #  piece2 * moves,
            # ...
            # piece9 * moves]
    return allcombos

def instructionLabel81(board, outcome, discount, stockfish=None):
    # given a move, find each category that contains that move.
    # then scale each category based on the inverse category size
    # and then scale the total based on how stockfish thinks this move changed
    # the over all position compared to before
    if not stockfish:
        assert outcome != "1/2-1/2", "NeuralMoveInstructionBot should not be trained on draws without stockfish!"
    else: # if stockfish!
        pre_move_score = stockfish.evalPos(board)
    move = board.pop()
    outputArray = move_to_cat(board, move)
    outputArray = convert_81(outputArray) # from 18 (9*2) to 81 (9*9)
    # scale with number of moves in each category
    # Fast?
    if not SPEEDHACK:
        density = np.array([0] * 81)
        for move in board.legal_moves:
            # For each possible move, find which categories it fits in and increase them by one
            # denisty is a nparray so it should do elementwise addition and not pythonList append
            density += convert_81(move_to_cat(board, move))
        # normalize density
        density = cu.normalized(density) # returns a np array
        # pythonlist - nparray is elementwize subtraction to nparray output
        outputArray -= density # subtract density to favour small categories
    if stockfish:
        post_move_score = stockfish.evalPos(board) # fixed point of view
        if not board.turn: # not because this is after we pop the move stack
            score = post_move_score - pre_move_score # maximize score for white
        else:
            score = pre_move_score - post_move_score # minimize score for white
        outputArray = [((0.5+score)*elem-0.5)*discount+0.5 for elem in outputArray]
    else:
        # If active player is white and lost
        if outcome == "0-1" and board.turn:
            outputArray = [1 - elem for elem in outputArray]
        # Or if active player is black and lost
        elif outcome == "1-0" and not board.turn:
            outputArray = [1 - elem for elem in outputArray]
        # Otherwise the player who is active won

        # Discount early moves to matter less than later moves
        outputArray = [(elem - 0.5)*discount+0.5 for elem in outputArray]
    # Push the move to reset the board again
    board.push(move)
    return outputArray

def pickMoveFromCategory(pieceList, moveList, board, verbose=False):
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

    # print(pieceList)
    # print(moveList)
    moveArray = np.multiply(pieceList.T, moveList)
    bestMove = lambda moveArray : np.unravel_index(indices=np.argmax(moveArray), shape=moveArray.shape)

    # backupMoves = board.copy().legal_moves
    # backupBoard = str(board.copy())
    # backupPieceList = pieceList.copy()
    # backupMoveList = moveList.copy()
    # backupMatrix = moveArray.copy()

    movePossible = False
    while(not movePossible):
        proposedSolution = bestMove(moveArray)
        if moveArray[proposedSolution[0]][proposedSolution[1]] == 0:
            raise ValueError("everything became zero")
        selectedMoves, movePossible = checkIfMove(proposedSolution, board)
        if verbose:
            translateMove(proposedSolution[0], proposedSolution[1])
            if not movePossible:
                print("But there doesn't appear to be any such moves")
        moveArray[proposedSolution[0]][proposedSolution[1]] = 0
    if verbose:
        print("I settled on making one of these moves: ", str(selectedMoves))
    return selectedMoves


def translateMove(pieceInt, moveInt):
    pieces = [ "bishop ", "knight ", "rook ", "A or B lane pawn ", "C or D lane pawn ", "E or F lane pawn ", "G or H lane pawn ", "King ", "Queen " ]
    moves = [ "castling", "forward capture", "backward capture", "horizontal capture", "checking move", "forward move", "backward move", "left move", "right move"]
    print("I would like to do an ", pieces[pieceInt], moves[moveInt])


def checkIfMove(move, board, debug=False):

    if debug:
        print("I want to make move: ", move)

    # Step 1, sort moves that matches piece type
    def pieceMatch(square):
        pieceType = board.piece_type_at(square)
        pieceRank = chess.square_rank(square)
        pieceFile = chess.square_file(square)
        if move[0] == 0:
            match = pieceType == 3 # Bishop
        elif move[0] == 1:
            match = pieceType == 2 # Knight
        elif move[0] == 2:
            match = pieceType == 4 # Rook
        elif move[0] == 3:
            match = pieceType == 1 and pieceFile < 2
        elif move[0] == 4:
            match = pieceType == 1 and 1 < pieceFile < 4
        elif move[0] == 5:
            match = pieceType == 1 and 3 < pieceFile < 6
        elif move[0] == 6:
            match = pieceType == 1 and 5 < pieceFile < 8
        elif move[0] == 7:
            match = pieceType == 6
        elif move[0] == 8:
            match = pieceType == 5
        else:
            print("move matrix dimension missmatch")
        return match

    if debug:
        for tmpMove in board.legal_moves:
            print("filterPiece: ",  tmpMove, pieceMatch(tmpMove.from_square))
    moves = [tempMove for tempMove in board.legal_moves if pieceMatch(tempMove.from_square)]
    # Step 2, sort moves that matches move type
    # def checking(board, move):
    #     board.push(move)
    #     check = board.is_check()
    #     board.pop()
    #     return check

    def moveMatch(candidateMove):
        if move[1] == 0:
            match = board.is_castling(candidateMove)
        elif move[1] == 1:
            match = board.is_capture(candidateMove) and chess.square_rank(candidateMove.from_square) < chess.square_rank(candidateMove.to_square) # forward capture (y-axis)
        elif move[1] == 2:
            match = board.is_capture(candidateMove) and chess.square_rank(candidateMove.from_square) > chess.square_rank(candidateMove.to_square) # backwards capture
        elif move[1] == 3:
            match = board.is_capture(candidateMove) and chess.square_rank(candidateMove.from_square) == chess.square_rank(candidateMove.to_square) # horizontal capture
        elif move[1] == 4:
            match = board.is_into_check(candidateMove)
        elif move[1] == 5:
            match = chess.square_rank(candidateMove.from_square) < chess.square_rank(candidateMove.to_square) # forward candidateMove
        elif move[1] == 6:
            match = chess.square_rank(candidateMove.from_square) > chess.square_rank(candidateMove.to_square)
        elif move[1] == 7:
            match = chess.square_file(candidateMove.from_square) > chess.square_file(candidateMove.to_square)
        elif move[1] == 8:
            match = chess.square_file(candidateMove.from_square) < chess.square_file(candidateMove.to_square)
        return match


    if debug:
        for tmpMove in board.legal_moves:
            print("filterMove: ",  tmpMove, moveMatch(tmpMove))
    moves = [tmpMove for tmpMove in moves if moveMatch(tmpMove)]
    # print(len(moves))

    if len(moves) == 0:
        return (None, False)
    else:
        # board.push(random.choice(moves))
        # print("wee")
        if debug:
            print(moves)
        random.shuffle(moves)
        return (moves, True)

if __name__ == "__main__":
    EPOCHS = 100
    GAMES = 60
    GAMES2 = int(GAMES / 2)
    GAMES3 = GAMES * 3
    BATCH_SIZE = 1000
    GPU = True
    # PLAYER = NeuralBoardValueBot(model=LOAD_FILE, gpu=False)
    model = nn.Sequential(
        nn.Linear(65, 256),
        nn.LeakyReLU(),
        nn.Linear(256, 256),
        nn.LeakyReLU(),
        nn.Linear(256, 384),
        nn.LeakyReLU(),
        nn.Linear(384, 256),
        nn.LeakyReLU(),
        nn.Linear(256, 256),
        nn.LeakyReLU(),
        nn.Linear(256, 18),
        nn.Sigmoid()
    )
    # LOAD_FILE = "instruction_neural_net_extralarge.pt"; PLAYER = NeuralMoveInstructionBot(model=LOAD_FILE, gpu=GPU); LABELLER = instructionLabel18; OUT_SIZE=18; ONLY_WINNER=True
    LOAD_FILE = "instruction_neural_net_v2_extralarge.pt"; PLAYER = NeuralMoveInstructionBotv2(model=LOAD_FILE, gpu=GPU); LABELLER = instructionLabel81; OUT_SIZE=81; ONLY_WINNER=False
    # LOAD_FILE = "not_as_bad_neural_net_large.pt"; PLAYER = NeuralBoardValueBot(model=LOAD_FILE, gpu=GPU); LABELLER = whiteWinnerLabeller; OUT_SIZE=1; ONLY_WINNER=False
    # GPU = torch.cuda.is_available()
    # OPPONENT = aggroBot()
    SPEEDHACK = True
    MAX_TRAIN_GAMES = 850
    optimizer = torch.optim.Adam(PLAYER.model.parameters())
    loss_fun = nn.MSELoss()
    games = []
    # opponentList = [aggroBot(), randomBot(), naiveMinimaxBot(), PLAYER]
    STOCKFISH = stockfish(time=0.0008)
    STOCHFISH = stochfish(time=0.0012)
    IGNORE_DRAWS = False
    # opponentList = [opponent, PLAYER]
    opponentList = [STOCKFISH]
    for epoch in range(EPOCHS):
        OPPONENT = random.choice(opponentList)
        print("Playing against", str(type(OPPONENT).__name__))
        print("Playing games as white")
        new_games = chessMaster.playSingleGames(PLAYER, OPPONENT, GAMES2, workers=2, display_progress=True, log=False)
        gameresults = (0,0,0)
        for g in new_games:
            gameresults = tuple(map(operator.add, gameresults, g.winner()))
        print("Playing games as black")
        new_games2 = chessMaster.playSingleGames(OPPONENT, PLAYER, GAMES2, workers=2, display_progress=True, log=False)
        for g in new_games2:
            invertedWinner = g.winner()
            invertedWinner = (invertedWinner[1],invertedWinner[0],invertedWinner[2])
            gameresults = tuple(map(operator.add, gameresults, invertedWinner))
        new_games += new_games2
        print("results: ", gameresults)
        with open('trainingLog.tsv', 'a') as f:
            f.write("%s\t%s\t%s\t%s\t%s" % (str(type(PLAYER).__name__),str(type(OPPONENT).__name__),str(gameresults[0]),str(gameresults[1]),str(gameresults[2])))
        if GAMES3 and STOCKFISH and STOCHFISH:
            print("Stock vs Stoch: " + str(GAMES3))
            new_games3 = chessMaster.playSingleGames(STOCKFISH, STOCHFISH, GAMES3, workers=2, display_progress=True, log=False)
            new_games += new_games3
        if IGNORE_DRAWS:
            new_games = [game.board for game in new_games if game.winner() != (0,0,1)]
        else:
            new_games = [game.board for game in new_games]
        if len(new_games)>0 and not new_games[0].is_game_over():
            exit()
        games = new_games + games
        # for game in games:
        #     # print(game, game.result(), game.is_fivefold_repetition(), len(game.move_stack), "\n\n")
        #     # print(labelMove(game))
        # This doesn't really do anything
        # If we want to save results between epochs we should save labeled tensors
        games = games[:min(MAX_TRAIN_GAMES, len(new_games))]
        # games = games[:MAX_TRAIN_GAMES]
        print("Training on " + str(len(games)) + " games.")
        if len(games) == 0:
            print("no games to train on")
            print("Epoch {} - Loss NaN".format(epoch))
            with open('trainingLog.tsv', 'a') as f:
                f.write("\t\n")
            continue
        matchesTensor, resultsTensor = matchesToTensor(games, LABELLER, OUT_SIZE, only_winner=ONLY_WINNER, stockfish=STOCKFISH)
        # matchesTensor, resultsTensor = matchesToTensor(games, whiteWinnerLabeller, 1)
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
        print("")
        # PLAYER.model.cpu()
        torch.save(PLAYER.model, LOAD_FILE)
        print("Epoch {} - Loss {}".format(epoch, epoch_loss/(i+1)))
        with open('trainingLog.tsv', 'a') as f:
            f.write("\t%s\n" % (str(epoch_loss/(i+1))))
    print("Drowning the stocky fishes")
    if STOCHFISH:
        STOCHFISH.quit()
    STOCKFISH.quit()
