import json
from bots import simple, minimax#, engines
from chessMaster import playSingleGames

class Managed_bot(object):
    def __init__ (self):
        pass

    def __init__(self, bot_ref):
        self.bot = bot_ref()
        self.scoreTable = {} # TODO placeholder

class Manager(object):
    def __init__(self):

        self.bots = {
            "randomBot":{},
            "aggroBot":{},
            "lowRankBot":{},
            "naiveMinimaxBot":{},
            "jaqueBot":{}
                     }
        self.save_file = "scorefile.json"
        self.loadedBots = []

        try:
            with open(self.save_file) as f:
                data = json.load(f)
                for bot in self.bots:
                    if bot in data:
                        results = data[bot]
                        self.bots[bot] = results
                    else:
                        results = "n/a"
        except:
            self.debugSave()
            with open(self.save_file) as f:
                data = json.load(f)
                for bot in self.bots:
                    if bot in data:
                        results = data[bot]
                        self.bots[bot] = results
                    else:
                        results = "n/a"

    def debugSave(self):
        for bot in self.bots:
            for oppo in self.bots:
                self.bots[bot][oppo] = {"matches": 0, "wins": 0, "losses": 0, "draws": 0, "winrate":0.5}
        with open(self.save_file, "w") as f:
            json.dump(self.bots, f)

    def save(self):
        with open(self.save_file, "w") as f:
            json.dump(self.bots, f)

    def statusGrid(self):
        print(self.bots)

    def load(self, name):
        if name == "debugSave":
            self.debugSave()
            return "debugSaved"
        if name == "randomBot":
            bot = simple.randomBot()
        elif name == "aggroBot":
            bot = simple.aggroBot()
        elif name == "lowRankBot":
            bot = simple.lowRankBot()
        elif name == "naiveMinimaxBot":
            bot = minimax.naiveMinimaxBot()
        elif name == "jaqueBot":
            bot = simple.jaqueBot()
        else:
            Exception(f"No bot called '{name}' found.")
        self.loadedBots.append(bot)
        print(f"bot {name} loaded.")

    def matchInfo(self):
        if len(self.loadedBots) == 2:
            bot0 = str(type(self.loadedBots[0]).__name__)
            bot1 = str(type(self.loadedBots[1]).__name__)
            print(f"match data from perspective of {bot0} vs {bot1}: \n {self.bots[bot0][bot1]}")

    def playMatch(self,number=20):
        if len(self.loadedBots) == 2:
            bot0obj = self.loadedBots[0]
            bot1obj = self.loadedBots[1]

            games = playSingleGames(bot0obj, bot1obj, int(number), workers=2, chessVariant='Standard', display_progress=True, log=False, save=True)

            bot0 = str(type(self.loadedBots[0]).__name__)
            bot1 = str(type(self.loadedBots[1]).__name__)
            for game in games:
                result = game.winner()
                if result == (0,0,1):
                    self.bots[bot0][bot1]["matches"] += 1
                    self.bots[bot0][bot1]["draws"] += 1
                    draws = self.bots[bot0][bot1]["draws"]
                    wins = self.bots[bot0][bot1]["wins"]
                    matches = self.bots[bot0][bot1]["matches"]
                    if wins != 0:
                        self.bots[bot0][bot1]["winrate"] = (float(wins)+0.5*float(draws)) / float(matches)
                    else:
                        self.bots[bot0][bot1]["winrate"] = 0
                    if bot0 != bot1:
                        self.bots[bot1][bot0]["matches"] += 1
                        self.bots[bot1][bot0]["draws"] += 1
                    if wins != 0:
                        self.bots[bot1][bot0]["winrate"] = (float(wins)+0.5*float(draws)) / float(matches)
                    else:
                        self.bots[bot1][bot0]["winrate"] = 0
                elif result == (1,0,0):
                    self.bots[bot0][bot1]["wins"] += 1
                    self.bots[bot0][bot1]["matches"] += 1
                    draws = self.bots[bot0][bot1]["draws"]
                    wins    = self.bots[bot0][bot1]["wins"]
                    matches = self.bots[bot0][bot1]["matches"]
                    if wins != 0:
                        self.bots[bot0][bot1]["winrate"] = (float(wins)+0.5*float(draws)) / float(matches)
                    else:
                        self.bots[bot0][bot1]["winrate"] = 0

                    self.bots[bot1][bot0]["losses"] += 1
                    if bot0 != bot1:
                        self.bots[bot1][bot0]["matches"] += 1
                    if wins != 0:
                        self.bots[bot1][bot0]["winrate"] = 1.0 - ((float(wins)+0.5*float(draws)) / float(matches))
                    else:
                        self.bots[bot1][bot0]["winrate"] = 0
                elif result == (0,1,0):
                    self.bots[bot1][bot0]["wins"] += 1
                    self.bots[bot1][bot0]["matches"] += 1
                    draws = self.bots[bot0][bot1]["draws"]
                    wins    = self.bots[bot1][bot0]["wins"]
                    matches = self.bots[bot1][bot0]["matches"]
                    if wins != 0:
                        self.bots[bot1][bot0]["winrate"] = (float(wins)+0.5*float(draws)) / float(matches)
                    else:
                        self.bots[bot1][bot0]["winrate"] = 0

                    self.bots[bot0][bot1]["losses"] += 1
                    if bot0 != bot1:
                        self.bots[bot0][bot1]["matches"] += 1
                    if wins != 0:
                        self.bots[bot0][bot1]["winrate"] = 1.0 - ((float(wins)+0.5*float(draws)) / float(matches))
                    else:
                        self.bots[bot0][bot1]["winrate"] = 0
                else:
                    print(f"unknown result: {result}")
            self.save()

if __name__ == "__main__":
    man = Manager()
    print("initialized chess bot manager.")
    # man.debugSave()
    # quit()
    # man.statusGrid()
    print("Select two bots to play:")
    bot_1 = input("Bot 1 >> ")
    man.load(bot_1)
    bot_2 = input("Bot 2 >> ")
    man.load(bot_2)
    num = input("How many matches to play? >> ")
    try:
        man.playMatch(int(num))
    except:
        print(f"{num} is not a number, using 20 instead")
        man.playMatch()
    man.matchInfo()
