"""
ConnectFour.pyw - a game of connect four

The ConnectFourBoard Class was derived from the work at https://github.com/Acamol/connect-four
"""

import tkinter as tk
import tkinter.messagebox as messagebox
import numpy as np
import platform
import random
from time import sleep
from Agent import basic_agent, minimax3_agent
from types import SimpleNamespace

# Double clicking the file or launching from another directory won't work unless we first "cd" to the app directory
if platform.system() == "Windows":
    from sys import argv
    from os import getcwd, chdir, path
    if len(argv) >= 1:
        file = argv[0]
        cwd = getcwd()
        dir = path.dirname(file)
        if not path.isabs(dir):
            dir = cwd + "\\" + dir
        chdir(dir)

# Game configuration; 7 columns, 6 rows, 4 in a row
# Changing these will let you play alternative games (e.g., tick-tack-toe = 3/3/3)
class Config():
    def __init__(self):
        self.rows = 6
        self.columns = 7
        self.inarow = 4

config = Config()
ROWS = config.rows          # Short-hand
COLS = config.columns       # Short-hand
IN_A_ROW = config.inarow    # Short-hand

# GUI color these constants
BACKGROUND = "blue"
OUTLINE_COLOR = "black"
PLAYER_COLORS = ["yellow", "red"]
BLANK_COLOR = "lightgrey"

# Other constants
OPPONENTS = ["person", "computer",  "computer pro"]
AGENTS =    [None,     basic_agent, minimax3_agent]
HELP_TEXT = """Try to make four in a row before your opponent.
Changing color or opponent will reset any game in progress.
The current game state (whose turn it is, if the game is over) is at the bottom of the screen.
When it's your turn, click on the column you want to drop you peice into.
"""

# A Player. agent callback function is None for human players
class Player():
    def __init__ (self, color : str, agent : callable):
        self.color = color
        self.agent = agent

    @property
    def is_computer(self):
        return self.agent is not None

# Game engine
class Game():
    def __init__(self, p1_agent, p2_agent):
        self.players = [
            Player(BLANK_COLOR, None),              # No player
            Player(PLAYER_COLORS[0], p1_agent),     # Player1 - yellow
            Player(PLAYER_COLORS[1], p2_agent),     # Player2 - red
        ]
        self.current_player = 1     # Index of current player; 1 or 2, or 0 if the game is over
        self.winning_player = 0     # If self.current_player == 0, gives result (0 = tie, 1 or 2 if that player won)
        self.grid = np.zeros([ROWS, COLS], dtype=int)   # Grid of player indexes

    @property
    def current_player_color(self) -> str:
        return self.players[self.current_player].color

    @property
    def current_player_agent(self) -> callable:
        return self.players[self.current_player].agent

    @property
    def is_game_over(self) -> bool:
        return self.current_player == 0

    @property
    def is_tie(self) -> bool:
        return self.is_game_over and self.winning_player == 0

    @property
    def winning_player_color (self) -> str:
        if not self.is_game_over or self.winning_player == 0:
            return None
        return self.players[self.winning_player].color

    # Count number of items in a row in a given direction that are the same as the peice in a given row and column
    def num_in_a_row (self, row : int, col : int, row_inc : int, col_inc : int):
        player = self.grid[row, col]
        num = 0
        while 0 <= row + row_inc < ROWS and 0 <= col + col_inc < COLS and self.grid[row + row_inc, col + col_inc] == player:
            row += row_inc
            col += col_inc
        num = 1
        while 0 <= row - row_inc < ROWS and 0 <= col - col_inc < COLS and self.grid[row - row_inc, col - col_inc] == player:
            row -= row_inc
            col -= col_inc
            num += 1
        return num

    # Check if the move at given row, col was a winning move
    def check_winning_move(self, row : int, col : int) -> bool:
        # horizontal
        if self.num_in_a_row(row, col, 1, 0) >= IN_A_ROW:
            return True
        # vertical
        if self.num_in_a_row(row, col, 0, 1) >= IN_A_ROW:
            return True
        # diagnal
        if self.num_in_a_row(row, col, 1, 1) >= IN_A_ROW:
            return True
        if self.num_in_a_row(row, col, 1, -1) >= IN_A_ROW:
            return True
        return False 

    # Called to let the current player make a move at colum col
    # Returns the row number the peice landed in, or -1 if it was an invalid move
    def move (self, col : int) -> bool:
        # First check if the game is alread over, or if it's an invalid move
        if self.current_player == 0 or self.grid[0, col] != 0:
            return -1
        
        # Drop the current players peice into the selected column
        for row in range(ROWS-1, -1, -1):
            if self.grid[row][col] == 0:
                self.grid[row][col] = self.current_player
                break

        # Check if it was a winning move
        if self.check_winning_move (row, col):
            self.winning_player = self.current_player   # We have a winner
            self.current_player = 0                     # And the game is over
            return row

        # Check if it is a tie
        if np.count_nonzero(self.grid) == ROWS * COLS:
            self.winning_player = 0                     # No winner
            self.current_player = 0                     # But the game is over
            return row

        # It's now the next players turn
        self.current_player = 3 - self.current_player   # 2->1, 1->2
        return row

# ConnectFourBoard - main control of the GUI
# Provides a 6x7 canvas round peices in a given background color
# Provides an API to set the peice at a given (row, col) to a given color
# on_click callback allows user to find out when a column is clicked
class ConnectFourBoard(tk.Canvas):
    def __init__ (self, on_click : callable, rows=ROWS, cols=COLS, tile_size = 60, fill_color=BLANK_COLOR, bg_color=BACKGROUND):
        super().__init__(bg = bg_color)
        self.fill_color = fill_color
        self.board = np.empty([rows, cols], dtype=tk.Canvas)
        self.peice_ids = np.zeros([rows, cols], dtype=int)
        self.on_click = on_click
        peice_size = tile_size * 5 // 6     # Each tile has a circle representing whatever peices is in that tile
        for row in range (rows):
            for col in range (cols):
                canvas_tile = tk.Canvas(self, bg=bg_color, height=tile_size, width=tile_size, relief="raised", highlightthickness=0)
                self.board[row, col] = canvas_tile
                padding = 2
                id = canvas_tile.create_oval((padding, padding, peice_size + padding, peice_size + padding), outline=OUTLINE_COLOR, fill=fill_color)
                self.board[row, col] = canvas_tile
                self.peice_ids[row, col] = id
                canvas_tile.grid(row=row, column=col, padx=5, pady=5)
                canvas_tile.bind('<Button-1>', lambda e, col=col : on_click (col))

    # Set the color of the peice at the given row and column
    def set_color (self, row : int, col : int, color : str):
        self.board[row, col].itemconfig(self.peice_ids[row, col], fill=color)

    # Reset the board for a new game
    def reset (self):
        for row in range (self.board.shape[0]):
            for col in range (self.board.shape[1]):
                id = self.peice_ids[row, col]
                self.board[row, col].itemconfig(id, fill=self.fill_color)

# Game GUI
class GameScreen(tk.Tk):
    def __init__(self):
        super().__init__()
        self.iconbitmap(default='ConnectFour.ico')
        self.title("Connect Four")

        # Create buttons at the top
        tk.Label(self, text="Your color:").grid (row=0, column=0, padx=(10, 0))
        self.color_select_btn = tk.Button (self, text=PLAYER_COLORS[0], width = 7, command=lambda : self.on_color_select_click())
        self.color_select_btn.grid(row=0, column=1, padx=(0, 10))

        tk.Label(self, text="Your opponent:").grid (row=0, column=2, padx=(10, 0))
        self.opponent_select_btn = tk.Button (self, text=OPPONENTS[1], width = 13, command=lambda : self.on_opponent_select_click())
        self.opponent_select_btn.grid(row=0, column=3, padx=(0, 10))

        help_btn = tk.Button (self, text="Help", command=lambda : self.on_help_click())
        help_btn.grid(row=0, column=4)

        # Main canvas for the game board in the middle
        self.board = ConnectFourBoard (self.on_board_clicked)
        self.board.grid(row=1, column=0, columnspan=5)
        
        # Game state info at the bottom
        self.game_state_label = tk.Label(self, text="")
        self.game_state_label.grid(row=2, column=0, columnspan=5, padx=2, pady=5)

        # Window shouldn't be resizable, since widgets are not
        self.resizable(False, False)

        # And finally, start a game
        self.new_game()
    
    def new_game (self):
        # Reset the Canvas to no pieces are played
        self.board.reset()

        # Create a new game against the selected opponent, giving user their selected color, and randomize who goes first
        agent_ix = OPPONENTS.index(self.opponent_select_btn["text"])
        agent = AGENTS[agent_ix]
        if self.color_select_btn["text"] == PLAYER_COLORS[1]:
            self.game = Game(agent, None)
        else:
            self.game = Game(None, agent)

        # Set who goes first - if it's the computer, have them move
        self.game.current_player = random.randint(1, 2)
        self.game_state_label["text"] = "New game: " + self.game.current_player_color + " goes first"
        if self.game.current_player_agent is not None:
            col = self.make_computer_move()

    def on_help_click(self):
        messagebox.showinfo("Help",  HELP_TEXT, icon="question")
    
    # Clicking on color_select rotates the button name through the PLAYER_COLORS options
    def on_color_select_click(self):
        color_ix = PLAYER_COLORS.index(self.color_select_btn["text"])
        self.color_select_btn["text"] = PLAYER_COLORS[(color_ix + 1) % len(PLAYER_COLORS)]
        self.new_game()

    # Clicking on oponent_select rotates the button name through the OPPONENTS options
    def on_opponent_select_click(self):
        opponent_ix = OPPONENTS.index(self.opponent_select_btn["text"])
        self.opponent_select_btn["text"] = OPPONENTS[(opponent_ix + 1) % len(OPPONENTS)]
        self.new_game()

    # This function has the current player place a tile in col and updates everything
    # Returns true of false depending on if the move was succesful
    def place_tile (self, col) -> bool:
        # Make the move and reflect it on the board
        player_color = self.game.current_player_color
        is_computer = self.game.current_player_agent is not None
        row = self.game.move(col)
        if row == -1:   # If they made an invalid move, ignore the click
            return False
        self.board.set_color (row, col, player_color)

        # If the computer made the move, flash the tile to bring attention to the placement
        if is_computer:
            self.after (100, lambda : self.board.set_color(row, col, BLANK_COLOR))
            self.after (100, lambda : self.board.set_color(row, col, player_color))

        # Update the game state text
        if self.game.is_game_over:
            if self.game.is_tie:
                self.game_state_label["text"] = "Game over - it's a tie!"
            else:
                self.game_state_label["text"] = "Game over - " + self.game.winning_player_color + " wins!"
        else:
            self.game_state_label["text"] = "It's " + self.game.current_player_color + "'s turn"
        return True


    # Column col was clicked. Make the move
    #def on_canvas_click(self, event, id, col):
    def on_board_clicked(self, col):
        # If it is the agents turn, ignore the click
        if self.game.current_player_agent is not None:
            return

        # If it's a persons turn, make the move and reflect it on the board
        if not self.place_tile (col):
            return      # If invalid move, return

        # If the persons opponent is a computer agent, have them play
        if not self.game.is_game_over and self.game.current_player_agent is not None:
            col = self.make_computer_move()
            

    # Get the computers move
    def make_computer_move (self) -> int:
        agent = self.game.current_player_agent
        assert agent is not None, "Trying to get computers move, but it's not the computers turn"

        # Get the agents move, after converting parameters to format in Agent.py
        obs = SimpleNamespace()
        obs.mark = self.game.current_player
        obs.board = self.game.grid.flatten().tolist()
        col = agent (obs, config)

        # Make the move after a slight delay so the computer appears to be thinking
        player_color = self.game.current_player_color
        self.after (500, lambda : self.place_tile(col))

root = GameScreen()
root.mainloop()
