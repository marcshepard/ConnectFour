"""
connect_four.pyw - a game of connect four

The ConnectFourBoard Class was derived from the work at https://github.com/Acamol/connect-four
"""

import tkinter as tk
from tkinter import messagebox
import platform
import random
from types import SimpleNamespace
import numpy as np
from agent import basic_agent, minimax3_agent

# Enable launch by double clicking
if platform.system() == "Windows":
    from sys import argv
    from os import getcwd, chdir, path
    if len(argv) >= 1:
        DIR = path.dirname(argv[0])
        if not path.isabs(DIR):
            DIR = getcwd() + "\\" + DIR
        chdir(DIR)

# Game configuration (change to 3/3/3 for tic-tac-toe)
ROWS = 6        # 6 rows
COLS = 7        # 7 columns
IN_A_ROW = 4    # 4 in a row to win

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

def make_player(color : str, agent : callable) -> SimpleNamespace:
    """A player for the Game() class; agent is None for humans, or a callback"""
    player = SimpleNamespace()
    player.color = color
    player.agent = agent
    return player

class Game():
    """Game engine"""
    def __init__(self, p1_agent, p2_agent):
        self.players = [
            None,                                       # No player
            make_player(PLAYER_COLORS[0], p1_agent),    # Player1 - yellow
            make_player(PLAYER_COLORS[1], p2_agent),    # Player2 - red
        ]
        self.current_player = 1     # Index of current player; 1 or 2, or 0 if the game is over
        self.winning_player = 0     # If current_player == 0, 0 = tie, 1 or 2 if that player won
        self.grid = np.zeros([ROWS, COLS], dtype=int)   # Grid of player indexes

    @property
    def current_player_color(self) -> str:
        """Current players color"""
        return self.players[self.current_player].color

    @property
    def current_player_agent(self) -> callable:
        """Current players agent"""
        return self.players[self.current_player].agent

    @property
    def is_computers_turn(self) -> bool:
        """Is it the computers turn?"""
        return self.players[self.current_player].agent is not None

    @property
    def is_game_over(self) -> bool:
        """Is it the game over?"""
        return self.current_player == 0

    @property
    def is_tie(self) -> bool:
        """Did the game end in a tie?"""
        return self.is_game_over and self.winning_player == 0

    @property
    def winning_player_color (self) -> str:
        """The color of the player who won, or None if game not over"""
        if not self.is_game_over or self.winning_player == 0:
            return None
        return self.players[self.winning_player].color

    def num_in_a_row (self, row : int, col : int, row_inc : int, col_inc : int):
        """Support method for check_winning_move
        Returns the number of items in a row in a given direction from row and column"""
        player = self.grid[row, col]
        num = 0
        while 0 <= row + row_inc < ROWS and 0 <= col + col_inc < COLS and \
                self.grid[row + row_inc, col + col_inc] == player:
            row += row_inc
            col += col_inc
        num = 1
        while 0 <= row - row_inc < ROWS and 0 <= col - col_inc < COLS and \
                self.grid[row - row_inc, col - col_inc] == player:
            row -= row_inc
            col -= col_inc
            num += 1
        return num

    def check_winning_move(self, row : int, col : int) -> bool:
        """Check if the move at given row, col was a winning move"""
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

    def move (self, col : int) -> int:
        """Called to let the current player make a move at colum col
        Returns the row number the peice landed in, or -1 if it was an invalid move"""

        # First check if the game is already over, or if it's an invalid move
        if self.current_player == 0 or self.grid[0, col] != 0:
            return -1

        # Else drop the current players peice into the selected column
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


class ConnectFourBoard(tk.Canvas):  # pylint: disable=too-many-ancestors
    """ConnectFourBoard - main control of the GUI
    Provides a 6x7 canvas round peices in a given background color
    Provides an API to set the peice at a given (row, col) to a given color
    on_click callback allows user to find out when a column is clicked
    """
    def __init__ (self, on_click : callable):
        super().__init__(bg=BACKGROUND)
        self.fill_color = BLANK_COLOR
        self.board = np.empty([ROWS, COLS], dtype=tk.Canvas)
        self.peice_ids = np.zeros([ROWS, COLS], dtype=int)
        self.on_click = on_click
        tile_size = 60
        peice_size = tile_size * 5 // 6     # Circles in each tile are 5/6 of the tile size
        for row in range (ROWS):
            for col in range (COLS):
                canvas_tile = tk.Canvas(self, bg=BACKGROUND, height=tile_size, width=tile_size,
                    highlightthickness=0)
                self.board[row, col] = canvas_tile
                padding = 2
                peice_id = canvas_tile.create_oval((padding, padding,
                    peice_size + padding, peice_size + padding),
                    outline=OUTLINE_COLOR, fill=BLANK_COLOR)
                self.board[row, col] = canvas_tile
                self.peice_ids[row, col] = peice_id
                canvas_tile.grid(row=row, column=col, padx=5, pady=5)
                canvas_tile.bind('<Button-1>', lambda e, col=col : on_click (col))

    def set_color (self, row : int, col : int, color : str):
        """Set the color of the peice at the given row and column"""
        self.board[row, col].itemconfig(self.peice_ids[row, col], fill=color)

    def reset (self):
        """Reset the board for a new game"""
        for row in range (self.board.shape[0]):
            for col in range (self.board.shape[1]):
                id = self.peice_ids[row, col]
                self.board[row, col].itemconfig(id, fill=self.fill_color)

# Game GUI
class GameScreen(tk.Tk):
    """ The main GUI for the game; some buttons, a grid, and a status panel"""
    def __init__(self):
        super().__init__()
        self.iconbitmap(default='ConnectFour.ico')
        self.title("Connect Four")

        # Create buttons at the top
        tk.Button (self, text="New game", command=self.on_new_game_click).grid(row=0, column=0)

        help_btn = tk.Button (self, text="Help", command=self.on_help_click)
        help_btn.grid(row=0, column=1)

        tk.Label(self, text="|").grid (row=0, column=2)
        tk.Label(self, text="Your color:").grid (row=0, column=3)
        self.color_select_btn = tk.Button (self, text=PLAYER_COLORS[0], width = 7,
            command=self.on_color_select_click)
        self.color_select_btn.grid(row=0, column=4)

        tk.Label(self, text="|").grid (row=0, column=5)
        tk.Label(self, text="Your opponent:").grid (row=0, column=6)
        self.opponent_select_btn = tk.Button (self, text=OPPONENTS[2], width = 13,
            command=self.on_opponent_select_click)
        self.opponent_select_btn.grid(row=0, column=7)

        # Main canvas for the game board in the middle
        self.board = ConnectFourBoard (self.on_board_clicked)
        self.board.grid(row=1, column=0, columnspan=8)

        # Game state info at the bottom
        self.game_state_label = tk.Label(self, text="")
        self.game_state_label.grid(row=2, column=0, columnspan=8, pady=5)

        # Window shouldn't be resizable, since widgets are not
        self.resizable(False, False)

        # And finally, start a game
        self.new_game()

    def new_game (self):
        """Start a new game"""
        # Reset the Canvas to no pieces played
        self.board.reset()

        # Create a new game, selected opponent, selected color, and randomize who goes first
        agent_ix = OPPONENTS.index(self.opponent_select_btn["text"])
        agent = AGENTS[agent_ix]
        if self.color_select_btn["text"] == PLAYER_COLORS[1]:
            self.game = Game(agent, None)
        else:
            self.game = Game(None, agent)

        # Set who goes first - if it's the computer, have them move
        self.game.current_player = random.randint(1, 2)
        self.game_state_label["text"] = "New game: " + self.game.current_player_color \
            + " goes first"
        if self.game.is_computers_turn:
            self.make_computer_move()

    def on_help_click(self):
        """Show the help string"""
        messagebox.showinfo("Help",  HELP_TEXT, icon="question")

    def on_new_game_click(self):
        """Start a new game"""
        self.new_game()

    def on_color_select_click(self):
        """Rotate the button name through the PLAYER_COLORS options"""
        color_ix = PLAYER_COLORS.index(self.color_select_btn["text"])
        self.color_select_btn["text"] = PLAYER_COLORS[(color_ix + 1) % len(PLAYER_COLORS)]
        self.new_game()

    def on_opponent_select_click(self):
        """Rotates the button name through the OPPONENTS options"""
        opponent_ix = OPPONENTS.index(self.opponent_select_btn["text"])
        self.opponent_select_btn["text"] = OPPONENTS[(opponent_ix + 1) % len(OPPONENTS)]
        self.new_game()

    def set_message(self, msg : str):
        """Set the game state text at the bottom of the GUI"""
        self.game_state_label["text"] = msg


    def place_tile (self, col) -> bool:
        """Current player places a tile in col; returns bool if move was succesful"""
        # Make the move and reflect it on the board
        player_color = self.game.current_player_color
        is_computers_turn = self.game.is_computers_turn
        row = self.game.move(col)
        if row == -1:   # If they made an invalid move, ignore the click
            return False
        self.board.set_color (row, col, player_color)

        # If the computer made the move, flash the tile to bring attention to the placement
        if is_computers_turn:
            self.after (100, lambda : self.board.set_color(row, col, BLANK_COLOR))
            self.after (200, lambda : self.board.set_color(row, col, player_color))

        # Update the game state text
        if self.game.is_game_over:
            if self.game.is_tie:
                self.game_state_label["text"] = "Game over - it's a tie!"
            else:
                self.game_state_label["text"] = "Game over - " \
                    + self.game.winning_player_color + " wins!"
        else:
            self.game_state_label["text"] = "It's " + self.game.current_player_color + "'s turn"
        return True

    def on_board_clicked(self, col):
        """When column col is clicked, make the move"""
        # If it is the agents turn or the game is over, ignore the click
        if self.game.is_game_over or self.game.is_computers_turn:
            return

        # If it's a persons turn, make the move and reflect it on the board
        if not self.place_tile (col):
            return      # If invalid move, return

        # Let the computer move, after slight delay so computer appears to be thinking
        if not self.game.is_game_over and self.game.is_computers_turn:
            self.after (500, self.make_computer_move)

    def make_computer_move (self):
        """Have the computer make a move"""
        agent = self.game.current_player_agent
        assert agent is not None, "Trying to get computers move, but it's not the computers turn"

        # Get the agents move, after converting parameters to the format required by Agent.py
        obs = SimpleNamespace()
        obs.mark = self.game.current_player
        obs.board = self.game.grid.flatten().tolist()

        config = SimpleNamespace()
        config.rows = ROWS
        config.columns = COLS
        config.inarow = IN_A_ROW

        col = agent (obs, config)

        # Then update the GUI
        self.place_tile(col)

root = GameScreen()
root.mainloop()
