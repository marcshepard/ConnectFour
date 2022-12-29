"""
ConnectFour.pyw - a game of connect four

GUI derived from https://github.com/Acamol/connect-four
"""

import tkinter as tk
import numpy as np
import platform
from Agent import basic_agent, minimax3_agent

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
P1_COLOR = "red"
P2_COLOR = "black"
BLANK_COLOR = "lightgrey"

# A Player. Player number 0 means "no player". Player 1 and 2 are actual players. Each actual player has agent set to None
# if it's a human, else to an agent callback function
class Player():
    def __init__ (self, player_num : int, agent : callable):
        if player_num == 0:
            self.color = BLANK_COLOR
        elif player_num == 1:
            self.color = P1_COLOR
        elif player_num == 2:
            self.color = P2_COLOR
        else:
            raise ValueError ("player_num must be 0, 1, or 2")
        self.agent = agent
        self.player_num = player_num

# Game engine
class Game():
    def __init__(self, p1_agent, p2_agent):
        self.players = [
            Player(0, None),        # No player
            Player(1, p1_agent),    # Player1
            Player(2, p2_agent),    # Player2
        ]
        self.current_player = 1     # Index of current player; 1 or 2, or 0 if the game is over
        self.winning_player = 0     # If self.current_player == 0, gives result (0 = tie, 1 or 2 if that player won)
        self.grid = np.zeros([ROWS, COLS], dtype=int)   # Grid of player indexes

    @property
    def current_player_color(self) -> str:
        return self.players[self.current_player].color

    @property
    def prev_player_color(self) -> str:
        return self.players[3 - self.current_player].color

    @property
    def current_player_agent(self) -> callable:
        return self.players[self.current_player].agent

    @property
    def current_player_type(self) -> str:
        return "you" if self.players[self.current_player].agent is None else "computer"

    @property
    def is_game_over(self):
        return self.current_player == 0

    @property
    def is_tie(self):
        return self.is_game_over and self.winning_player == 0

    # Count number of items in a row in a given direction that are the same as the peice in a given row and column
    def num_in_a_row (self, row : int, col : int, row_inc : int, col_inc : int):
        player = self.grid[row, col]
        num = 0
        while row + row_inc < ROWS and col + col_inc < COLS and self.grid[row + row_inc, col + col_inc] == player:
            num += 1
            row += row_inc
            col += col_inc
        return num

    # Check if the move at given row, col was a winning move
    def check_winning_move(self, row : int, col : int) -> bool:
        # horizontal
        if self.num_in_a_row(row, col, 1, 0) + self.num_in_a_row(row, col, -1, 0) > IN_A_ROW - 1:
            return True
        # vertical
        if self.num_in_a_row(row, col, 0, 1) + self.num_in_a_row(row, col, 0, -1) > IN_A_ROW - 1:
            return True
        # diagnal
        if self.num_in_a_row(row, col, 1, 1) + self.num_in_a_row(row, col, -1, -1) > IN_A_ROW - 1:
            return True
        if self.num_in_a_row(row, col, 1, -1) + self.num_in_a_row(row, col, -1, 1) > IN_A_ROW - 1:
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

# Game GUI
class GameScreen(tk.Tk):
    def __init__(self):
        super().__init__()
        # TODO - self.iconbitmap(default='logo.ico')
        self.title("Connect Four")

        # Buttons at the top
        new_game_btn = tk.Button (self, text="New game", command=lambda : self.on_help_click())
        opponent_btn = tk.Button (self, text="Computer Opponent - easy", command=lambda : self.on_opponent_click())
        help_btn = tk.Button (self, text="Help", command=lambda : self.on_opponent_click())
        new_game_btn.grid(row=0, column=0)
        opponent_btn.grid(row=0, column=1)
        help_btn.grid(row=0, column=2)

        # Main canvas for the game board
        canvas = tk.Canvas(self, bg=BACKGROUND)
        self.create_board(canvas)
        canvas.grid(row=1, column=0, columnspan=3)

        # Game state info at the bottom
        #self.game = Game(None, basic_agent)
        self.game = Game(None, None)
        self.game_state_label = tk.Label(self, text="It's " + self.game.current_player_color + "'s turn (" + self.game.current_player_type + ")")
        self.game_state_label.grid(row=2, column=0, columnspan=3)

    # Create an empty board with circle shapes for discs - originally all grey (for no play)
    def create_board (self, canvas):
        self.board = np.empty([ROWS, COLS], dtype=tk.Canvas)
        self.buttons = np.zeros([ROWS, COLS], dtype=int)
        for row in range (ROWS):
            for col in range (COLS):
                canvas_tile = tk.Canvas(canvas, bg=BACKGROUND, height=60, width=60, relief="raised", highlightthickness=0)
                self.board[row, col] = canvas_tile
                padding = 2
                id = canvas_tile.create_oval((padding, padding, 50 + padding, 50 + padding), outline=OUTLINE_COLOR, fill=BLANK_COLOR)
                canvas_tile.configure(width=canvas_tile.winfo_reqwidth(), height=canvas_tile.winfo_reqheight())

                self.board[row, col] = canvas_tile
                self.buttons[row, col] = id

                canvas.rowconfigure(row, weight=1)
                canvas.columnconfigure(col, weight=1)

                canvas_tile.grid(row=row, column=col, padx=3, pady=3, sticky=tk.E + tk.W + tk.N + tk.S)
                canvas_tile.bind('<Button-1>', lambda e, id=id, col=col: self.on_canvas_click(e, id, col))

    def on_help_click(self):
        pass
    
    def on_new_game_click(self):
        pass

    def on_opponent_click(self):
        pass

    # Column j was clicked. Make the move
    def on_canvas_click(self, event, id, col):
        print ("Column = ", col)
        print (self.game.grid)
        # If it is the agents turn, ignore the click
        if self.game.current_player_agent is not None:
            return

        # Make the move and reflect it on the board
        row = self.game.move(col)
        if row == -1:   # If they made an invalid move, ignore the click
            return
        self.board[row, col].itemconfig(id, fill=self.game.prev_player_color)

        # Check if the game is over
        if self.game.is_game_over:
            self.game_state_label = "Game over"
            return

        # Set game state to show it is the next players turn
        self.game_state_label["text"] = "It's " + self.game.current_player_color + "'s turn (" + self.game.current_player_type + ")"


        """
        self.text1.config(text="Round {}".format(self.game.round))
        self.text2.config(text="{}'s turn".format(next_player_color))
        if self.game.winner in ["X", "Y"]:
            self.text3.config(text="{} has won!".format(player_color))
            self.text2.config(text="The End")
            for i, j in IT.product(range(self._rows), range(self._cols)):
                self.board[i][j].unbind("<Button-1>")
            self.flash_discs(
                self.game.get_winning_discs(),
                "red" if self.game.winner == 'X' else "yellow",
                "blue")

        if self.game.winner == 'D':
            self.text1.config(text="Round 42")
            self.text2.config(text="The End")
            self.text3.config(text="Draw")
            for i, j in IT.product(range(self._rows), range(self._cols)):
                self.board[i][j].unbind("<Button-1>")
        """

    def flash_discs(self, discs, winner, altcolor):
        """Alternate color of given discs"""
        for disc in discs:
            row = disc[0]
            col = disc[1]
            self.board[row][col].itemconfig(
                self.buttons[row][col], fill=altcolor)

        self.after(250, self.flash_discs, discs, altcolor, winner)


root = GameScreen()
root.mainloop()
