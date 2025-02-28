import pygame
import numpy as np
from collections import deque

class Minesweeper:
    def __init__(self, rowdim, coldim, mine_count, gui=False):
        self.rowdim = rowdim
        self.coldim = coldim
        self.mine_count = mine_count
        self.minefield = np.zeros((rowdim, coldim), dtype='int')
        self.playerfield = np.ones((rowdim, coldim), dtype='int') * 9
        self.explosion = False
        self.done = False
        self.score = 0
        self.np_random = np.random.RandomState()
        self.move_num = 0
        if gui:
            self.init_gui()

    def generate_field(self, first_action):
        """
        Place mines randomly except for the first revealed tile.
        first_action is a flattened index where the user clicked first.
        """
        first_row, first_col = divmod(first_action, self.coldim)
        # Create a list of all possible tile indices except the first click
        all_positions = [(r, c) for r in range(self.rowdim) for c in range(self.coldim)
                         if not (r == first_row and c == first_col)]
        # Randomly choose positions for mines
        mine_positions = self.np_random.choice(len(all_positions), self.mine_count, replace=False)
        # Set mines
        for pos_index in mine_positions:
            r, c = all_positions[pos_index]
            self.minefield[r, c] = -1
        # Calculate adjacent mine counts
        for r in range(self.rowdim):
            for c in range(self.coldim):
                if self.minefield[r, c] == -1:
                    continue
                self.minefield[r, c] = self.count_adjacent_mines(r, c)

    def count_adjacent_mines(self, row, col):
        """
        Return the number of mines surrounding the tile at (row, col).
        """
        count = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                rr = row + dr
                cc = col + dc
                if 0 <= rr < self.rowdim and 0 <= cc < self.coldim:
                    if self.minefield[rr, cc] == -1:
                        count += 1
        return count

    def auto_reveal_tiles(self, action):
        """
        Perform a flood fill from the clicked tile if it is 0 (i.e., no adjacent mines).
        Reveals all connected zero tiles and their non-zero neighbors.
        """
        stack = deque([action])
        revealed_count = 0
        while stack:
            current = stack.pop()
            row, col = divmod(current, self.coldim)
            if self.playerfield[row, col] == 9:  # If hidden, reveal now
                self.playerfield[row, col] = self.minefield[row, col]
                revealed_count += 1
                # If this tile is also 0, reveal neighbors
                if self.minefield[row, col] == 0:
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            rr = row + dr
                            cc = col + dc
                            if 0 <= rr < self.rowdim and 0 <= cc < self.coldim:
                                if self.playerfield[rr, cc] == 9:  # Still hidden
                                    stack.append(rr * self.coldim + cc)
        # Flatten the updated self.playerfield for uniform handling in step
        flattened = self.playerfield.flatten()
        return flattened, revealed_count

    def step(self, action):
        # If this is the first move, generate a new board with mines placed
        if self.move_num == 0:
            self.generate_field(action)

        # Flatten current playerfield and actual minefield for indexing
        state = self.playerfield.flatten()
        minefield_state = self.minefield.flatten()

        # Reveal the clicked tile
        state[action] = minefield_state[action]

        # Count how many remain hidden
        num_hidden_tiles = np.count_nonzero(state == 9)

        # If this tile was a mine, game over
        if state[action] == -1:
            self.done = True
            self.explosion = True
            reward = -1
            score = 0
            # Reveal all mines and mark the exploded one
            state = self.minefield.flatten()
            state[action] = -2  # -2 to denote the exploded mine
            state = state.reshape(self.rowdim, self.coldim)
            self.playerfield = state

        # If all non-mine tiles revealed, game is won
        elif num_hidden_tiles == self.mine_count:
            self.done = True
            reward = 1.0
            score = 1

        # If tile is 0, auto reveal
        elif state[action] == 0:
            state, revealed_count = self.auto_reveal_tiles(action)
            num_hidden_tiles = np.count_nonzero(state == 9)
            self.done = (num_hidden_tiles == self.mine_count)
            reward = 1.0 if self.done else 0.1
            score = revealed_count  # or some weighting if preferred

        # Otherwise, just a normal reveal
        else:
            self.done = False
            reward = 0.1
            score = 1

        # Update fields if not done
        if not self.done:
            state = state.reshape(self.rowdim, self.coldim)
            self.playerfield = state

        self.score += score
        self.move_num += 1
        return self.playerfield, reward, self.done

    def reset(self):
        self.score = 0
        self.move_num = 0
        self.explosion = False
        self.done = False
        self.minefield = np.zeros((self.rowdim, self.coldim), dtype='int')
        self.playerfield = np.ones((self.rowdim, self.coldim), dtype='int') * 9
        return self.playerfield

    def init_gui(self):
        pygame.init()
        pygame.mixer.quit()  # Stop any sound if needed
        self.tile_rowdim = 32
        self.tile_coldim = 32
        self.game_width = self.coldim * self.tile_coldim
        self.game_height = self.rowdim * self.tile_rowdim
        self.ui_height = 32
        self.gameDisplay = pygame.display.set_mode((self.game_width, self.game_height + self.ui_height))
        pygame.display.set_caption('Minesweeper')
        # Load tile images
        self.tile_dict = {
            -2: pygame.image.load('img/explode.jpg').convert(),  # Exploded mine
            -1: pygame.image.load('img/mine.jpg').convert(),     # Hidden mine
            0: pygame.image.load('img/0.jpg').convert(),
            1: pygame.image.load('img/1.jpg').convert(),
            2: pygame.image.load('img/2.jpg').convert(),
            3: pygame.image.load('img/3.jpg').convert(),
            4: pygame.image.load('img/4.jpg').convert(),
            5: pygame.image.load('img/5.jpg').convert(),
            6: pygame.image.load('img/6.jpg').convert(),
            7: pygame.image.load('img/7.jpg').convert(),
            8: pygame.image.load('img/8.jpg').convert(),
            9: pygame.image.load('img/hidden.jpg').convert()     # Hidden tile
        }
        self.myfont = pygame.font.SysFont('Segoe UI', 24)
        self.font_color = (255, 255, 255)

    def render(self):
        self.gameDisplay.fill((0, 0, 0))
        # Draw each tile
        for row in range(self.rowdim):
            for col in range(self.coldim):
                tile_value = self.playerfield[row, col]
                self.gameDisplay.blit(self.tile_dict[tile_value],
                                      (col * self.tile_coldim, row * self.tile_rowdim))

        # Display Moves and Score
        text = self.myfont.render(f'Moves: {self.move_num}   Score: {self.score}', True, self.font_color)
        self.gameDisplay.blit(text, (10, self.game_height + 2))

        # Display end-game message
        if self.done:
            msg = 'VICTORY!' if not self.explosion else 'DEFEAT!'
            color = (0, 255, 0) if not self.explosion else (255, 0, 0)
            text = self.myfont.render(msg, True, color)
            # Center the message roughly
            self.gameDisplay.blit(text, (self.game_width // 2 - 50, self.game_height + 2))

        pygame.display.update()

    def close(self):
        pygame.quit()

def main():
    rows, cols, mines = 9, 9, 10
    env = Minesweeper(rows, cols, mines, gui=True)
    env.reset()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and not env.done:
                x, y = pygame.mouse.get_pos()
                if y < env.game_height:  # Only register clicks inside the grid
                    col = x // env.tile_coldim
                    row = y // env.tile_rowdim
                    if 0 <= row < rows and 0 <= col < cols:
                        action = row * cols + col
                        env.step(action)

        env.render()
        pygame.time.wait(10)

    env.close()

if __name__ == "__main__":
    main()
