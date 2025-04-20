
from approvedimports import *

class DepthFirstSearch(SingleMemberSearch):
    """your implementation of depth first search to extend
    the superclass SingleMemberSearch search.
    Adds  a __str__method
    Over-rides the method select_and_move_from_openlist
    to implement the algorithm
    """

    def __str__(self):
        return "depth-first"

    def select_and_move_from_openlist(self) -> CandidateSolution:
        """void in superclass
        In sub-classes should implement different algorithms
        depending on what item it picks from self.open_list
        and what it then does to the openlist

        Returns
        -------
        next working candidate (solution) taken from openlist
        """

        # create a candidate solution variable to hold the next solution
        next_soln = CandidateSolution()

        # ====> insert your pseudo-code and code below here
        if not self.open_list:
            raise Exception("Open list is empty, no solution found.")

        next _soln = self.open_list.pop()

        self.closed_list.append(next_soln)

        # <==== insert your pseudo-code and code above here
        return next_soln

class BreadthFirstSearch(SingleMemberSearch):
    """your implementation of depth first search to extend
    the superclass SingleMemberSearch search.
    Adds  a __str__method
    Over-rides the method select_and_move_from_openlist
    to implement the algorithm
    """

    def __str__(self):
        return "breadth-first"

    def select_and_move_from_openlist(self) -> CandidateSolution:
        """Implements the breadth-first search algorithm

        Returns
        -------
        next working candidate (solution) taken from openlist
        """
        # create a candidate solution variable to hold the next solution
        next_soln = CandidateSolution()

        # ====> insert your pseudo-code and code below here
        if self.open_list:
            next_soln = self.open_list.pop(0)
            self.closed_list.append(next_soln)
            next_soln.expand()
            self.open_list.extend(next_soln.children)
        # <==== insert your pseudo-code and code above here
        return next_soln

class BestFirstSearch(SingleMemberSearch):
    """Implementation of Best-First search."""

    def __str__(self):
        return "best-first"

    def select_and_move_from_openlist(self) -> CandidateSolution:
        """Implements Best First by finding, popping and returning member from openlist
        with best quality.

        Returns
        -------
        next working candidate (solution) taken from openlist
        """

        # create a candidate solution variable to hold the next solution
        next_soln = CandidateSolution()

        # ====> insert your pseudo-code and code below here
        self.closedlist.append(next_soln)
        for child in next_soln.children:
            if child not in self.closedlist:
                self.openlist.append(child)
        # <==== insert your pseudo-code and code above here
        return next_soln

class BestFirstSearch(SingleMemberSearch):
    """Implementation of Best-First search."""

    def __str__(self):
        return "best-first"

    def select_and_move_from_openlist(self) -> CandidateSolution:
        """Implements Best First by finding, popping and returning member from openlist
        with best quality.

        Returns
        -------
        next working candidate (solution) taken from openlist
        """

        # create a candidate solution variable to hold the next solution
        next_soln = CandidateSolution()

        # ====> insert your pseudo-code and code below here
        self.closedlist.append(next_soln)
        for child in next_soln.children:
            if child not in self.closedlist:
                self.openlist.append(child)
        # <==== insert your pseudo-code and code above here
        return next_soln

class AStarSearch(SingleMemberSearch):
    """Implementation of A-Star  search."""

    def __str__(self):
        return "A Star"

    def select_and_move_from_openlist(self) -> CandidateSolution:
        """Implements A-Star by finding, popping and returning member from openlist
        with lowest combined length+quality.

        Returns
        -------
        next working candidate (solution) taken from openlist
        """
        
        # create a candidate solution variable to hold the next solution
        next_soln = CandidateSolution()

        # ====> insert your pseudo-code and code below here
        self.closedlist.append(next_soln)
        for child in next_soln.expand:
            if child not in self.closedlist:
                self.openlist.append(child)
        # <==== insert your pseudo-code and code above here
        return next_soln
wall_colour= 0.0
hole_colour = 1.0

def create_maze_breaks_depthfirst():
    # ====> insert your code below here
    #remember to comment out any mention of show_maze() before you submit your work
    import random

    def carve_passage(x,y,maze,wall_colour,hole_colour):
        maze.contents[y][x] = hole_colour
        directions = [(0,1),(0,-1),(1,0),(-1,0)]
        random.shuffle(directions)
        for dx,dy in directions:
            if maze.contents[y+dy][x+dx] == wall_colour:
                carve_passage(x+dx,y+dy,maze,wall_colour,hole_colour)
    # <==== insert your code above here

def create_maze_depth_better():
    # ====> insert your code below here
    #remember to comment out any mention of show_maze() before you submit your work
import random
    import numpy as np

    # Maze dimensions
    rows, cols = 10, 10  # Adjust size as needed
    maze = np.ones((rows, cols), dtype=int)  # 1 represents walls
    
    # Initialize the stack and visited set
    stack = []
    visited = set()
    
    # Choose a random starting point and mark it as a passage (0)
    start_x, start_y = random.randrange(rows), random.randrange(cols)
    maze[start_x, start_y] = 0
    stack.append((start_x, start_y))
    visited.add((start_x, start_y))
    
    # Possible movement directions: (row_offset, col_offset)
    directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
    
    while stack:
        x, y = stack[-1]
        random.shuffle(directions)  # Randomize directions to create unique mazes
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in visited:
                maze[nx, ny] = 0  # Mark as passage
                maze[x + dx // 2, y + dy // 2] = 0  # Remove wall in between
                stack.append((nx, ny))
                visited.add((nx, ny))
                break
        else:
            stack.pop()  # Backtrack when no available moves
    
    # return maze for further processing
    return maze
    # <==== insert your code above here
