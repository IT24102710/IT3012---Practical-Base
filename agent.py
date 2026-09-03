import random
import heapq
from collections import deque

from SimpleReflexAgent import SimpleReflexAgent
from ModelBasedAgent import ModelBasedAgent
# agent.py
class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        # If standing directly on food, or just wander / move towards coordinates
        pos = percept['agent_pos']
        # Simple heuristic or fallback random sweep
        return random.choice(self.actions_pool)


class SearchAgent:
    """
    A goal-based agent that plans offline using uninformed search
    (BFS, DFS or UCS) over the exposed world model, then executes the
    resulting action sequence one step at a time.
    """

    MOVES = {
        'Up': (0, 1),
        'Down': (0, -1),
        'Left': (-1, 0),
        'Right': (1, 0)
    }

    def __init__(self):
        self.plan = []                
        self.active_algo = 'BFS'     

    # -- helpers -------------------------------------------------------

    def _get_neighbors(self, pos, walls, grid_size):
        """Yield (action, next_pos) pairs for every legal move from pos."""
        x, y = pos
        width, height = grid_size
        for action, (dx, dy) in self.MOVES.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in walls:
                yield action, (nx, ny)

    def _reconstruct_path(self, came_from, start, goal):
        """Walk backwards through the came_from chain to rebuild the action list."""
        actions = []
        node = goal
        while node != start:
            prev_node, action = came_from[node]
            actions.append(action)
            node = prev_node
        actions.reverse()
        return actions



    def bfs_search(self, start_pos, goal_pos, walls, grid_size):
        """Breadth-First Search: FIFO frontier -> explores shallowest nodes first."""
        start, goal = tuple(start_pos), tuple(goal_pos)
        walls = set(tuple(w) for w in walls)

        if start == goal:
            return []

        frontier = deque([start])
        reached = {start}              
        came_from = {}

        while frontier:
            current = frontier.popleft()          # FIFO pop
            for action, neighbor in self._get_neighbors(current, walls, grid_size):
                if neighbor not in reached:
                    reached.add(neighbor)
                    came_from[neighbor] = (current, action)
                    if neighbor == goal:
                        return self._reconstruct_path(came_from, start, goal)
                    frontier.append(neighbor)

        return None  

    def dfs_search(self, start_pos, goal_pos, walls, grid_size):
        """Depth-First Search: LIFO frontier -> explores deepest nodes first."""
        start, goal = tuple(start_pos), tuple(goal_pos)
        walls = set(tuple(w) for w in walls)

        if start == goal:
            return []

        frontier = [start]
        reached = {start}
        came_from = {}

        while frontier:
            current = frontier.pop()               # LIFO pop
            if current == goal:
                return self._reconstruct_path(came_from, start, goal)
            for action, neighbor in self._get_neighbors(current, walls, grid_size):
                if neighbor not in reached:
                    reached.add(neighbor)
                    came_from[neighbor] = (current, action)
                    frontier.append(neighbor)

        return None

    def ucs_search(self, start_pos, goal_pos, walls, grid_size):
        """Uniform-Cost Search: priority queue ordered by path cost g(n)."""
        start, goal = tuple(start_pos), tuple(goal_pos)
        walls = set(tuple(w) for w in walls)

        if start == goal:
            return []

        counter = 0  # tie-breaker so heapq never tries to compare tuples of coords directly
        frontier = [(0, counter, start)]
        best_cost = {start: 0}
        came_from = {}

        while frontier:
            cost, _, current = heapq.heappop(frontier)
            if current == goal:
                return self._reconstruct_path(came_from, start, goal)
            if cost > best_cost.get(current, float('inf')):
                continue  # stale frontier entry
            for action, neighbor in self._get_neighbors(current, walls, grid_size):
                new_cost = cost + 1  # every step has uniform cost of 1 on this grid
                if new_cost < best_cost.get(neighbor, float('inf')):
                    best_cost[neighbor] = new_cost
                    came_from[neighbor] = (current, action)
                    counter += 1
                    heapq.heappush(frontier, (new_cost, counter, neighbor))

        return None


    def sense_and_act(self, percept: dict) -> str:
        if not self.plan:
            all_food = percept.get('all_food', [])
            if not all_food:
                return 'Stay'

            agent_pos = tuple(percept.get('agent_pos', (0, 0)))
            walls = percept.get('walls', [])
            grid_size = percept.get('grid_size', (10, 10))

            # Find the closest food pellet as the goal
            goal = min(
                all_food,
                key=lambda f: abs(f[0] - agent_pos[0]) + abs(f[1] - agent_pos[1])
            )

            if self.active_algo == 'BFS':
                path = self.bfs_search(agent_pos, goal, walls, grid_size)
            elif self.active_algo == 'DFS':
                path = self.dfs_search(agent_pos, goal, walls, grid_size)
            elif self.active_algo == 'UCS':
                path = self.ucs_search(agent_pos, goal, walls, grid_size)
            else:
                raise ValueError(f"Unknown active_algo: {self.active_algo}")

            self.plan = path if path else []

        if not self.plan:
            # No reachable food found from the current position
            return 'Stay'

        return self.plan.pop(0)