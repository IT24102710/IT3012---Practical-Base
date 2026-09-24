import random
import heapq
import math
from collections import deque

from SimpleReflexAgent import SimpleReflexAgent
from ModelBasedAgent import ModelBasedAgent
from logic_engine import KnowledgeBase
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

    # Maps an action name to its (dx, dy) effect on the grid coordinates.
    # These MUST match the movement logic in VisualGridHuntGame.execute_action.
    MOVES = {
        'Up': (0, 1),
        'Down': (0, -1),
        'Left': (-1, 0),
        'Right': (1, 0)
    }

    def __init__(self):
        self.plan = []                # Step 1.3.1: holds the queued sequence of actions
        self.active_algo = 'BFS'      # Step 1.3.1: 'BFS', 'DFS', or 'UCS'

        # Part 3 (Practical 05): Knowledge Base for logical feasibility checking
        self.kb = KnowledgeBase()

        # Rule 1: TargetVisible ∧ HasDust ⇒ SafeToEngage
        self.kb.tell_rule(['TargetVisible', 'HasDust'], 'SafeToEngage')
        # Rule 2: SafeToEngage ∧ BloodseekerMissing ⇒ Retreat
        self.kb.tell_rule(['SafeToEngage', 'BloodseekerMissing'], 'Retreat')

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
        reached = {start}              # the 'reached' set: prevents Tree Search -> infinite loops
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

        return None  # No path found

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

 
    def manhattan_distance(self, pos, goal):
        """h(n) = |x1 - x2| + |y1 - y2|  -- admissible for 4-way (no diagonal) movement."""
        x1, y1 = pos
        x2, y2 = goal
        return abs(x1 - x2) + abs(y1 - y2)
 
    def euclidean_distance(self, pos, goal):
        """h(n) = sqrt((x1 - x2)^2 + (y1 - y2)^2) -- straight-line distance."""
        x1, y1 = pos
        x2, y2 = goal
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
 
 
    def astar_search(self, start_pos, goal_pos, walls, grid_size, heuristic_type='manhattan',
                     percept=None):
        """
        A* Search: f(n) = g(n) + h(n).
        Frontier entries: (f_cost, g_cost, current_pos, path_taken)

        Step 3.2 (Practical 05): Before expanding a neighbor, the KB is
        consulted to verify the tile is logically *Feasible* (not just
        physically reachable).  If 'Retreat' is inferred, the tile is
        marked Infeasible and skipped.
        """
        start, goal = tuple(start_pos), tuple(goal_pos)
        walls = set(tuple(w) for w in walls)
 
        heuristic_fn = self.manhattan_distance if heuristic_type == 'manhattan' else self.euclidean_distance
 
        if start == goal:
            return []
 
        # Build helper sets from the percept for fast tile-level KB assertions
        food_set = set(tuple(f) for f in (percept.get('all_food', []) if percept else []))
        opponent_set = set(tuple(o) for o in (percept.get('opponents', []) if percept else []))
        toxic_set = set(tuple(t) for t in (percept.get('toxic_traps', []) if percept else []))

        counter = 0  # tie-breaker so heapq never compares path lists directly
        g_start = 0
        h_start = heuristic_fn(start, goal)
        f_start = g_start + h_start
 
        frontier = [(f_start, g_start, counter, start, [])]
        reached_states = set()
 
        while frontier:
            f_cost, g_cost, _, current_pos, path_taken = heapq.heappop(frontier)
 
            if current_pos == goal_pos or current_pos == goal:
                return path_taken
 
            if current_pos in reached_states:
                continue
            reached_states.add(current_pos)
 
            for action, neighbor in self._get_neighbors(current_pos, walls, grid_size):
                if neighbor in reached_states:
                    continue

                # ----------------------------------------------------------
                # Step 3.2: KB Feasibility Check
                # 1. Clear all previous percept-facts for this tile evaluation
                self.kb.clear_facts()

                # 2. Feed tile-specific percepts into the KB
                if neighbor in food_set:
                    self.kb.tell_fact('TargetVisible')
                if neighbor in toxic_set:
                    # A toxic trap acts like a HasDust threat
                    self.kb.tell_fact('HasDust')
                    self.kb.tell_fact('BloodseekerMissing')   # no safe escort → must Retreat
                if neighbor in opponent_set:
                    # An opponent tile also triggers all threat signals
                    self.kb.tell_fact('TargetVisible')
                    self.kb.tell_fact('HasDust')
                    self.kb.tell_fact('BloodseekerMissing')

                # 3. Run the inference engine
                self.kb.forward_chain()

                # 4. If 'Retreat' is deduced, the tile is Infeasible — skip it
                if 'Retreat' in self.kb.facts:
                    continue   # logically infeasible: do not add to open list
                # ----------------------------------------------------------

                g_new = g_cost + 1
                h_new = heuristic_fn(neighbor, goal)
                f_new = g_new + h_new
                counter += 1
                heapq.heappush(frontier, (f_new, g_new, counter, neighbor, path_taken + [action]))
 
        return None  # No path found

 

    def sense_and_act(self, percept: dict) -> str:
        if not self.plan:
            all_food = percept.get('all_food', [])
            if not all_food:
                return 'Stay'

            agent_pos = tuple(percept.get('agent_pos', (0, 0)))
            walls = percept.get('walls', [])
            grid_size = percept.get('grid_size', (10, 10))
            remaining_food = percept.get('remaining_food', len(all_food))  # global state pulled from percept

            # Find the closest food item (Manhattan distance) to act as the goal_pos
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
            elif self.active_algo == 'AStar':
                path = self.astar_search(agent_pos, goal, walls, grid_size,
                                         heuristic_type='manhattan', percept=percept)
            else:
                raise ValueError(f"Unknown active_algo: {self.active_algo}")

            self.plan = path if path else []

        if not self.plan:
            # No reachable food found from the current position
            return 'Stay'

        return self.plan.pop(0)