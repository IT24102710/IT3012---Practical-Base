class ModelBasedAgent:

    def __init__(self):
        self.visited_cells = set()
        self.current_pos=[0, 0]  # Starting position (x, y)
        self.facing='Up'  # Initial facing direction
        self.last_action = None  # Last action taken

        self.visited_cells.add(tuple(self.current_pos))  # Mark starting position as visited


    def _get_left_dir(self) -> str:
        left_map = {'Up': 'Left', 'Left': 'Down', 'Down': 'Right', 'Right': 'Up'}
        return left_map[self.facing]

    def _get_right_dir(self) -> str:
        right_map = {'Up': 'Right', 'Right': 'Down', 'Down': 'Left', 'Left': 'Up'}
        return right_map[self.facing]

    def _get_ahead_pos(self) -> tuple:
        dx, dy = 0, 0
        if self.facing == 'Up':
            dy = 1
        elif self.facing == 'Down':
            dy = -1
        elif self.facing == 'Left':
            dx = -1
        elif self.facing == 'Right':
            dx = 1
        return (self.current_pos[0] + dx, self.current_pos[1] + dy)


    def sense_and_act(self, percept: dict) -> str:

        if self.last_action=='Left':
            self.facing=self._get_left_dir()
        elif self.last_action=='Right':
            self.facing=self._get_right_dir()
        elif self.last_action=='Up':
            ahead=self._get_ahead_pos()
            self.current_pos=list(ahead)
            self.visited_cells.add(tuple(self.current_pos))

        self.visited_cells.add(tuple(self.current_pos))

        ahead_pos = self._get_ahead_pos()
        wall_ahead = percept.get('wall_ahead', False)
        food_here = percept.get('food_here', False)

        if food_here:
            action='Stay'

        elif wall_ahead:
            left_dir = self._get_left_dir()

            offsets={
                'Up': (0, 1),
                'Down': (0, -1),
                'Left': (-1, 0),
                'Right': (1, 0)
            }

            lx, ly = offsets[left_dir]
            left_pos = (self.current_pos[0] + lx, self.current_pos[1] + ly)

            if left_pos not in self.visited_cells:
                action='Right'

            else:
                action='Left'

        elif ahead_pos in self.visited_cells:
            action='Left'
        else:
            action='Up'

        self.last_action=action
        return action