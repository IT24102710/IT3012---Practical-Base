class SimpleReflexAgent:

    def sense_and_act(self, percept: dict) -> str:

        if percept.get('food_here'):
            return 'Stay' 

        if percept.get('wall_ahead'):
            return 'Left'

        return 'Up' 

        
        