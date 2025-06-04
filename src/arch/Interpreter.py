class Interpreter:
    """
        Interpreter
    """

    @staticmethod
    def getState(vision):
        state = []
        state.extend(
            [1 if len(vision[i]) == 0 or vision[i][0] == 'S' or vision[i][0] == 'W' else 0
                for i in range(len(vision))]
        )
        state.extend(
            [1 if 'G' in vision[i] else 0 for i in range(len(vision))]
        )
        state.extend(
            [1 if 'R' in vision[i] else 0 for i in range(len(vision))]
        )
        return state
    
    #TODO Achieving a higher length at the end of a session (15, 20, 25, 30, 35).
