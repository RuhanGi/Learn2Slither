class Interpreter:
    """
        Interpreter
    """

    @staticmethod
    def getState(vision):
        state = []
        state.extend(
            [1 if vision[i][0] == 'S' or vision[i][0] == 'W' else 0
                for i in range(len(vision))]
        )
        state.extend(
            [1 if 'G' in vision[i] else 0 for i in range(len(vision))]
        )
        state.extend(
            [1 if 'R' in vision[i] else 0 for i in range(len(vision))]
        )
        return state
