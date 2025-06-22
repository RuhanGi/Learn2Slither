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
            # [Interpreter.getDist(vision[i]) for i in range(len(vision))]
        )
        state.extend(
            [1 if Interpreter.getDist(vision[0]) > Interpreter.getDist(vision[2]) else 0,
             1 if Interpreter.getDist(vision[1]) > Interpreter.getDist(vision[3]) else 0])
        state.extend(
            [1 if 'G' in vision[i] else 0 for i in range(len(vision))]
        )
        state.extend(
            [1 if 'R' in vision[i] else 0 for i in range(len(vision))]
        )
        return state

    @staticmethod
    def getDist(sight):
        for i in range(len(sight)):
            if sight[i] == 'S' or sight[i] == 'W':
                return i
        return len(sight)
