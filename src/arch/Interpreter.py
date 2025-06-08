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
        foresight = 4
        reach = 0
        while reach < min(len(sight), foresight):
            if sight[reach] == 'S' or sight[reach] == 'W':
                break
            reach += 1
        return round(1 - reach / foresight, 2)

    #TODO Achieving a higher length at the end of a session (15, 20, 25, 30, 35).
