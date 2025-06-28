def getDist(sight):
    for i in range(len(sight)):
        if sight[i] == 'S' or sight[i] == 'W':
            return i
    return len(sight)


def checkDanger(visioni):
    return len(visioni) == 0 or visioni[0] == 'S' or visioni[0] == 'W'


class Interpreter:
    """
        Interpreter
    """

    @staticmethod
    def getState(vision):
        state = []
        state.extend([
            1 if checkDanger(vision[i]) else 0
            for i in range(len(vision))
        ])
        state.extend(
            [1 if getDist(vision[0]) > getDist(vision[2]) else 0,
             1 if getDist(vision[1]) > getDist(vision[3]) else 0])
        state.extend(
            [1 if 'G' in vision[i] else 0 for i in range(len(vision))]
        )
        state.extend(
            [1 if 'R' in vision[i] else 0 for i in range(len(vision))]
        )
        return state
