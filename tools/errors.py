"""___Classes_______________________________________________________________"""

class BrainWaveEngineError(Warning):
    pass

class InitUnknownError(BrainWaveEngineError):
    pass

class WrongInputSize(BrainWaveEngineError):
    pass
