from data import Data
class HW:
    """呼叫以前的功課"""
    def __init__(self) -> None:
        self.data = Data().data_encoder()
    

    def decision(self):
        """決策樹"""
        