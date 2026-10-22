class Dual:
    def __init__(self, value, derivative):
        self.value = value
        self.derivative = derivative

    def __repr__(self):
        return f"{self.value} + {self.derivative}ε"
    