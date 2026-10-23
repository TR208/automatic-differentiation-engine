class Dual:
    def __init__(self, value, derivative):
        self.value = value
        self.derivative = derivative

    def __repr__(self):
        return f"{self.value} + {self.derivative}ε"

    def __add__(self,other):
        if isinstance(other,Dual):
            return Dual(self.value+other.value,self.derivative+other.derivative)
        elif(isinstance(other,int) or isinstance(other,float) or isinstance(other,complex)):
            return Dual(self.value+other,self.derivative)
        else:
            raise f"{type(self)} and {type(other)} are not compatible for the the operation +"

    def __radd__(self, other):
        return self+other

    def __sub__(self, other):
        if isinstance(other,Dual):
            return Dual(self.value-other.value,self.derivative-other.derivative)
        elif(isinstance(other,int) or isinstance(other,float) or isinstance(other,complex)):
            return Dual(self.value-other,self.derivative)
        else:
            raise f"{type(self)} and {type(other)} are not compatible for the the operation -"

    def __rsub__(self, other):
        return (-1)*self+other

    def __mul__(self, other):
        if isinstance(other,Dual):
            return Dual(
                self.value*other.value,
                self.derivative*other.value +self.value*other.derivative
            )
        elif(isinstance(other,int) or isinstance(other,float) or isinstance(other,complex)):
            return Dual(
                self.value*other,
                self.derivative*other
            )
        else:
            raise f"{type(self)} and {type(other)} are not compatible for the the operation *"

    def __rmul__(self, other):
        return self*other

    def __truediv__(self, other):
        if isinstance(other,Dual):
            return Dual(
                self.value*other.value,
                (self.derivative*other.value-other.derivative*self.value)/((other.value) **2)
            )
        elif(isinstance(other,int) or isinstance(other,float) or isinstance(other,complex)):
            return self*(1/other)
        else:
            raise f"{type(self)} and {type(other)} are not compatible for the the operation /"

    def __rtruediv__(self, other):
        if (isinstance(other,int) or isinstance(other,float) or isinstance(other,complex)):
            return Dual(
                other/self.value,
                -other/(self.derivative**2))
            
        else:
            raise f"{type(self)} and {type(other)} are not compatible for the the operation /"

    

        