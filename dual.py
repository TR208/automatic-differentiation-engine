import numpy as np

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
                self.value/other.value,
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

    def __pow__(self, other):
        if(isinstance(other,Dual)):
            return Dual(
                self.value**other.value,
                other.value*(self.value**(other.value-1))*self.derivative + (self.value**other.value)*other.derivative*np.log(self.value)
            )
        elif (isinstance(other,int) or isinstance(other,float) or isinstance(other,complex)):
            return Dual(
                self.value**other.value,
                other.value*(self.value**(other.value-1))*self.derivative
            )
        else:
            raise f"{type(self)} and {type(other)} are not compatible for the the operation **"

    def __rpow__(self, other):
        if (isinstance(other,int) or isinstance(other,float) or isinstance(other,complex)):
            return Dual(
                other**self.value,
                (other**self.value)*self.derivative*np.log(self.value)
            )
        else:
            raise f"{type(self)} and {type(other)} are not compatible for the the operation **"
x=Dual(4,1)
y=x*x
z=y/x
sub=y-x
add=x+x
print(x)        ## 4+1e
print(y)        ## 16+8e
print(z)        ## 4+1e
print(sub)      ## 12+7e
print(add)      ## 8+2e
        