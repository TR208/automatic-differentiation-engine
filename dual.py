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
            raise TypeError(f"{type(self)} and {type(other)} are not compatible for the the operation +")

    def __radd__(self, other):
        return self+other

    def __sub__(self, other):
        if isinstance(other,Dual):
            return Dual(self.value-other.value,self.derivative-other.derivative)
        elif(isinstance(other,int) or isinstance(other,float) or isinstance(other,complex)):
            return Dual(self.value-other,self.derivative)
        else:
            raise TypeError(f"{type(self)} and {type(other)} are not compatible for the the operation -")

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
            raise TypeError(f"{type(self)} and {type(other)} are not compatible for the the operation *")

    def __rmul__(self, other):
        return self*other

    def __truediv__(self, other):
        if isinstance(other,Dual):
            try:
                return Dual(
                    self.value/other.value,
                    (self.derivative*other.value-other.derivative*self.value)/((other.value) **2)
                )
            except ZeroDivisionError:
                raise ZeroDivisionError("Error! You cannot divide with a function whose value is 0.")
        
        elif(isinstance(other,int) or isinstance(other,float) or isinstance(other,complex)):
            return self*(1/other)
        else:
            raise TypeError(f"{type(self)} and {type(other)} are not compatible for the the operation /")

    def __rtruediv__(self, other):
        if (isinstance(other,int) or isinstance(other,float) or isinstance(other,complex)):
            try:
                return Dual(
                    other/self.value,
                    -other*self.derivative/(self.value**2))
            except ZeroDivisionError:
                raise ZeroDivisionError("Error! You cannot divide with a function whose value is 0.")
            
        else:
            raise TypeError(f"{type(self)} and {type(other)} are not compatible for the the operation /")

    def __pow__(self, other):
        if(isinstance(other,Dual)):
            if(self.value==0 and other.value==0):
                raise ValueError("Raising 0 to the power of 0 is not defined.")
            return Dual(
                self.value**other.value,
                other.value*(self.value**(other.value-1))*self.derivative + (self.value**other.value)*other.derivative*np.log(self.value,dtype=complex)
            )
        elif (isinstance(other,int) or isinstance(other,float) or isinstance(other,complex)):
            if(self.value==0 and other==0):
                raise ValueError("Raising 0 to the power of 0 is not defined.")
            return Dual(
                self.value**other,
                other*(self.value**(other-1))*self.derivative
            )
        else:
            raise TypeError(f"{type(self)} and {type(other)} are not compatible for the the operation **")

    def __rpow__(self, other):
        if (isinstance(other,int) or isinstance(other,float) or isinstance(other,complex)):
            return Dual(
                other**self.value,
                (other**self.value)*self.derivative*np.log(other,dtype=complex)
            )
        else:
            raise TypeError(f"{type(self)} and {type(other)} are not compatible for the the operation **")
x=Dual(4,1)
y=x*x
z=y/x
sub=y-x
add=x+x
exp1=x**2
exp2=exp1**x
print(x)        ##should be --- 4+1e
print(y)        ##should be --- 16+8e
print(z)        ##should be --- 4+1e
print(sub)      ##should be --- 12+7e
print(add)      ##should be --- 8+2e
print(exp1)     ##should be --- 16+8e        
print(exp2)     ##should be --- 65536+312,776.4e