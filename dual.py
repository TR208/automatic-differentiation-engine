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
    # Member 3 code
    
    def sin(self): 
        return Dual(
            np.sin(self.value),
            np.cos(self.value)*self.derivative
        )

    def cos(self):
        return Dual(
            np.cos(self.value),
            -np.sin(self.value)*self.derivative
        )

    def tan(self):
        cos_val = np.cos(self.value)
        if np.isclose(cos_val, 0.0, atol=1e-15):
            raise ValueError(f"Domain error: tan(x) is undefined at x = {self.value} (cos(x) = 0).")
        return Dual(
            np.tan(self.value),
            self.derivative / np.cos(self.value)**2
        )
    
    def exp(self):
        val = np.exp(self.value)
        return Dual(val, val * self.derivative)
    
    def log(self):
        if isinstance(self.value, complex):
            raise TypeError("Domain error: log is restricted to positive real numbers.")
        if self.value <= 0:
            raise ValueError(f"Domain error: log(x) is undefined for x = {self.value} <= 0.")
        return Dual(
            np.log(self.value),
            self.derivative / self.value
        )


if __name__ == "__main__":
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
    # ================= Member 3 Tests =================
    m = Dual(2.0, 1.0)

    e = m.exp()
    l = m.log()
    s = m.sin()
    c = m.cos()
    t = m.tan()

    print(e)      ## should be --- 7.3891 + 7.3891ε
    print(l)      ## should be --- 0.6931 + 0.5ε
    print(s)      ## should be --- 0.9093 - 0.4161ε
    print(c)      ## should be --- -0.4161 - 0.9093ε
    print(t)      ## should be --- -2.1850 + 5.7744ε

    # Composed functions (Chain Rule)
    f1 = (m.exp()).sin()      # sin(exp(x))
    f2 = (m**2 + 3*m).log()   # log(x^2 + 3x)

    print(f1)     ## should be --- 0.8980 + 3.2508ε
    print(f2)     ## should be --- 2.3026 + 0.7ε
