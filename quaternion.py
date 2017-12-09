''' -*- Mode: Python; tab-width: 4; python-indent: 4; indent-tabs-mode: nil -*- '''

# Based on java package ca.uwaterloo.alumni.dwharder.Numbers
# Some documentation (although for the Java implementation) at:
# https://ece.uwaterloo.ca/~dwharder/Java/doc/ca/uwaterloo/alumni/dwharder/Numbers/Quaternion.html

import math
import pdb
import random

import complex
LOG10      = math.log(10.0)
PI2        = 1.5707963267948966   # pi/2.0
NaN        = float("NaN")
Infinity   = float('inf')
# isIterable, below, from http://bytes.com/topic/python/answers/514838-how-test-if-object-sequence-iterable
isIterable = lambda v: bool(getattr(v, '__iter__', False)) # Turns out we don't want strings so omit that
isInfinite = lambda r: math.isinf(r)
isNaN      = lambda r: math.isnan(r)
qrint      = lambda x: math.floor(x+0.499999999999999)
sign       = lambda r: math.copysign(1.0,r)
sqr        = lambda r: r*r

actions    = []                 # Filled in by @canuse decorator


def canuse(func):
    'Decorator to collect functions we can apply'
    actions.append(func)
    return func


def catch_None(f):
    'Decorator to catch returned None value and display arguments from the call'
    def deco_f(*args, **kwargs):
        retv  =  f(*args, **kwargs)
        if retv is None:
            print "*** None returned from %s(%s)" % (f.__name__, args,)
        return retv
    return deco_f


class Quaternion(object):
    'Hamiltonian numbers: i^2 = j^2  = k^2 = ijk = -1'

    units  = ['i', 'j', 'k']

    @staticmethod
    def swap2(s, subs):
        'Replace strXYstr with strYXstr'
        i = 0

        while True:
            j  = s[i:].find(subs)
            if j < 0: return s
            i += j
            s  = s[:i] + s[i+1] + s[i] + s[i+2:]
            i += 2              # Always move forward so as not to loop


    def __repr__(self):
        qs = "Quaternion(%10.5f %+10.5fi %+10.5fj %+10.5fk)" % (self.r, self.i, self.j, self.k)
        return qs


    def __str__(self):
        qs = "%10.4f %+10.4fi %+10.4fj %+10.4fk" % (self.r, self.i, self.j, self.k)
        qs = Quaternion.swap2(qs, " +")
        qs = Quaternion.swap2(qs, " -")
        return qs


    def  __init__(self, *args):
        '''
        >>> print Quaternion(2.71828, 3.1416, 1.618, -0.101001) 
            2.7183   + 3.1416i   + 1.6180j   - 0.1010k
        >>> print Quaternion(2.71828)
            2.7183   + 0.0000i   + 0.0000j   + 0.0000k
        >>> print Quaternion( 3.1416, 1.618, -0.101001)
            0.0000   + 3.1416i   + 1.6180j   - 0.1010k
        >>> print Quaternion( [1.11, -2.222, -3.3333])
            0.0000   + 1.1100i   - 2.2220j   - 3.3333k
        >>> print Quaternion( [1.11, 22.222, -333.33, 4444.44])
            1.1100  + 22.2220i - 333.3300j+ 4444.4400k
        >>> print Quaternion( 1287.06, [1.11, 22.222, -333.33])
         1287.0600   + 1.1100i  + 22.2220j - 333.3300k
        '''

        'Principal entry point to create and initialize a quaternion.'

        if len(args) == 4:
            self.r = args[0]
            self.i = args[1]
            self.j = args[2]
            self.k = args[3]
        elif len(args) == 1 and not isIterable(args[0]): # Below we handle args[0]==vector
            self.r = args[0]
            self.i = 0
            self.j = 0
            self.k = 0
        elif len(args) == 3:
            self.r = 0
            self.i = args[0]
            self.j = args[1]
            self.k = args[2]
        elif len(args) == 0:    # So quaternion.Quaternion() creates Q(0,0,0,0)
            self.r = self.i = self.j = self.k = 0
        elif len(args) == 1 and isIterable(args[0]): # Handle args[0]==vector
            v = args[0]                              # Some kind of iterable
            if len(v) == 3:                          # 3-element vector: imaginaries
                self.r = 0.0
                self.i = v[0]
                self.j = v[1]
                self.k = v[2]
            elif len(v) == 4:   # 4-element vector: all coefficients
                self.r = v[0]
                self.i = v[1]
                self.j = v[2]
                self.k = v[3]
        elif len(args) == 2 and isIterable(args[1]): # (real), [i,j,k vector]
            v = args[1]
            if len(v) == 3:     # Only allow real and vector of 3 imaginaries
                self.r = args[0]
                self.i = v[0]
                self.j = v[1]
                self.k = v[2]
        else:
            raise ValueError,"Cannot create Quaternion with arguments %s" % (args,)

        return None


    @staticmethod
    def random(*args):               # Quaternion of random values
        if len(args) == 0:           # i.e., random() (BUT NOT random(0)!!!)
            return Quaternion( math.random(), math.random(), math.random(), math.random() )
        elif len(args) == 1 and (isinstance(args,int) or isinstance(args,long)):
            n = args[0]
            if n == 0:          # THIS is random(0), not random()
                return Quaternion(math.random(), 0.0, 0.0, 0.0);
            if n == 1: 
                return Quaternion(0.0, math.random(), 0.0, 0.0);
            if n == 2: 
                return Quaternion(0.0, 0.0, math.random(), 0.0);
            if n == 3: 
                return Quaternion(0.0, 0.0, 0.0, math.random());
            if n > 3:
                raise ValueError("Coefficient index (%d) greater than 3 in random() call")
        elif len(args) == 1 and isIterable(args[0]):
            v = args[0]
            if len(v) == 4:
                qc = []
                for j in range(4):
                    if v[j]: 
                        qc.append(random.random())
                    else: qc.append(0)
                return Quaternion(qc) # Take my 4-element vector, please
            else:
                raise ValueError("quaternion.random(vector[]) must have length 4 vector, not %d" % len(v))


    @staticmethod
    def randomReal():
        return Quaternion(random.random(), 0.0, 0.0, 0.0)


    @staticmethod
    def randomImaginary():
        return Quaternion(0.0, random.random(), random.random(), random.random())


    @staticmethod
    def sqr(x):
        return x * x


    @staticmethod
    def sign(x):
        if x > 0.0:
            return 1.0
        
        if x < 0.0:
            return -1.0
        
        if x == 0.0:
            if quaternion.isPositiveZero(x):
                return  1.0
            else:
                return -1.0


    @staticmethod
    def isPositiveZero(x):
        return math.copysign(1,x)


    def makeQuaternion(self, *args):
        '''
        >>> print Quaternion(1.0, 2.0, 3.0, 4.0).makeQuaternion(47.201)
           47.2010   + 2.0000i   + 3.0000j   + 4.0000k
        >>> print Quaternion(1.0, 2.0, 3.0, 4.0).makeQuaternion(complex.Complex(601.44, 32.919))
          601.4400  + 32.9190i   + 3.0000j   + 4.0000k
        >>> print Quaternion(1.0, 2.0, 3.0, 4.0).makeQuaternion(90.807, 10.001)
           90.8070  + 20.0020i  + 30.0030j  + 40.0040k
        '''
        if len(args) == 1:      # float or Complex
            if isinstance(args[0], float):
                return Quaternion(args[0], self.i, self.j, self.k)
            elif isinstance(args[0],complex.Complex):
                return Quaternion(args[0].r, args[0].i, self.j, self.k)
        if len(args) == 2:
            rreal = args[0]
            iimag = args[1]
            if isNaN(iimag) or isInfinite(iimag): # Propagate crap
                if self.i == 0:
                    the_i = self.i # 0
                else:
                    the_i = self.i * iimag

                if self.j == 0:
                    the_j = self.j # 0
                else:
                    the_j = self.j * iimag

                if self.k == 0:
                    the_k = self.k # 0
                else:
                    the_k = self.k * iimag

                return Quaternion(rreal, the_i, the_j, the_k)
            else:
                return Quaternion(rreal, self.i * iimag, self.j * iimag, self.k * iimag)


    def makeQuaternionI(self, rreal, iimag):
        '''
        >>> print Quaternion(1.0, 2.0, 3.0, 4.0).makeQuaternionI(47.201, 23.5)
           47.2010  + 23.5000i   + 3.0000j   + 4.0000k
        
        '''
        return Quaternion(rreal, iimag, self.j, self.k)


    @staticmethod
    def branchCut(r, i, u):     # real, imaginary, quaternion
        absImagu = u.absImag()
        if (u.r != 0.0 or absImagu == 0.0 or isNaN(absImagu) or isInfinite(absImagu)):
            raise TypeError("final argument to branchCut() must be a finite nonzero imaginary, not %s" % u)
        iimag = i / absIimagu
        return Quaternion(r, u.i * iimag, u.j * iimag, u.k * iimag)



    def equals(self, *args):
        if len(args) == 2:      # object r, float eps
            if (eps < 0.0):
                raise ValueError(("equals() -- the epsilon value must be non-negative, not %f") % eps)

            r = args[0]
            if isinstance(r, Quaternion):
                if self.isZero():
                    if r.abs() < eps:
                        return True
                    else:
                        return False

                if r.isZero():
                    if self.abs() < eps:
                        return True
                    else:
                        return False

                # Two values, neither zero.

                if self.subtract(r).abs() / math.min(r.abs(), self.abs()) < eps:
                    return True
                else:
                    return False
            else:
                return False    # r not a Quaternion, unequal by definition

        elif len(args) == 1:
            r = args[0]
            if isinstance(r, Quaternion):
                if r.r == self.r and r.i == self.i and r.j == self.j and r.k == self.k:
                    return True
                else:
                    return False
            else:               # Not a Quaternion
                    return False

        else:
            raise ValueError("equals() only accepts 1 or two arguments -- %d supplied" % len(args))


    def hashCode(self):
        '''
        >>> print Quaternion(17.5, 81.31, -92.51, 1502.9).hashCode()
        1798172337
        >>> print Quaternion(0.0,0.0,0.0,0.0).hashCode()
        0
        '''
        if self.isZero():
            return 0

        if self.isInfinite() or self.isNaN():
            hashr = hash(self.r)
            hashi = hash(self.i)
            hashj = hash(self.j)
            hashk = hash(self.k)

            return hashr + 536870923 * hashi + 1073741827 * hashj + 1610612741 * hashk

        return hash(self.r + self.i * 0.342562356254152 + self.j * 0.832532153125423 + self.k * 0.234577589359823)

    def toString(self):
        "I really don't get what this is about, so I take the function name at face value"
        return self.__repr__()


    @staticmethod
    def imaginaryUnits(newunits):
        if (len(newunits) < 3):
            raise ValueError("The number of entries in the units array must be at least 3 but got %d" % len(newunits))

        oldUnits = quaternion.units
        quaternion.units = newunits
        return oldUnits


    @staticmethod
    def fromArray(arr):         # Turn array of 4+ into Quaternion, using first 4 entries
        '''
        >>> print Quaternion.fromArray([71.3, 82.31, -93.01, 15.49])
           71.3000  + 82.3100i  - 93.0100j  + 15.4900k
        '''
        if isIterable(arr):
            if len(arr) >= 4:
                return Quaternion(arr[0], arr[1], arr[2], arr[3])
        raise ValueError('fromArray() requires array of length 4')


    def toArray(self, *args):
        '''
        >>> print Quaternion(71.3, 82.31, -93.01, 1504.9).toArray()  # Here len(args) == 0
        [71.3, 82.31, -93.01, 1504.9]
        >>> print Quaternion(71.3, 82.31, -93.01, 1504.9).toArray([1,2])  # Here len(args) == 2
        [71.3, 82.31]
        '''
        if len(args) == 0:
            return [self.r, self.i, self.j, self.k]
        else:
            v = args[0]         # Vector length allows us to limit length of returned vector
            result = []
            for i in range(len(v)):
                result.append(self.coefficient(i))
            return result


    def toImagArray(self):
        '''
        >>> print Quaternion(17.5, 81.31, -92.51, 1502.9).toImagArray()
        [81.31, -92.51, 1502.9]
        '''
        return [self.i, self.j, self.k]


    def isZero(self):
        '''
        >>> print Quaternion(71.3, 82.31, -93.01, 1504.9).isZero()
        False
        >>> print Quaternion().isZero()
        True
        '''
        if self.r == 0.0 and self.i == 0.0 and self.j == 0.0 and self.k == 0.0:
            return True
        else:
            return False


    def isReal(self):
        '''
        >>> print Quaternion().isReal() # All zeroes
        True
        '''
        if self.i == 0.0 and self.j == 0.0 and self.k == 0.0:
            return True
        else:
            return False


    def isImaginary(self,*args):
        '''
        >>> print Quaternion().isImaginary() # All zeroes
        True
        '''
        if len(args) == 0:
            if self.r == 0.0:
                return True
            else:
                return False
        else:
            n = args[0]

            if i == 1: 
                if self.r == 0.0 and self.j == 0.0 and self.k == 0.0:
                    return True
                else:
                    return False
            
            if i == 2: 
                if self.r == 0.0 and self.i == 0.0 and self.k == 0.0: 
                    return True
                else:
                    return False
            
            if i == 3: 
                if self.r == 0.0 and self.i == 0.0 and self.j == 0.0:
                    return True
                else:
                    return False
            
            else:
                raise ValueError("isImaginary() -- the argument n must be 1, 2, or 3 not %d" % n)
    

    def isImaginaryI(self):
        if self.r == 0.0 and self.j == 0.0 and self.k == 0.0:
            return True
        else:
            return False


    def isImaginaryJ(self):
        if self.r == 0.0 and self.i == 0.0 and self.k == 0.0:
            return True
        else:
            return False


    def isImaginaryK(self):
        if self.r == 0.0 and self.i == 0.0 and self.j == 0.0:
            return True
        else:
            return False


    def isNaN(self):
        if not (isNaN(self.r) or isNaN(self.i) or isNaN(self.j) or isNaN(self.k)):
            return False
        else:
            return True


    def isInfinite(self):
        if not (isInfinite(self.r) or isInfinite(self.i) or isInfinite(self.j) or isInfinite(self.k)):
            return False
        else:
            return True


    def abs(self):
        return math.sqrt(self.abs2())


    def abs2(self):
        if self.isInfinite():
            return Infinity
        else:
            return self.r * self.r + self.i * self.i + self.j * self.j + self.k * self.k


    def absImag(self):
        return math.sqrt(self.abs2Imag())


    def abs2Imag(self):
        if (isInfinite(self.i) or isInfinite(self.j) or isInfinite(self.k)):
            return Infinity
        return self.i * self.i + self.j * self.j + self.k * self.k


    def real(self): 
        return self.r
    

    def imag(self): 
        return Quaternion(0.0, self.i, self.j, self.k)
    

    def imagI(self):
        return self.i
    

    def imagJ(self): 
        return self.j
    

    def imagK(self): 
        return self.k


    def coefficient(self,i):
        if i == 0:
            return self.r

        if i == 1:
            return self.i

        if i == 2:
            return self.j

        if i == 3:
            return self.k

        raise ValueError("coefficient() - the argument must be 0, 1, 2, or 3, not %d" % i)


    def project(self, *args):
        '''
        >>> print Quaternion(71.3, 82.31, -93.01, 1504.9).project([0,1,1,0]) # [] is selection vector
            0.0000  + 82.3100i  - 93.0100j   + 0.0000k
        >>> print Quaternion(71.3, 82.31, -93.01, 1504.9).project(2) # Select element 2 (j)
            0.0000   + 0.0000i  - 93.0100j   + 0.0000k
        '''
        if isIterable(args[0]): # Boolean selection vector. Should be len 4 but we allow smaller.
            v  = args[0]
            gr = gi = gj = gk = 0.0
            if (len(v) >= 1) and v[0]: gr = self.r
            if (len(v) >= 2) and v[1]: gi = self.i
            if (len(v) >= 3) and v[2]: gj = self.j
            if (len(v) >= 4) and v[3]: gk = self.k
            return Quaternion(gr, gi, gj, gk)

        n  = args[0]
        if isinstance(n,int) or isinstance(n,long):   # Otherwise, select a single coefficient
            if n == 0: 
                return Quaternion(self.r, 0.0, 0.0, 0.0)
            
            if n == 1: 
                return Quaternion(0.0, self.i, 0.0, 0.0)
            
            if n == 2: 
                return Quaternion(0.0, 0.0, self.j, 0.0)
            
            if n == 3: 
                return Quaternion(0.0, 0.0, 0.0, self.k)
            
        raise TypeError("project() needs integer 0-3 or boolvec of length >= 4, not %s" % (args,))


    def conjugate(self):
        '''
        >>> print Quaternion(71.3, 82.31, -93.01, 1504.9).conjugate() # Negate the imaginaries
           71.3000  - 82.3100i  + 93.0100j- 1504.9000k
        '''
        return Quaternion(self.r, - self.i, - self.j, - self.k)


    def argument(self):
        '''
        >>> print Quaternion(71.3, 82.31, -93.01, 1504.9).argument()
        1.52361334412
        '''
        return math.atan2(self.absImag(), self.r)


    def signum(self):
        '''
        >>> print Quaternion(71.3, 82.31, -93.01, 1504.9).signum()
            0.0472   + 0.0544i   - 0.0615j   + 0.9955k
        '''
        abs = self.abs2()
        if abs == 0.0:
            return self

        if isInfinite(abs) or isNaN(abs):
            if self.isNaN():
                return NaN
            
            if not(isInfinite(self.i) or isInfinite(self.j) or isInfinite(self.k)):
                return ONE

            if not(isInfinite(self.r) or isInfinite(self.j) or isInfinite(self.k)):
                return I

            if not(isInfinite(self.r) or isInfinite(self.i) or isInfinite(self.k)):
                return J

            if not(isInfinite(self.r) or isInfinite(self.i) or isInfinite(self.j)):
                return K

            return NaN

        abs = math.sqrt(abs)
        return Quaternion(self.r / abs, self.i / abs, self.j / abs, self.k / abs)


    # ---------- Arithmetic operations ---------- #


    def add(self, *args):
        '''
        >>> print Quaternion(71.3, 82.31, -93.01, 1504.9).add(Quaternion(2, 6.7, -4.4, -700.9))
           73.3000  + 89.0100i  - 97.4100j + 804.0000k
        >>> print Quaternion(71.3, 82.31, -93.01, 1504.9).add(-1.3)
           70.0000  + 82.3100i  - 93.0100j+ 1504.9000k
        >>> print Quaternion(71.3, 82.31, -93.01, 1504.9).add(1,0.69)
           71.3000  + 83.0000i  - 93.0100j+ 1504.9000k
        '''
        if len(args) == 1:
            r = args[0]
            if isinstance(r, Quaternion): # Q + Q
                return Quaternion(self.r + r.r, self.i + r.i, self.j + r.j, self.k + r.k)
            elif isinstance(r, float) or isinstance(r, int) or isinstance(r, long): # Q + float (updates .r field only)
                return Quaternion(self.r + r, self.i, self.j, self.k)
            else:
                raise TypeError("quaternion.add() with a single argument wants only Quaternion or float, not %s" % type(args[0]))
        if len(args) == 2:  # arg1 = int, selecting which member to update
            n = args[0]
            x = args[1]
            if n == 0:
                return Quaternion(self.r + x, self.i, self.j, self.k)

            if n == 1:
                return Quaternion(self.r, self.i + x, self.j, self.k)

            if n == 2:
                return Quaternion(self.r, self.i, self.j + x, self.k)

            if n == 3:
                return Quaternion(self.r, self.i, self.j, self.k + x)

            raise ValueError("quaternion.add() - first argument of two must be 0, 1, 2, or 3 but got %d" % n)

        raise ValueError("quaternion.add() accepts only 1 or 2 arguments, not %d" % len(args))

    def setR(self, x):
        return Quaternion(x, self.i, self.j, self.k)

    def addI(self, x):
        '''
        >>> print Quaternion(71.3, 82.31, -93.01, 1504.9).addI(6.39)
           71.3000  + 88.7000i  - 93.0100j+ 1504.9000k
        '''
        return Quaternion(self.r, self.i + x, self.j, self.k)

    def setI(self, x):
        return Quaternion(self.r, x, self.j, self.k)

    def addJ(self, x):
        '''
        >>> print Quaternion(71.3, 82.31, -93.01, 1504.9).addJ(93.02)
           71.3000  + 82.3100i   + 0.0100j+ 1504.9000k
        '''
        return Quaternion(self.r, self.i, self.j + x, self.k)

    def setJ(self,x):
        return Quaternion(self.r, self.i, x, self.k)

    def addK(self, x):
        '''
        >>> print Quaternion(71.3, 82.31, -93.01, 1504.9).addK(95.1)
           71.3000  + 82.3100i  - 93.0100j+ 1600.0000k
        '''
        return Quaternion(self.r, self.i, self.j, self.k + x)

    def setK(self,x):
        return Quaternion(self.r, self.i, self.j, x)

    def subtract(self, r):      # self - r
        '''
        >>> print Quaternion(73.3000, 89.0100,- 97.4100, 804.0000).subtract(Quaternion(2, 6.7, -4.4, -700.9))
           71.3000  + 82.3100i  - 93.0100j+ 1504.9000k
        >>> print Quaternion(70., 82.31, -93.01, 1504.9).subtract(Quaternion(-1.3,0,0,0))
           71.3000  + 82.3100i  - 93.0100j+ 1504.9000k
        '''
        return Quaternion(self.r - r.r, self.i - r.i, self.j - r.j, self.k - r.k)


    def subtractR(self, r):     # Reverse-subtract, r - self
        '''
        >>> print Quaternion(73.3000, 89.0100,- 97.4100, 804.0000).subtractR(Quaternion(2, 6.7, -4.4, -700.9))
         - 71.3000  - 82.3100i  + 93.0100j- 1504.9000k
        >>> print Quaternion(70., 82.31, -93.01, 1504.9).subtractR(Quaternion(-1.3,0,0,0))
         - 71.3000  - 82.3100i  + 93.0100j- 1504.9000k
        '''
        return Quaternion(r.r - self.r, r.i - self.i, r.j - self.j, r.k - self.k)


    def negate(self):
        '''
        >>> print Quaternion(73.3000, 89.0100,- 97.4100, 804.0000).negate()
         - 73.3000  - 89.0100i  + 97.4100j - 804.0000k
        '''
        return Quaternion(- self.r, - self.i, - self.j, - self.k)


    def multiply(self, r):
        '''
        >>> print Quaternion(73.3000, 89.0100,- 97.4100, 804.0000).multiply(Quaternion(2, 6.7, -4.4, -700.9))
        562645.2290- 71143.1390i- 68291.2490j- 50028.9730k
        >>> print Quaternion(73.3000, 89.0100,- 97.4100, 804.0000).multiply(10)
          733.0000 + 890.1000i - 974.1000j+ 8040.0000k
        '''
        # java code looks wrong. See http://www.mathworks.com/help/aeroblks/quaternionmultiplication.html
        if isinstance(r,Quaternion):
            return Quaternion(\
self.r * r.r - self.i * r.i - self.j * r.j - self.k * r.k, \
self.r * r.i + self.i * r.r - self.j * r.k + self.k * r.j, \
self.r * r.j + self.i * r.k + self.j * r.r - self.k * r.i, \
self.r * r.k - self.i * r.j + self.j * r.i + self.k * r.r)

        if isinstance(r, float) or isinstance(r, int) or isinstance(r, long):
            if isInfinite(r) or isNaN(r):
                if self.r == 0.0 and self.i == 0.0 and self.j == 0.0 and self.k == 0.0:
                    return NaN

                if self.r == 0:
                    fr = quaternion.sign(r)
                else:
                    fr = r

                if self.i == 0:
                    fi = quaternion.sign(r)
                else:
                    fi = r

                if self.j == 0:
                    fj = quaternion.sign(r)
                else:
                    fj = r

                if self.k == 0:
                    fk = quaternion.sign(r)
                else:
                    fk = r

                return Quaternion(self.r * fr, self.i * fi, self.j * fj, self.k * fk)

            return Quaternion(self.r * r, self.i * r, self.j * r, self.k * r)


    def divide(self, r):
        '''
        >>> print Quaternion(73.3000, 89.0100,- 97.4100, 804.0000).multiply(Quaternion(2, 6.7, -4.4, -700.9))
        562645.2290- 71143.1390i- 68291.2490j- 50028.9730k
        >>> print Quaternion(2, 6.7, -4.4, -700.9).divide(Quaternion(562645.2290, -71143.1390, - 68291.2490,- 50028.9730))  # See multiplication above for the inverse
           73.3000  + 89.0100i  - 97.4100j + 804.0000k
        '''
        # java code looks wrong. See http://www.mathworks.com/help/aeroblks/quaterniondivision.html
        abs2 = self.abs2()
        if abs2 == 0: return self # Don't want to throw exception
        return Quaternion( \
   (self.r  * r.r + self.i * r.i + self.j * r.j + self.k * r.k) / abs2,\
 (( self.r) * r.i - self.i * r.r - self.j * r.k + self.k * r.j) / abs2,\
 (( self.r) * r.j + self.i * r.k - self.j * r.r - self.k * r.i) / abs2,\
 (( self.r) * r.k - self.i * r.j + self.j * r.i - self.k * r.r) / abs2)

    divide_into = divide

    def divide_by(self,r):
        '''
        >>> print Quaternion(562645.2290, -71143.1390, - 68291.2490,- 50028.9730).divide_by(Quaternion(2, 6.7, -4.4, -700.9))  # See multiplication above for the inverse
           73.3000  + 89.0100i  - 97.4100j + 804.0000k
        '''
        return r.divide_into(self)

    def pow(self, r):
        '''
        >>> q = Quaternion(5.1012, -1.2721, -43.06, 11.130)
        >>> print q.multiply(q)
        -1953.6365  - 12.9785i - 439.3153j + 113.5527k
        >>> print q.pow(2)
        -1953.6365  - 12.9785i - 439.3153j + 113.5527k
        >>> print q.pow(2.0)
        -1953.6365  - 12.9785i - 439.3153j + 113.5527k
        '''
        if isinstance(r,Quaternion):
            return self.log().multiply(r).exp()
        elif isinstance(r, float):
            absImag = self.absImag()
            result  = complex.Complex(self.r, absImag).pow(r)
            if (absImag == 0.0):
                return self.makeQuaternion(result.r);
            return self.makeQuaternion(result.r, result.i / absImag)
        elif isinstance(r, int) or isinstance(r, long):
            absImag = self.absImag()
            result  = complex.Complex(self.r, absImag).pow(r)
            if (absImag == 0.0):
                return self.makeQuaternion(result.r)
            return self.makeQuaternion(result.r, result.i / absImag)


    @canuse
    def sqr(self):
        '''
         >>> print Quaternion(5.1012, -1.2721, -43.06, 11.130).sqr()
         -1953.6365  - 12.9785i - 439.3153j + 113.5527k
        '''
        return self.makeQuaternion(self.r * self.r - self.i * self.i - self.j * self.j - self.k * self.k, 2.0 * self.r)

    def horner(self,v):              # v is vector of coefficients
        '''
        >>> print Quaternion(5.1012, -1.2721, -43.06, 11.130).horner([])
            0.0000   + 0.0000i   + 0.0000j   + 0.0000k
        >>> print Quaternion(5.1012, -1.2721, -43.06, 11.130).horner([123.456]) # 123.456 (no X)
          123.4560   + 0.0000i   + 0.0000j   + 0.0000k
        >>> print Quaternion(5.1012, -1.2721, -43.06, 11.130).horner([1.0,2.0]) # 2X+1.
           11.2024   - 2.5442i  - 86.1200j  + 22.2600k
        >>> print Quaternion(5.1012, -1.2721, -43.06, 11.130).horner([1,2]) # 2X+1.
           11.2024   - 2.5442i  - 86.1200j  + 22.2600k
        >>> print Quaternion(5.1012, -1.2721, -43.06, 11.130).horner([2.0,0,1]) # X^2+2.
        -1951.6365  - 12.9785i - 439.3153j + 113.5527k
        '''
        if len(v) == 0:
            return Quaternion(0.0, 0.0, 0.0, 0.0)

        if len(v) == 1:
            return Quaternion(v[0])

        absImag = self.absImag()
        result  = complex.Complex(self.r, absImag).horner(v)
        if (absImag == 0.0):
            return self.makeQuaternion(result.r)

        return self.makeQuaternion(result.r, result.i / absImag)


    @canuse
    def matmul(self, matrix):
        '''
        # Matrix multiply
        '''

        return Quaternion(self.r * matrix[ 0] + self.i * matrix[ 4] + self.j * matrix[ 8] + self.k * matrix[12],
                          self.r * matrix[ 1] + self.i * matrix[ 5] + self.j * matrix[ 9] + self.k * matrix[14],
                          self.r * matrix[ 2] + self.i * matrix[ 6] + self.j * matrix[10] + self.k * matrix[14],
                          self.r * matrix[ 3] + self.i * matrix[ 7] + self.j * matrix[11] + self.k * matrix[15] )



    @canuse
    def sqrt(self, *args):
        '''
        >>> qs = Quaternion(5.1012, -1.2721, -43.06, 11.130).sqr()
        >>> print qs.sqrt()
            5.1012   - 1.2721i  - 43.0600j  + 11.1300k
        '''
        if len(args) == 0:
            absx    = self.abs()
            absImag = self.absImag()

            if absx == 0.0:
                return self

            if absImag == 0.0:
                if self.r > 0.0:
                    return Quaternion(math.sqrt(self.r), 0.0, 0.0, 0.0)
                else:
                    return Quaternion(0.0, math.sqrt(- self.r), 0.0, 0.0)

            imag = math.sqrt(0.5 * (absx - self.r)) / absImag
            return Quaternion(math.sqrt(0.5 * (absx + self.r)), self.i * imag, self.j * imag, self.k * imag)

        elif len(args) == 1:
            u = args[0]
            absImag = self.absImag()
            if absImag != 0.0 or self.r >= 0.0:
                return self.sqrt()
            z = complex.Complex(self.r, absImag).sqrt()
            return quaternion.branchCut(z.real(), z.imagI(), u)


    def rotateAround(self, r):
        if self.r != 0.0:
            raise ValueError("rotateAround() -- cannot rotate a non-imaginary quaternion.")

        absr2 = r.abs2Imag()
        if absr2 == 0.0:
            raise ValueError("rotateAround() -- cannot rotate around a quaternion with a zero imaginary part.")

        rr2 = r.r * r.r
        ri2 = r.i * r.i
        rj2 = r.j * r.j
        rk2 = r.k * r.k
        rri = r.r * r.i
        rrj = r.r * r.j
        rrk = r.r * r.k
        rij = r.i * r.j
        rik = r.i * r.k
        rjk = r.j * r.k

        absr2den = absr2 + rr2

        return Quaternion(0.0, (2.0 * (self.j * (rij - rrk) + self.k * (rik + rrj)) + self.i * (rr2 + ri2 - rj2 - rk2)) / absr2, (2.0 * (self.i * (rij + rrk) + self.k * (rjk - rri)) + self.j * (rr2 - ri2 + rj2 - rk2)) / absr2, (2.0 * (self.i * (rik - rrj) + self.j * (rjk + rri)) + self.k * (rr2 - ri2 - rj2 + rk2)) / absr2den)


# ------------ Transcendental functions ------------- #
# ---- z<name> catches exceptions and returns 0 ----- #


    @canuse
    def zexp(self, *args):
        'Like exp(), but returns 0 for exceptions'
        try:
            rv = self.exp()
        except (ValueError,ZeroDivisionError,OverflowError) as err:
            rv = ZERO
        return rv

    def exp(self):
        '''
        >>> print Quaternion(5.1012, -1.2721, -43.06, 11.130).exp()
          143.2361   - 2.2964i  - 77.7321j  + 20.0919k
        >>> print Quaternion(5.1012, -1.2721, -43.06, 11.130).log().exp()
            5.1012   - 1.2721i  - 43.0600j  + 11.1300k
        '''
        absImag = self.absImag()
        if absImag == 0.0:
            return self.makeQuaternion(math.exp(self.r));

        expr = math.exp(self.r)
        return self.makeQuaternion(expr * math.cos(absImag), expr * (math.sin(absImag) / absImag))


    @canuse
    @catch_None
    def zlog(self, *args):
        'Like log(), but returns 0 for argument 0'
        if self.isZero():
            return self
        else:
            return self.log(*args)

    def log(self, *args):
        'Unit tests are with exp() above'
        absImag = self.absImag()
        if len(args) == 0:
            if absImag == 0.0:
                if self.r > 0.0:
                    return self.makeQuaternion(math.log(self.r))
                elif self.r == 0.0:
                    return self.makeQuaternion()

                return self.makeQuaternionI(math.log(- self.r), 3.141592653589793)

            return self.makeQuaternion(math.log(self.abs()), math.atan2(absImag, self.r) / absImag)

        if len(args) >= 1:
            u = args[0]

            if absImag != 0.0 or self.r > 0.0:
                return self.log()

            z = complex.Complex(self.r, absImag).log()
            return Quaternion.branchCut(z.real(), z.imagI(), u)


    def log10(self, *args):
        absImag = self.absImag()

        if len(args) == 0:
            if absImag == 0.0:
               if self.r >= 0.0:
                   return self.makeQuaternion(math.log10(self.r))
               else:
                   return self.makeQuaternionI(math.log10(- self.r), 3.141592653589793)
            return self.makeQuaternion(math.log10(self.abs()), math.atan2(absImag, self.r) / absImag / math.log(10.0))
        if len(args) >= 1:
            if absImag != 0.0 or self.r > 0.0:
                return self.log10()
            z = complex.Complex(self.r, absImag).log10()
            return quaternion.branchCut(z.real(), z.imagI(), u)
 

    # ---------- 6 trig functions ---------- #


    @canuse
    def zsin(self):
        try:
            rv = self.sin()
        except (ValueError,ZeroDivisionError,OverflowError) as err:
            rv = ZERO
        return rv

    def sin(self):
        '''
        >>> print Quaternion(0.1012, -0.2721, -0.06, 0.130).sin()
            0.1058   - 0.2750i   - 0.0606j   + 0.1314k
        '''
        absImag = self.absImag()
        if absImag == 0.0:
            return self.makeQuaternion(math.sin(self.r))
        else:
            return self.makeQuaternion(math.sin(self.r) * math.cosh(absImag),\
                                       math.cos(self.r) * math.sinh(absImag) / absImag)

    @canuse
    def zcos(self):
        try:
            rv = self.cos()
        except (ValueError,ZeroDivisionError,OverflowError) as err:
            rv = ZERO
        return rv

    def cos(self):
        '''
        >>> q = Quaternion(0.1012, -0.2721, -0.06, 0.130)
        >>> print q.cos()
            1.0423   + 0.0279i   + 0.0062j   - 0.0133k
        >>> print q.sin().sqr().add(q.cos().sqr()) # Must fudge -sign in coefficient of i, sorry
            1.0000   - 0.0000i   + 0.0000j   + 0.0000k
        '''
        absImag = self.absImag()
        if absImag == 0.0:
            return self.makeQuaternion(math.cos(self.r))
        else:
            return self.makeQuaternion(math.cos(self.r) * math.cosh(absImag),\
                                  (- math.sin(self.r)) * (math.sinh(absImag) / absImag))


    @canuse
    def ztan(self):
        '''
        >>> q = Quaternion(0.1012, -0.2721, -0.06, 0.130)
        >>> print q.ztan()
            0.0924   - 0.2663i   - 0.0587j   + 0.1272k
        >>> print q.sin().divide_by(q.cos())
            0.0924   - 0.2663i   - 0.0587j   + 0.1272k
        '''
        try:
            rv = self.tan()
        except (ValueError,ZeroDivisionError,OverflowError) as err:
            rv = ZERO
        return rv

    def tan(self):
        '''
        >>> q = Quaternion(0.1012, -0.2721, -0.06, 0.130)
        >>> print q.tan()
            0.0924   - 0.2663i   - 0.0587j   + 0.1272k
        >>> print q.sin().divide_by(q.cos())
            0.0924   - 0.2663i   - 0.0587j   + 0.1272k
        '''
        absImag = self.absImag()
        if absImag == 0.0:
            return self.makeQuaternion(math.tan(self.r))
        else:
            denom = sqr(math.cos(self.r)) + sqr(math.sinh(absImag))
            return self.makeQuaternion(math.cos(self.r) * math.sin(self.r) / denom,\
                                       math.cosh(absImag) * (math.sinh(absImag) / absImag) / denom)


    @canuse
    def zsec(self):
        '''
        >>> q = Quaternion(0.1012, -0.2721, -0.06, 0.130)
        >>> print q.zsec()
            0.9586   - 0.0257i   - 0.0057j   + 0.0123k
        >>> print q.cos()
            1.0423   + 0.0279i   + 0.0062j   - 0.0133k
        >>> print q.cos().multiply(q.zsec())
            1.0000   - 0.0000i   + 0.0000j   + 0.0000k
        '''
        try:
            rv = self.sec()
        except (ValueError,ZeroDivisionError,OverflowError) as err:
            rv = ZERO
        return rv

    def sec(self):
        '''
        >>> q = Quaternion(0.1012, -0.2721, -0.06, 0.130)
        >>> print q.sec()
            0.9586   - 0.0257i   - 0.0057j   + 0.0123k
        >>> print q.cos()
            1.0423   + 0.0279i   + 0.0062j   - 0.0133k
        >>> print q.cos().multiply(q.sec())
            1.0000   - 0.0000i   + 0.0000j   + 0.0000k
        '''
        absImag = self.absImag()
        if absImag == 0.0:
            return self.makeQuaternion(1.0 / math.cos(self.r))
        else:
            denom = sqr(math.cos(self.r)) + sqr(math.sinh(absImag))
            return self.makeQuaternion(math.cos(self.r) * math.cosh(absImag) / denom,\
                                       math.sin(self.r) * (math.sinh(absImag) / absImag) / denom)

    @canuse
    def zcsc(self):
        '''
        >>> q = Quaternion(0.1012, -0.2721, -0.06, 0.130)
        >>> print q.zcsc()
            0.9822   + 2.5519i   + 0.5627j   - 1.2192k
        >>> print q.sin()
            0.1058   - 0.2750i   - 0.0606j   + 0.1314k
        >>> print q.zcsc().multiply(q.sin())
            1.0000   - 0.0000i   + 0.0000j   + 0.0000k
        '''
        try:
            rv = self.csc()
        except (ValueError,ZeroDivisionError,OverflowError) as err:
            rv = ZERO
        return rv


    def csc(self):
        '''
        >>> q = Quaternion(0.1012, -0.2721, -0.06, 0.130)
        >>> print q.csc()
            0.9822   + 2.5519i   + 0.5627j   - 1.2192k
        >>> print q.sin()
            0.1058   - 0.2750i   - 0.0606j   + 0.1314k
        >>> print q.csc().multiply(q.sin())
            1.0000   - 0.0000i   + 0.0000j   + 0.0000k
        '''
        absImag = self.absImag()
        if absImag == 0.0:
            return self.makeQuaternion(1.0 / math.sin(self.r))
        else:
            denom = sqr(math.sin(self.r)) + sqr(math.sinh(absImag))
            return self.makeQuaternion(math.sin(self.r) * math.cosh(absImag) / denom,
                                   (- math.cos(self.r)) * (math.sinh(absImag) / absImag) / denom)


    @canuse
    def zcot(self):
        try:
            rv = self.cot()
        except (ValueError,ZeroDivisionError,OverflowError) as err:
            rv = ZERO
        return rv

    def cot(self):
        '''
        >>> q = Quaternion(0.1012, -0.2721, -0.06, 0.130)
        >>> print q.cot()
            0.9327   + 2.6872i   + 0.5925j   - 1.2838k
        >>> print q.tan()
            0.0924   - 0.2663i   - 0.0587j   + 0.1272k
        >>> print q.tan().multiply(q.cot()) # tan*cot = 1
            1.0000   - 0.0000i   + 0.0000j   + 0.0000k
        >>> print q.sin().divide_into(q.cos()) # cos/sin = cot
            0.9327   + 2.6872i   + 0.5925j   - 1.2838k
        '''
        absImag = self.absImag()
        if absImag == 0.0:
            return self.makeQuaternion(1.0 / math.tan(self.r))
        else:
            denom = sqr(math.sin(self.r)) + sqr(math.sinh(absImag))
            return self.makeQuaternion(math.sin(self.r) * math.cos(self.r) / denom,
                                    (- math.cosh(absImag)) * (math.sinh(absImag) / absImag) / denom)


    # ---------- 6 hyperbolic trig functions ---------- #


    @canuse
    def zsinh(self):
        try:
            rv = self.sinh()
        except (ValueError,ZeroDivisionError,OverflowError) as err:
            rv = ZERO
        return rv


    def sinh(self):
        '''
        >>> print Quaternion().sinh() # sinh(0) = 0
            0.0000   + 0.0000i   + 0.0000j   + 0.0000k
        >>> q = Quaternion(0.1012, -0.2721, -0.06, 0.130)
        >>> print q.sinh()
            0.0966   - 0.2692i   - 0.0594j   + 0.1286k
        '''
        absImag = self.absImag()
        if absImag == 0.0:
            return self.makeQuaternion(math.sinh(self.r))
        else:
            return self.makeQuaternion(math.sinh(self.r) * math.cos(absImag), math.cosh(self.r) * (math.sin(absImag) / absImag))


    @canuse
    def zcosh(self):
        try:
            rv = self.cosh()
        except (ValueError,ZeroDivisionError,OverflowError) as err:
            rv = ZERO
        return rv

    def cosh(self):
        '''
        >>> print Quaternion().cosh() # coshh(0) = 1.0
            1.0000   + 0.0000i   + 0.0000j   + 0.0000k
        >>> q = Quaternion(0.1012, -0.2721, -0.06, 0.130)
        >>> print q.cosh()
            0.9580   - 0.0272i   - 0.0060j   + 0.0130k
        '''
        absImag = self.absImag()
        if absImag == 0.0:
            return self.makeQuaternion(math.cosh(self.r))
        else:
            return self.makeQuaternion(math.cosh(self.r) * math.cos(absImag), math.sinh(self.r) * math.sin(absImag) / absImag)


    @canuse
    def ztanh(self):
        '''
        >>> print Quaternion().ztanh() # ztanh(0) = 0
            0.0000   + 0.0000i   + 0.0000j   + 0.0000k
        >>> q = Quaternion(0.1012, -0.2721, -0.06, 0.130)
        >>> print q.ztanh()
            0.1109   - 0.2779i   - 0.0613j   + 0.1328k
        >>> print q.sinh().divide_by(q.cosh())
            0.1109   - 0.2779i   - 0.0613j   + 0.1328k
        '''
        try:
            rv = self.tanh()
        except (ValueError,ZeroDivisionError,OverflowError) as err:
            rv = ZERO
        return rv


    def tanh(self):
        '''
        >>> print Quaternion().tanh() # tanh(0) = 0
            0.0000   + 0.0000i   + 0.0000j   + 0.0000k
        >>> q = Quaternion(0.1012, -0.2721, -0.06, 0.130)
        >>> print q.tanh()
            0.1109   - 0.2779i   - 0.0613j   + 0.1328k
        >>> print q.sinh().divide_by(q.cosh())
            0.1109   - 0.2779i   - 0.0613j   + 0.1328k
        '''

        absImag = self.absImag()
        if absImag == 0.0:
            return self.makeQuaternion(math.tanh(self.r))
        else:
            denom = sqr(math.sinh(self.r)) + sqr(math.cos(absImag))
            return self.makeQuaternion(math.cosh(self.r) * math.sinh(self.r) / denom, math.cos(absImag) * (math.sin(absImag) / absImag) / denom)


    @canuse
    def zsech(self):
        try:
            rv = self.sech()
        except (ValueError,ZeroDivisionError,OverflowError) as err:
            rv = ZERO
        return rv


    def sech(self):
        '''
        >>> print Quaternion().sech() # tanh(0) = 0
            1.0000   + 0.0000i   + 0.0000j   + 0.0000k
        >>> q = Quaternion(0.1012, -0.2721, -0.06, 0.130)
        >>> print q.sech()
            1.0428   + 0.0296i   + 0.0065j   - 0.0141k
        >>> print q.cosh().divide_into(Quaternion(1.0,0.0,0.0,0.0))
            1.0428   + 0.0296i   + 0.0065j   - 0.0141k
        '''
        absImag = self.absImag()
        if absImag == 0.0:
            return self.makeQuaternion(1.0 / math.cosh(self.r))
        else:
            denom = sqr(math.sinh(self.r)) + sqr(math.cos(absImag))
            return self.makeQuaternion(math.cosh(self.r) * math.cos(absImag) / denom, (- math.sinh(self.r)) * (math.sin(absImag) / absImag) / denom)

    @canuse
    def zcsch(self):
        try:
            rv = self.csch()
        except (ValueError,ZeroDivisionError,OverflowError) as err:
            rv = ZERO
        return rv

    def csch(self):
        '''
        >>> print Quaternion().csch() # csch(0) = 0
               inf   + 0.0000i   + 0.0000j   + 0.0000k
        >>> q = Quaternion(0.1012, -0.2721, -0.06, 0.130)
        >>> print q.csch()
            0.9484   + 2.6426i   + 0.5827j   - 1.2625k
        >>> print q.sinh().divide_into(Quaternion(1.0,0.0,0.0,0.0))
            0.9484   + 2.6426i   + 0.5827j   - 1.2625k
        '''
        absImag = self.absImag()
        if absImag == 0.0:
            if self.r == 0:     # csch(0) is undefined
                return Quaternion(Infinity, 0.0, 0.0, 0.0)
            return self.makeQuaternion(1.0 / math.sinh(self.r))
        else:
            denom = sqr(math.sinh(self.r)) + sqr(math.sin(absImag))
            return self.makeQuaternion(math.sinh(self.r) * math.cos(absImag) / denom, (- math.cosh(self.r)) * (math.sin(absImag) / absImag) / denom)


    @canuse
    def zcoth(self):
        try:
            rv = self.coth()
        except (ValueError,ZeroDivisionError,OverflowError) as err:
            rv = ZERO
        return rv

    def coth(self):
        '''
        >>> print Quaternion().coth() # coth(0) undefined
               inf   + 0.0000i   + 0.0000j   + 0.0000k
        >>> q = Quaternion(0.1012, -0.2721, -0.06, 0.130)
        >>> print q.coth() # Reciprocal of tanh(), as we will see below
            1.0002   - 2.5058i   - 0.5525j   + 1.1972k
        >>> print q.tanh().divide_into(Quaternion(1.0,0.0,0.0,0.0))
            1.0002   + 2.5058i   + 0.5525j   - 1.1972k
        '''
        absImag = self.absImag()
        if absImag == 0.0:      # All imaginaries == 0?
            if self.r == 0:     # coth(0) is undefined
                return Quaternion(Infinity, 0.0, 0.0, 0.0)
            else:
                return self.makeQuaternion(1.0 / math.tanh(self.r))
        else:
            denom = sqr(math.sinh(self.r)) + sqr(math.sin(absImag))
            return self.makeQuaternion(math.sinh(self.r) * math.cosh(self.r) / denom, math.cos(absImag) * (math.sin(absImag) / absImag) / denom)


    # ---------- 6 arc trig functions ---------- #


    @canuse
    def zasin(self):
        try:
            rv = self.asin()
        except (ValueError,ZeroDivisionError,OverflowError) as err:
            rv = ZERO
        return rv

    def asin(self, *args):
        '''
        >>> q = Quaternion(0.1012, -0.2721, -0.06, 0.130)
        >>> print q.asin()
            0.0968   - 0.2692i   - 0.0594j   + 0.1286k
        >>> print q.asin().sin()
            0.1012   - 0.2721i   - 0.0600j   + 0.1300k
        '''
        absImag = self.absImag()
        if len(args) == 0:
            z = complex.Complex(self.r, absImag).asin()
            if absImag == 0.0:
                return self.makeQuaternion(z)
            else:
                return self.makeQuaternion(z.real(), z.imagI() / absImag)

        elif len(args) == 1:
            u = args[0]
            if absImag != 0.0 or self.r >= -1.0 and self.r <= 1.0:
                return self.asin()
            else:
                z = complex.Complex(self.r, absImag).asin()
                return quaternion.branchCut(z.real(), z.imagI(), u)


    @canuse
    def zacos(self):
        try:
            rv = self.acos()
        except (ValueError,ZeroDivisionError,OverflowError) as err:
            rv = ZERO
        return rv

    def acos(self, *args):
        '''
        >>> q = Quaternion(0.1012, -0.2721, -0.06, 0.130)
        >>> print q.acos()
            1.4740   + 0.2692i   + 0.0594j   - 0.1286k
        >>> print q.acos().cos()
            0.1012   - 0.2721i   - 0.0600j   + 0.1300k
        '''
        absImag = self.absImag()
        if len(args) == 0:
            z = complex.Complex(self.r, absImag).acos()
            if absImag == 0.0:
                return self.makeQuaternion(z)
            else:
                return self.makeQuaternion(z.real(), z.imagI() / absImag)

        if len(args) == 1:
            u = args[0]
            if absImag != 0.0 or self.r >= -1.0 and self.r <= 1.0:
                return self.acos()
            else:
                z = complex.Complex(self.r, absImag).acos()
                return quaternion.branchCut(z.real(), z.imagI(), u)


    @canuse
    def zatan(self):
        '''
        >>> q = Quaternion(0.1012, -0.2721, -0.06, 0.130)
        >>> print q.zatan()
            0.1112   - 0.2778i   - 0.0613j   + 0.1327k
        >>> print q.zatan().tan()
            0.1012   - 0.2721i   - 0.0600j   + 0.1300k
        '''
        try:
            rv = self.atan()
        except (ValueError,ZeroDivisionError,OverflowError) as err:
            rv = ZERO
        return rv

    def atan(self, *args):
        '''
        >>> q = Quaternion(0.1012, -0.2721, -0.06, 0.130)
        >>> print q.atan()
            0.1112   - 0.2778i   - 0.0613j   + 0.1327k
        >>> print q.atan().tan()
            0.1012   - 0.2721i   - 0.0600j   + 0.1300k
        '''
        absImag = self.absImag()

        if len(args) == 0:
            z = complex.Complex(self.r, absImag).atan()
            if absImag == 0.0:
                return self.makeQuaternion(z)
            else:
                return self.makeQuaternion(z.real(), z.imagI() / absImag)


        if len(args) == 1:
            u = args[0]
            if self.r != 0.0 or absImag < 1.0:
                return self.atan()
            else:
                z = complex.Complex(self.r, absImag).atan()
                return quaternion.branchCut(z.real(), z.imagI(), u)


    @canuse
    def zasec(self):
        try:
            rv = self.asec()
        except (ValueError,ZeroDivisionError,OverflowError) as err:
            rv = ZERO
        return rv

    def asec(self, *args):
        '''
        >>> q = Quaternion(0.1012, -0.2721, -0.06, 0.130)
        >>> print q.asec()
            1.2675   - 1.6300i   - 0.3594j   + 0.7787k
        >>> print q.asec().sec()
            0.1012   - 0.2721i   - 0.0600j   + 0.1300k
        '''

        absImag = self.absImag()
        if len(args) == 0:
            z = complex.Complex(self.r, absImag).asec()
            if absImag == 0.0:
                return self.makeQuaternion(z)
            else:
                return self.makeQuaternion(z.real(), z.imagI() / absImag)

        if len(args) == 1:
            u = args[0]

            if self.r != 0.0 or absImag >= 1.0:
                return self.asec()
            else:
                z = complex.Complex(self.r, absImag).asec()
                return quaternion.branchCut(z.real(), z.imagI(), u)


    @canuse
    def zacsc(self):
        try:
            rv = self.acsc()
        except (ValueError,ZeroDivisionError,OverflowError) as err:
            rv = ZERO
        return rv


    def acsc(self,*args):
        '''
        >>> q = Quaternion(0.1012, -0.2721, -0.06, 0.130)
        >>> print q.asec()
            1.2675   - 1.6300i   - 0.3594j   + 0.7787k
        >>> print q.asec().sec()
            0.1012   - 0.2721i   - 0.0600j   + 0.1300k
        '''

        absImag = self.absImag()

        if len(args) == 0:
            z = complex.Complex(self.r, absImag).acsc()
            if absImag == 0.0:
                return self.makeQuaternion(z)
            else:
                return self.makeQuaternion(z.real(), z.imagI() / absImag)


        if len(args) >= 1:
            if self.r != 0.0 or absImag >= 1.0:
                return self.acsc()
            else:
                z = complex.Complex(self.r, absImag).acsc()
                return quaternion.branchCut(z.real(), z.imagI(), u)


    @canuse
    def zacot(self):
        try:
            rv = self.acot()
        except (ValueError,ZeroDivisionError,OverflowError) as err:
            rv = ZERO
        return rv

    def acot(self, *args):
        '''
        >>> q = Quaternion(0.1012, -0.2721, -0.06, 0.130)
        >>> print q.acot()
            1.4596   + 0.2778i   + 0.0613j   - 0.1327k
        >>> print q.acot().cot()
            0.1012   - 0.2721i   - 0.0600j   + 0.1300k
        '''
        absImag = self.absImag()

        if len(args) == 0:
            z = complex.Complex(self.r, absImag).acot()
            if absImag == 0.0:
                return self.makeQuaternion(z)
            else:
                return self.makeQuaternion(z.real(), z.imagI() / absImag)

        if len(args) >= 1:
            if self.r != 0.0 or absImag >= 1.0:
                return self.acot()
            else:
                z = complex.Complex(self.r, absImag).acot()
                return quaternion.branchCut(z.real(), z.imagI(), u)


    # ---------- 6 arc hyper trig functions ---------- #


    @canuse
    def zasinh(self):
        try:
            rv = self.asinh()
        except (ValueError,ZeroDivisionError,OverflowError) as err:
            rv = ZERO
        return rv

    def asinh(self, *args):
        '''
        >>> q = Quaternion(0.1012, -0.2721, -0.06, 0.130)
        >>> print q.asinh()
            0.1061   - 0.2750i   - 0.0606j   + 0.1314k
        >>> print q.asinh().sinh()
            0.1012   - 0.2721i   - 0.0600j   + 0.1300k
        '''
        absImag = self.absImag()

        if len(args) == 0:
            z = complex.Complex(self.r, absImag).asinh()
            if absImag == 0.0:
                return self.makeQuaternion(z)
            else:
                return self.makeQuaternion(z.real(), z.imagI() / absImag)

        if len(args) >= 1:
            if self.r != 0.0 or absImag <= 1.0:
                return self.asinh()

            z = complex.Complex(self.r, absImag).asinh()
            return quaternion.branchCut(z.real(), z.imagI(), u)


    @canuse
    def zacosh(self):
        '''
        >>> q = Quaternion(0.1012, -0.2721, -0.0600, 0.1300)
        >>> print q.zacosh()
            0.3042   - 1.3044i   - 0.2876j   + 0.6232k
        >>> print q.zacosh().cosh()
            0.1012   - 0.2721i   - 0.0600j   + 0.1300k
        '''
        try:
            rv = self.acosh()
        except (ValueError,ZeroDivisionError,OverflowError) as err:
            rv = ZERO
        return rv

    def acosh(self, *args):
        '''
        >>> q = Quaternion(0.1012, -0.2721, -0.0600, 0.1300)
        >>> print q.acosh()
            0.3042   - 1.3044i   - 0.2876j   + 0.6232k
        >>> print q.acosh().cosh()
            0.1012   - 0.2721i   - 0.0600j   + 0.1300k
        '''
 
        absImag = self.absImag()

        if len(args) == 0:
            z = complex.Complex(self.r, absImag).acosh()
            if absImag == 0.0:
                return self.makeQuaternion(z)
            else:
                return self.makeQuaternion(z.real(), z.imagI() / absImag)

        if len(args) >= 1:
            if absImag != 0.0 or self.r >= 1.0:
                return self.acosh()

            z = complex.Complex(self.r, absImag).acosh()
            return quaternion.branchCut(z.real(), z.imagI(), u)


    @canuse
    def zatanh(self, *args):
        try:
            rv = self.atanh()
        except (ValueError,ZeroDivisionError,OverflowError) as err:
            rv = ZERO
        return rv

    def atanh(self, *args):
        '''
        >>> q = Quaternion(0.1012, -0.2721, -0.06, 0.130)
        >>> print q.atanh()
            0.0926   - 0.2663i   - 0.0587j   + 0.1272k
        >>> print q.atanh().tanh()
            0.1012   - 0.2721i   - 0.0600j   + 0.1300k
        '''
        absImag = self.absImag()

        if len(args) == 0:
            z = complex.Complex(self.r, absImag).atanh()
            if absImag == 0.0:
                return self.makeQuaternion(z)
            else:
                return self.makeQuaternion(z.real(), z.imagI() / absImag)

        if len(args) >= 1:
            if absImag != 0.0 or self.r > -1.0 and self.r < 1.0:
                return self.atanh()

            z = complex.Complex(self.r, absImag).atanh()
            return quaternion.branchCut(z.real(), z.imagI(), u)


    @canuse
    def zasech(self):
        try:
            rv = self.asech()
        except (ValueError,ZeroDivisionError,OverflowError) as err:
            rv = ZERO
        return rv

    def asech(self, *args):

        '''
        >>> q = Quaternion(0.1012, -0.2721, -0.06, 0.130)
        >>> print q.asech()
            1.8418   + 1.1216i   + 0.2473j   - 0.5359k
        >>> print q.asech().sech()
            0.1012   - 0.2721i   - 0.0600j   + 0.1300k
        '''
        absImag = self.absImag()

        if len(args) == 0:
            z = complex.Complex(self.r, absImag).asech()
            if absImag == 0.0:
                return self.makeQuaternion(z)
            else:
                return self.makeQuaternion(z.real(), z.imagI() / absImag)

        if len(args) >= 1:
            if absImag != 0.0 or self.r > 0.0 and self.r <= 1.0:
                return self.asech()

            z = complex.Complex(self.r, absImag).asech()
            return quaternion.branchCut(z.real(), z.imagI(), u)


    @canuse
    def zacsch(self):
        try:
            rv = self.acsch()
        except (ValueError,ZeroDivisionError,OverflowError) as err:
            rv = ZERO
        return rv

    def acsch(self, *args):
        '''
        >>> q = Quaternion(0.1012, -0.2721, -0.06, 0.130)
        >>> print q.acsch()
            1.7997   + 1.0940i   + 0.2412j   - 0.5227k
        >>> print q.acsch().csch()
            0.1012   - 0.2721i   - 0.0600j   + 0.1300k
        '''

        absImag = self.absImag()

        if len(args) == 0:
            z = complex.Complex(self.r, absImag).acsch()
            if absImag == 0.0:
                return self.makeQuaternion(z)
            else:
                return self.makeQuaternion(z.real(), z.imagI() / absImag)

        if len(args) >= 1:
            if self.r != 0.0 or absImag >= 1.0:
                return self.acsch()

            z = complex.Complex(self.r, absImag).acsch()
            return quaternion.branchCut(z.real(), z.imagI(), u)

    @canuse
    def zacoth(self, *args):
        try:
            rv = self.acoth()
        except (ValueError,ZeroDivisionError,OverflowError) as err:
            rv = ZERO
        return rv

    def acoth(self, *args):
        '''
        >>> q = Quaternion(0.1012, -0.2721, -0.06, 0.130)
        >>> print q.acoth()
            0.0926   + 1.1238i   + 0.2478j   - 0.5369k
        >>> print q.acoth().coth().conjugate()
            0.1012   - 0.2721i   - 0.0600j   + 0.1300k
        '''

        absImag = self.absImag()

        if len(args) == 0:
            z = complex.Complex(self.r, absImag).acoth()
            if absImag == 0.0:
                return self.makeQuaternion(z)
            else:
                return self.makeQuaternion(z.real(), z.imagI() / absImag)

        if len(args) >= 1:
            if absImag != 0.0 or self.r < -1.0 or self.r > 1.0:
                return self.acoth()

            z = complex.Complex(self.r, absImag).acoth()
            return quaternion.branchCut(z.real(), z.imagI(), u)


    @canuse
    def ceil(self):
        '''
        >>> q = Quaternion(10.1012, -4.2721, -0.500, 5.730)
        >>> print q.ceil()
           11.0000   - 4.0000i   - 0.0000j   + 6.0000k
        '''
        return Quaternion(math.ceil(self.r), math.ceil(self.i), math.ceil(self.j), math.ceil(self.k))


    @canuse
    def floor(self):
        '''
        >>> q = Quaternion(10.1012, -4.2721, -0.500, 5.730)
        >>> print q.floor()
           10.0000   - 5.0000i   - 1.0000j   + 5.0000k
        '''
        return Quaternion(math.floor(self.r), math.floor(self.i), math.floor(self.j), math.floor(self.k))


    @canuse
    def rint(self):             # Nearest integer, rounding .5 DOWN, wasn't my choice
        '''
        >>> q = Quaternion(10.1012, -4.2721, -0.500, 5.730)
        >>> print q.rint()
           10.0000   - 4.0000i   - 1.0000j   + 6.0000k
        '''
        return Quaternion(qrint(self.r), qrint(self.i), qrint(self.j), qrint(self.k))


    @canuse
    def shrink1(self):           # Return coefficients in the range [0-1) -- "OLD" version
        '''
        >>> print Quaternion(4.500, -2.500, 17.892, 11.000).shrink1()
            0.4500   + 0.2500i   + 0.1789j   + 0.1100k
        '''
        if False:
            return Quaternion(self.r - math.floor(self.r),\
                          self.i - math.floor(self.i),\
                          self.j - math.floor(self.j),\
                          self.k - math.floor(self.k))
        else:
            qr = abs(self.r)
            qi = abs(self.i)
            qj = abs(self.j)
            qk = abs(self.k)

            if math.isinf(qr) or math.isnan(qr): qr = 0.0
            if math.isinf(qi) or math.isnan(qi): qi = 0.0
            if math.isinf(qj) or math.isnan(qj): qj = 0.0
            if math.isinf(qk) or math.isnan(qk): qk = 0.0

            while qr >= 1.0: qr *= 0.1
            while qi >= 1.0: qi *= 0.1
            while qj >= 1.0: qj *= 0.1
            while qk >= 1.0: qk *= 0.1

            return Quaternion(qr, qi, qj, qk)

    @canuse
    def shrink(self):           # Return coefficients in the range [0.0039-1)
        '''
        >>> print Quaternion(4.500, -2.500, 17.892, 11.000).shrink1()
            0.4500   + 0.2500i   + 0.1789j   + 0.1100k
        '''
        if False:
            return Quaternion(self.r - math.floor(self.r),\
                          self.i - math.floor(self.i),\
                          self.j - math.floor(self.j),\
                          self.k - math.floor(self.k))
        else:
            qr = abs(self.r)
            qi = abs(self.i)
            qj = abs(self.j)
            qk = abs(self.k)

            if math.isinf(qr) or math.isnan(qr): qr = 0.0
            if math.isinf(qi) or math.isnan(qi): qi = 0.0
            if math.isinf(qj) or math.isnan(qj): qj = 0.0
            if math.isinf(qk) or math.isnan(qk): qk = 0.0

            while qr >= 1.0: qr /= 256.
            while qi >= 1.0: qi /= 256.
            while qj >= 1.0: qj /= 256.
            while qk >= 1.0: qk /= 256.

            while qr > 0 and qr < .0039: qr *= 256.
            while qi > 0 and qi < .0039: qi *= 256.
            while qj > 0 and qj < .0039: qj *= 256.
            while qk > 0 and qk < .0039: qk *= 256.

            return Quaternion(qr, qi, qj, qk)


    @staticmethod
    def get_actions():
        'Return array of actionable routines, i.e., those with @canuse decorator'
        return actions


    @staticmethod
    def unittest():
        'Run all the unit tests found in comments after the "def" statements'
        import doctest
        doctest.testmod()                                                      



## END of class Quaternion


ZERO = Quaternion(0.0, 0.0, 0.0, 0.0)
ONE  = Quaternion(1.0, 0.0, 0.0, 0.0) 
I    = Quaternion(0.0, 1.0, 0.0, 0.0)	 
J    = Quaternion(0.0, 0.0, 1.0, 0.0)	 
K    = Quaternion(0.0, 0.0, 0.0, 1.0)	 
NaN  = Quaternion(NaN, NaN, NaN, NaN) 



if __name__ == '__main__':

    # Main line

    Quaternion.unittest()

    print len(actions),actions
    print actions[0](Quaternion(0.1012, -0.2721, -0.06, 0.130))
    #print (Quaternion(0.1012, -0.2721, -0.06, 0.130)).(actions[0])
