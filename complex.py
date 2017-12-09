''' -*- Mode: Python; tab-width: 4; python-indent: 4; indent-tabs-mode: nil -*- '''

# Based on java package ca.uwaterloo.alumni.dwharder.Numbers
# Some documentation (although for the Java implementation) at:
# https://ece.uwaterloo.ca/~dwharder/Java/doc/ca/uwaterloo/alumni/dwharder/Numbers/Complex.html

import math
import pdb
import random

LOG10      = math.log(10.0)
PI2        = 1.5707963267948966   # pi/2.0
units      = "i"                  # Sometimes "j" is preferred, see complex.imaginaryUnit()
NaN        = float("NaN")
Infinity   = float('inf')
# isIterable, below, from http://bytes.com/topic/python/answers/514838-how-test-if-object-sequence-iterable
isIterable = lambda v: bool(getattr(v, '__iter__', False)) # Turns out we don't want strings so omit that
isInfinite = lambda r: math.isinf(r)
isNaN      = lambda r: math.isnan(r)
sign       = lambda r: math.copysign(1.0,r)
sqr        = lambda r: r*r



class Complex (object):


    def _Complex0(self):
        """
        >>> print Complex()
         0.0000 + 0.0000i

        """
        'Create a complex from zero arguments'
        self.r = self.i = 0.0
        return self
    

    def _Complex2(self,a,b):
        '''
        >>> print Complex(123.456, -888.888)
         123.4560 -888.8880i
        '''
        "Create and return a Complex, given real and imaginary parts, each assumed to be float."
        self.r = a
        self.i = b
        return self


    def _Complex1(self,b):
        '''
        >>> print Complex(888.888)
         0.0000 + 888.8880i
        '''
        "Create and return a Complex, given just the imaginary part, assumed to be float. Real part is 0."
        self.r = 0.0
        self.i = b
        return self


    def _ComplexA(self, a):
        '''
        >>> print Complex([-101.5, 203]) 
        -101.5000 + 203.0000i

        '''
        "Create and return a Complex, given a vector of real and imaginary parts, or just the imaginary"
        if len(a) == 1: # Just the imaginary
            return complex._Complex1(a[0])
        elif len(a) == 2: # Both real and imaginary
            return complex._Complex2(a[0], a[1]) 
        raise ValueError( "ComplexA() expecting an array of length 1 or 2, but got length " + len(a))


    @staticmethod                       # Make complex.random() a function, not a class method
    def random(arg):
        """
            result = complex.random(arg) where arg is:
            int 0: return Complex with random real and 0 imaginary
            int 1: return Complex with 0 real and random imaginary
            int 2: return Complex with random real and random imaginary (1+ from java code)
            array: return Complex with 0 or random real if array[0], and
                   0 or random imaginary if array[1]
        """

        self = Complex()                # Initialized to (0.0, 0.0)
        
        if type(arg) is int:            # complex.random(int)
            if arg == 0:
                self.r = random.random()
            elif arg == 1:
                self.i = random.random()
            elif arg == 2:
                self.r = random.random()
                self.i = random.random()
            else:
                raise ValueError( "Bad int arg (%d) passed to complex.random()" % arg)
                    
        elif isIterable(arg): # Expect bool vector of length 2 or more
            if arg[0]: self.r = random.random()
            if arg[1]: self.i = random.random()

        else:                           # complex.random(garbage)
            raise ValueError("Unknown argument type (%s) passed to complex.random()" % type(arg))
        return self 


    @staticmethod
    def randomReal():
        'Return complex with random real and zero imaginary'
        return Complex(random.random(), 0.0)


    @staticmethod
    def randomImaginary():
        'Return complex with random imaginary and zero real'
        return Complex(0.0, random.random())


    def __init__(self, *args):
        'Principal entry point to create and initialize a Complex object.'

        self.r = self.i = 0
        if len(args) == 0:
            self = Complex._Complex0(self)
        elif len(args) == 1:
            args0 = args[0]
            ta = type(args0)          # Ipart or vector
            if ta is int:               # Convert int to float, but don't change args[0]
                args0 = float(args0)
                ta    = float

            if ta is float:             # Complex(foo) ==> _Complex1(foo)
                self = Complex._Complex1(self, args0)
            elif isIterable(args0):
                lz = len(args[0])
                if lz == 1:             # Complex([foo]) ==> _Complex1(foo)
                    self = Complex._Complex1(self, args[0][0])
                elif lz==2:             # Complex([foo,bar]) ==> _Complex2(foo,bar)
                    self = Complex._Complex2(self, args[0][0], args[0][1])
                else:
                    raise ValueError( "Bad argument (%s) to Complex()" % (args,))
        elif len(args) == 2:
            self = Complex._Complex2(self, args[0], args[1])
        else:
            raise TypeError( "Bad argument (%s) passed to Complex()" % (args,))
        return None


    def __repr__(self):
        return "Complex(%8.4f,%8.4f)" % (self.r,self.i)


    def __str__(self):
        """
        >>> Number = Complex(1.2345,2.446688)
        >>> print "str(Number):", Number
        str(Number):  1.2345 + 2.4467i
        >>> Number = Complex(1.2345,-2.446688)
        >>> print "str(Number):", Number
        str(Number):  1.2345 -2.4467i
        >>> print Complex(12345678.90,0.000000008123)
         12345678.9000 +0.00000000812i

        """
        'Tries to ensure plenty of space for small numbers'
        pm = '+'
        fracr = fraci = 4       # Fractional part
        if self.i < 0: pm = ''
        if abs(self.r) < 0.001:
            widr = 7
            if abs(self.r) > 0.000000000001:
                fracr = abs(math.log10(abs(self.r))) + 3
        else:
            widr = 7+max(0,math.log10(abs(self.r)))

        if abs(self.i) < 0.001:
            widi = 7
            if abs(self.i) > 0.000000000001:
                fraci = abs(math.log10(abs(self.i))) + 3
        else:
            widi = 7+max(0,math.log10(abs(self.i)))
        fmt  = "%s%d.%df %s%s%d.%df%s" % ("%", widr, fracr, pm, "%", widi, fraci, units)
        return fmt % (self.r, self.i)


    def equals(self, *args):
        """
        >>> print Complex(11.0,4.33).equals(Complex(11.01, 4.332),.0002)
        False
        >>> print Complex(11.0,4.33).equals(Complex(11.0001, 4.332),.0002)
        True
        """
        '''
            Returns True or False depending on arguments being equal or not. If a third
            argument is given, it is taken as eps, a tolerance.
        '''

        if len(args) == 1:              # Test for absolute equality
            w = args[0]                 # Pick up the first argument, what might equal self
            if isinstance(w, Complex):
                if w.r == self.r and w.i == self.i:
                    return True
            return False   

        if len(args) > 1:               # eps supplied?
            eps = args[1]               # Yes
            z   = args[0]               # First argument

            if eps < 0.0:
                raise ValueError( "Illegal value for tolerance (%8.5f) < 0 passed to equals()" % eps)

            # We're going to divide by abs(z), so first see if it's zero. If zero, test self.abs()
            # for being near zero and return True or False accordingly.

            if z.isZero():
                if self.abs() < eps: return True
                return False

            # Bug in java code. We need to ensure self.abs() nonzero since otherwise we're
            # going to divide by it ... whoops. But we can still do a sane check vs. z.

            if self.abs() == 0:         # Test the exact thing we might divide by
                if z.abs() < eps:       # Is other value close to 0?
                    return True         # Why yes, zero and close-to-zero
                return False            # No, zero and far-from-zero

            # Here when z is nonzero. Check absolute value of their difference, divided by the
            # smaller of their absolute values, against eps. Note .abs() produces a real value.

            if (self.subtract(z).abs() / min(z.abs(), self.abs())) < eps:
                return True
        
        return False

    def hashCode(self):
        'Hash a complex number. Python hashes 0.0 to 0 so (0.0 + 0.0i).hashCode() == 0'
        rh = hash(self.r)
        ih = hash(self.i)

        return rh + 536870923 * ih


    def toString(self, *args):
        'No idea what the multiple-argument call is supposed to do, so stringify the self argument.'
        return self.str()
    

    @staticmethod
    def imaginaryUnit(s):
        'Set unit to first character of s, return previous unit. Usually used to change i to j.'
        oldunit = units[0]
        units   = s[0]
        return oldUnit

    @staticmethod
    def toMathMLContentString(x):
        if math.isnan(x):
            return "<notanumber/>"

        if math.isinf(x):
            if x < 0:
                return "<apply><minus/><infinity/></apply>"
            else:
                return "<infinity/>"

        return "<cn>" + str(x) + "</cn>"


    def toMathMLContent(self):

        if self.isNaN() or self.isInfinite():
            real = Complex.toMathMLContentString(self.r)
            if complex.sign(self.i) == 1.0:
                sign = "<plus/>"
            else:
                sign = "<minus/>"
            imag = Complex.toMathMLContentString(abs(self.i))
            return "<apply>" + sign + real + "<apply><times/><imaginaryi/>" + imag + "</apply></apply>"

        return "<cn type=\"complex-cartesian\">" + str(self.r) + "<sep/>" + str(self.i) + "</cn>"


    def toMathMLPresentation(self):
        if isInfinite(self.r):
            if self.r > 0.0: real = "<mi>&infin;</mi>"
            else: real = "<mrow><mo>-</mo><mi>&infin;</mi></mrow>"
        elif isNaN(self.r): real = "<mi>&NaN;</mi>"
        else: real = "<mn>" + str(self.r) + "</mn>"
        
        if isInfinite(self.i): imag = "<mi>&infin;</mi>"
        elif isNaN(self.i): imag = "<mi>&NaN;</mi>"
        else: imag = "<mn>" + str(abs(self.i)) + "</mn>"

        if complex.sign(self.i) == 1.0:
            sign = "<mo>+</mo>"
        else: sign = "<mo>-</mo>"

        return "<mrow>" + real + sign + "<mrow><mo><mi>&ImaginaryI;</mi>&InvisibleTimes;</mo>" + imag + "</mrow></mrow>"


    def toArray(self, *args):
        if len(args) == 0:
            return [self.r, self.i]

        result = []
        length = len(args[0])
        for i in range(length):
            result.append(self.coefficient(i))
        return result


    def isZero(self):
        'Return True or False depending on self being (0,0) or not'
        if self.r == 0.0 and self.i == 0.0:
            return True
        return False


    def isReal(self):
        if self.i == 0.0:
            return True

        return False


    def isImaginary(self):
        if self.r == 0.0:
            return True

        return False


    def isNaN(self):
        'Return True if either part of self is NaN. Otherwise return False.'
        if math.isnan(self.r) or math.isnan(self.i):
            return True

        return False


    def isInfinite(self):
        'Converted from java code.'
        if not (isInfinite(self.r) or isInfinite(self.i)):
            return False
        else:
            return True


    def abs(self):
        return math.sqrt(self.abs2())


    def abs2(self):
        return self.r * self.r + self.i * self.i


    def real(self):
        'Return the real part of self'
        return self.r


    def imag(self):
        'Return a complex with 0 real and self.i imaginary part'
        return Complex(0.0, self.i)


    def imagI(self):
        'Return the imaginary part of self as a real'
        return self.i


    def coefficient(self, n):
        'Return the nth coefficient of self. n==0 returns the real part, n==1 the imaginary.'
        if n == 0:
            return self.r
        elif n == 1:
            return self.i
        else:
            raise ValueError( "complex.coefficient(): the argument n must be 0 or 1 but got %d" % n)


    def project(self, *args):
        'Like coeffcient(), but returns values as appropriate complex numbers'
        if isinstance(args[0], int) or isinstance(args[0], long):
            if args[0] == 0:
                return Complex(self.r, 0.0)
            elif args[0] == 1:
                return Complex(0.0, self.i)
            else:
                raise ValueError( "complex.project(): the argument n must be 0 or 1 but got %d" % n)
        elif isIterable(args[0]):
            rh = ih = 0.0
            if args[0][0]: rh = self.r
            if args[0][1]: ih = self.i
            return Complex(rh,ih)


    def conjugate(self):
        return Complex(self.r, - self.i)


    def argument(self):
        return math.atan2(self.i, self.r)


    def signum(self):
        s_abs = self.abs()
        if s_abs == 0.0:
            return self
 
        if isInfinite(s_abs) or isNaN(s_abs):
            if isInfinite(self.r) and isInfinite(self.i):
                return Complex(NaN, NaN)

            if self.isNaN():
                return Complex(NaN, NaN)

            if isInfinite(self.r):
                return ONE

            return I

        s_abs = math.sqrt(s_abs)
        return Complex(self.r / s_abs, self.i / s_abs)


    @staticmethod
    def isPositiveZero(r):
        "Some platforms support positive/negative zero. Return True if argument is positive zero."
        if r is 0.0:                    # Is distinguishes 0.0 and -0.0
            return True
        return False


    def csgn(self):
        'Complex sgn, apparently'
        if self.r  > 0.0: return  1.0
        if self.r  < 0.0: return -1.0
        if self.i == 0.0: return  0.0   # It's this way in the java code
        if Complex.isPositiveZero(self.r) or isNaN(self.r):
            return 1.0
        return -1.0


    def add(self, *args):
        '''
        >>> a = Complex( 47.8000, 23.0000)
        >>> b = Complex(-11.0500, 3.1300)
        >>> print a.add(b)
         36.7500 + 26.1300i
        >>> print a.add(1,23.45)
         47.8000 + 46.4500i
        >>> print b.add(11.03)
        -0.0200 + 3.1300i
        '''
        "Various add opportunities"

        if not isinstance(self.r, float):
            print "CRAZY ARGUMENT"
            #pdb.set_trace()

        args0 = args[0]

        # [1] add a Complex to a Complex
        if isinstance(args0, Complex):
            return Complex(self.r + args0.r, self.i + args0.i)

        # [2] add a real to the nth component
        if isinstance(args0, int) or isinstance(args0, long):
            if   args0 == 0:
                return Complex(self.r + args[1], self.i)
            elif args0 == 1:
                return Complex(self.r, self.i + args[1])
            else:
                raise ValueError("add(): the argument n must be 0 or 1 but got " + n)

        # [3] add a float to the real component
        if isinstance(args0, float):
            return Complex(self.r + args0, self.i)


    def addI(self, x):
        "Add float x to self.i, don't change self.r"
        return Complex(self.r, self.i + x)


    def subtract(self, w):
        '''
        >>> a = Complex( 47.8000, 23.0000)
        >>> b = Complex(-11.0500, 3.1300)
        >>> print a.subtract(b)
         58.8500 + 19.8700i
        '''
        'subtract complex w pointwise from self'
        return Complex(self.r - w.r, self.i - w.i)


    def negate(self):
        '''
        >>> b = Complex(-11.0500, 3.1300)
        >>> print b.negate()
         11.0500 -3.1300i
        '''
        'Negate both r and i fields of argument'
        return Complex(- self.r, - self.i)


    def multiply(self, *args):
        '''
        >>> a = Complex( 47.8000, 23.0000)
        >>> b = Complex(-11.0500, 3.1300)
        >>> print a.multiply(b)
        -600.1800 -104.5360i
        >>> print a.multiply(101.0101)
         4828.2828 + 2323.2323i
        '''
        'Multiply complex self by a float constant, or another complex'
        args0 = args[0]

        # [1] Float argument (with auto-convert of int to float)
        if isinstance(args0, int) or isinstance(args0, long):
            args0 = float(args0) # Allow lazy coding
        if isinstance(args0, float):
            if isInfinite(args0) or isNaN(args0): # [1a] Edge cases

                if self.r == 0.0 and self.i != 0.0:
                    return Complex(self.r * Complex.sign(args0), self.i * args0)

                if self.r != 0.0 and self.i == 0.0:
                    return Complex(self.r * args0, self.i * Complex.sign(x))

            return Complex(self.r * args0, self.i * args0) # [1b] Normal case

        # [2] Complex argument. Handle edge cases, then standard case.
        if isinstance(args0, Complex):
            # [2a] A bunch of edge cases
            if self.isNaN() or args0.isNaN() or self.isInfinite() or args0.isInfinite():
                if self.isZero() or args0.isZero():
                    return ZERO

                if self.r == 0.0 and args0.r == 0.0:
                    return Complex((- self.i) * args0.i, 0.0)

                if self.r == 0.0 and args0.i == 0.0:
                    return Complex(0.0, self.i * args0.r)

                if self.i == 0.0 and args0.r == 0.0:
                    return Complex(0.0, self.r * args0.i)

                if self.i == 0.0 and args0.i == 0.0:
                    return Complex(self.r * args0.r, 0.0)

                if self.r == 0.0:
                    return Complex((- self.i) * args0.i, self.i * args0.r)

                if self.i == 0.0:
                    return Complex(self.r * args0.r, self.r * args0.i)

                if args0.r == 0.0:
                    return Complex((- self.i) * args0.i, self.r * args0.i)

                if args0.i == 0.0:
                    return Complex(self.r * args0.r, self.i * args0.r)

                if isNaN(self.r) and isNaN(self.i) or isNaN(args0.r) and isNaN(args0.i):
                    return Complex(NaN, NaN)

                if self.isInfinite() or args0.isInfinite():
                    return Complex(Infinity, Infinity)


            # [2b] The normal case
            return Complex (self.r * args0.r - self.i * args0.i, self.r * args0.i + self.i * args0.r)
        else:
            raise ValueError("Can only multiply Complex by Complex or float")


    def multiplyI(self, x):
        '''
        >>> a = Complex( 47.8000, 23.0000)
        >>> print a.multiplyI(10.)
        -230.0000 + 478.0000i
        '''
        "Multiply self by float taken to be an imaginary constant"
        if isInfinite(x) or isNaN(x):
            # Edge cases
            if self.r == 0.0 and self.i != 0.0:
                return Complex((- self.i) * x, self.r * Complex.sign(x))
            if self.r != 0.0 and self.i == 0.0:
                return Complex((- self.i) * Complex.sign(x), self.r * x)
        # Normal case 
        return Complex((- self.i) * x, self.r * x)


    def divide(self, w):
        '''
        >>> a = Complex(47.8,23.0)
        >>> d = Complex(1755.8400, 2198.8000)
        >>> q = d.divide(a)
        >>> print "Calculated quotient:",q
        Calculated quotient:  47.8000 + 23.0000i
        >>> print "Correct answer is:", a
        Correct answer is:  47.8000 + 23.0000i
        >>> print Complex(527., 192.63).divide(Complex(131.53, 53.131))
         3.9532 -0.1324i
        '''
        "Divide one complex by another. Evidently too difficult to handle edge cases."
        abs2 = w.abs2()
        return Complex((self.r * w.r + self.i * w.i) / abs2, ((- self.r) * w.i + self.i * w.r) / abs2)


    def pow(self, *args):
        '''
        >>> a = Complex(47.8,23.0)
        >>> print a.sqr()
         1755.8400 + 2198.8000i
        >>> print a.pow(2)
         1755.8400 + 2198.8000i
        >>> print a.multiply(a)
         1755.8400 + 2198.8000i
        '''
        "Raise self to powers of various types"

        if isIterable(args):
            args0 = args[0]
        else:
            args0 = args

        if isinstance(args0, Complex):
            a = 0.5 * math.log(self.abs2())
            b = math.atan2(self.i, self .r)
            c = math.exp(w.r * a - w.i * b)
            d = w.i * a + w.r * b
            return Complex(c * math.cos(d), c * math.sin(d))

        elif isinstance(args0, float):
            if args0 == 0.0:
                return ONE

            if self.isZero():
                return self

            if self.i == 0.0:
                return Complex(math.pow(self.r, args0), self.i)

            expr = math.pow(self.abs2(), 0.5 * args0) # 
            arg  = args0 * math.atan2(self.i, self.r)
            return Complex(expr * math.cos(arg), expr * math.sin(arg))

        elif isinstance(args0,int) or isinstance(args0,long):
            n = args0           # DS cut-n-paste
            if n == 0:
                return ONE

            if self.r == 0.0 and self.i == 0.0:
                return self

            if self.i == 0.0:
                return Complex(math.pow(self.r, n), self.i)

            if self.r == 0.0:
                result = n % 4
                if result < 0:
                    result += 4

                if result == 0:
                    return Complex(math.pow(self.i, n), 0.0)

                elif result == 1:
                    return Complex(0.0, math.pow(self.i, n))

                elif result == 2:
                    return Complex(- math.pow(self.i, n), 0.0)

                elif result == 3:
                    return Complex(0.0, - math.pow(self.i, n))

                return None

            if (n % 2) == 0:
                expr = math.pow(self.abs2(), n / 2)
            else:
                expr = math.pow(self.abs(), n)

            arg = n * math.atan2(self.i, self.r)
            return Complex(expr * math.cos(arg), expr * math.sin(arg))
        else:
            raise NotImplementedError("No code for Complex ** %s" % type(args0))


    def sqr(self):
        'Square a complex'

        """
        sqrn = Complex(5.2, -277.1)
        >>> print sqrn
        5.2000 -277.1000i
        >>> print sqrn.sqr()
        -76757.3700 -2881.8400i
        >>> print sqrn.sqr().sqrt()
        5.2000 -277.1000i
        """
        return Complex(self.r * self.r - self.i * self.i, 2.0 * self.r * self.i)


    def horner (self, *args):
        '''
        >>> v = [2, -7, 5, -65][::-1]    # Evaluate 2x^3 - 7x^2 + 5x - 65
        >>> b = Complex(12.2, -4.66)
        >>> p = b.horner(v)
        >>> print p
         1148.2433 -3186.5490i
        >>> p = Complex(10.,0.).horner(v)
        >>> print p
         1285.0000 + 0.0000i
        '''
        "Evaluate a polynomial using Horner's rule. Argument is an iterable of coefficients."
        vec = args[0]
        if isIterable(vec):     # Acceptable?
            pass                # Yes, skip over complaint
        else:
            raise TypeError("horner() needs an iterable, %s is not iterable" % str(type(vec)))

        if len(vec) == 0:
            return ZERO

        if isinstance(vec[0], int) or isinstance(vec[0], long): # Convert int to float
            vec = [float(x) for x in vec]

        if isinstance(vec[0], float):
            if self.isNaN() or self.isInfinite():
                #pdb.set_trace()
                result = self.multiply(vec[-1])
                for i in range(len(vec)-2, 0, -1): # All but the two endpoints
                    result = self.multiply(result.add(vec[i]))
                return result.add(vec[0])
            else:               # Normal path
                a = self.r * vec[-1]
                b = self.i * vec[-1]
                for j in range(len(vec)-2, 0, -1): # All but the two endpoints
                    at = a + vec[j]
                    a = self.r * at - self.i * b
                    b = self.r * b  + self.i * at
                return Complex(a + vec[0], b)

        elif isinstance(vec[0],Complex):
            result = self.multiply(v[-1]) # Big endpont
            for i in range(len(v)-2, 0, -1): # All but the two endpoints
                result = self.multiply(result.add(v[i]))
            return result.add(v[0]) # Little endpoint

        else:
            raise TypeError("Unhandled datatype (%s) passed to horner()" % type(vec[0]))


    def sqrt(self):
        """
        >>> a = Complex(0.73,0.22)
        >>> print a.sqrt()
         0.8638 + 0.1273i
        >>> print a.sqrt().sqr()
         0.7300 + 0.2200i
        """
        'Complex square root'
        val = math.sqrt(self.abs2())
        return Complex(0.5 * math.sqrt(2.0 * val + 2.0 * self.r), 0.5 * Complex(self.i, -self.r).csgn() * math.sqrt(2.0 * val - 2.0 * self.r))


    def exp(self):
        """ 
        'Complex exponential'
        >>> elog = Complex(3.0, 6.152)
        >>> print "Key:", elog
        Key:  3.0000 + 6.1520i
        >>> print 'Exp:', elog.exp()
        Exp:  19.9130 -2.6274i
        >>> print 'B&F:',elog.exp().log()
        B&F:  3.0000 -0.1312i
        """

        expr = math.exp(self.r)
        return Complex(expr * math.cos(self.i), expr * math.sin(self.i))


    def log(self):
        'Complex natural log'

        if self.i == 0.0:       # Pure real number?
            if self.r < 0.0:
                return Complex(0.5 * math.log(self.abs2()), 3.141592653589793 * Complex.sign(self.i))

            if self.r == 0.0:
                return Complex(-Infinity, NaN)

            return Complex(0.5 * math.log(self.abs2()), 0.0 * Complex.sign(self.i)) # self.r > 0.0

        return Complex(0.5 * math.log(self.abs2()), math.atan2(self.i, self.r))


    def log10(self):
        return Complex(0.5 * math.log(self.abs2()) / LOG10, math.atan2(self.i, self.r) / LOG10)


    # All 6 trig functions
    def sin(self):
        """
        >>> angle = Complex(0.30, 0.40)
        >>> print angle
         0.3000 + 0.4000i
        >>> print angle.sin()
         0.3195 + 0.3924i
        >>> print angle.sin().asin()
         0.3000 + 0.4000i
        """
        if self.i == 0.0:
            return Complex(math.sin(self.r), self.i)

        if self.r == 0.0:
            return Complex(self.r, math.sinh(self.i))

        return Complex(math.sin(self.r) * math.cosh(self.i), math.cos(self.r) * math.sinh(self.i))


    def cos(self):
        """
        >>> angle = Complex(0.30, 0.40)
        >>> print angle
         0.3000 + 0.4000i
        >>> print angle.cos()
         1.0328 -0.1214i
        >>> print angle.cos().acos()
         0.3000 + 0.4000i

        """
        if self.i == 0.0:
            return Complex(math.cos(self.r), self.i)

        if self.r == 0.0:
            return Complex(math.cosh(self.i), self.r)

        return Complex(math.cos(self.r) * math.cosh(self.i), (- math.sin(self.r)) * math.sinh(self.i))


    def tan(self):
        """
        >>> angle = Complex(.3000, .4000)
        >>> print angle
         0.3000 + 0.4000i
        >>> print angle.tan(),angle.sin().divide(angle.cos())
         0.2611 + 0.4106i  0.2611 + 0.4106i
        >>> print angle.tan().atan()
         0.3000 + 0.4000i
        """
        if self.i == 0.0:
            return Complex(math.tan(self.r), self.i)

        if self.r == 0.0:
            return Complex(self.r, math.tanh(self.i))

        denom = sqr(math.cos(self.r)) + sqr(math.sinh(self.i))
        if denom == 0.0:
            raise ZeroDivisionError("Division by zero in Complex tan(self) call")

        return Complex(math.sin(self.r) * (math.cos(self.r) / denom), math.sinh(self.i) / denom * math.cosh(self.i))


    def sec(self):
        '''
        >>> a = Complex(.5,1.2)
        >>> print a.sec()
         0.5212 + 0.2374i
        >>> print a.cos().multiply(a.sec())
         1.0000 + 0.0000i
        '''
        if self.i == 0.0:
            return Complex(1.0 / math.cos(self.r), self.i)

        if self.r == 0.0:
            return Complex(1.0 / math.cosh(self.i), self.r)

        denom = sqr(math.cos(self.r)) + sqr(math.sinh(self.i))
        if denom == 0.0:
            raise ZeroDivisionError("Division by zero in Complex sec(self) call")

        return Complex(math.cosh(self.i) * (math.cos(self.r) / denom), math.sinh(self.i) / denom * math.sin(self.r))


    def csc(self):
        '''
        >>> a = Complex(.5,1.2)
        >>> print a.csc()
         0.3461 -0.5281i
        >>> print a.sin().multiply(a.csc())
         1.0000 + 0.0000i
        '''
        if self.i == 0.0:
            return Complex(1.0 / math.sin(self.r), self.i)

        if self.r == 0.0:
            return Complex(self.r, -1.0 / math.sinh(self.i))

        denom = sqr(math.sin(self.r)) + sqr(math.sinh(self.i))
        if denom == 0.0:
            raise ZeroDivisionError("Division by zero in Complex csc(self) call")

        return Complex(math.cosh(self.i) * (math.sin(self.r) / denom), (- math.sinh(self.i) / denom) * math.cos(self.r))


    def cot(self):
        '''
        >>> a = Complex(.5,1.2)
        >>> print a.cot()
         0.1677 -1.0896i
        >>> print a.tan().multiply(a.cot())
         1.0000 + 0.0000i
        '''

        if self.i == 0.0:
            return Complex(1.0 / math.tan(self.r), self.i)

        if self.r == 0.0:
            return Complex(self.r, -1.0 / math.tanh(self.i))

        denom = sqr(math.sin(self.r)) + sqr(math.sinh(self.i))
        if denom == 0.0:
            raise ZeroDivisionError("Division by zero in Complex csc(self) call")

        return Complex(math.cos(self.r) * (math.sin(self.r) / denom), (- math.sinh(self.i) / denom) * math.cosh(self.i))

    # 6 Hyperbolic trig functions
    def sinh(self):
        """
        >>> angle = Complex(0.30,0.40)
        >>> sinhangle = angle.sinh()
        >>> print angle
         0.3000 + 0.4000i
        >>> print sinhangle
         0.2805 + 0.4071i
        >>> print sinhangle.asinh()
         0.3000 + 0.4000i
        """
        if self.i == 0.0:
            return Complex(math.sinh(self.r), self.i)

        if self.r == 0.0:
            return Complex(self.r, math.sin(self.i))

        return Complex(math.sinh(self.r) * math.cos(self.i), math.cosh(self.r) * math.sin(self.i))


    def cosh(self):
        """
        >>> angle = Complex(0.30,0.40)
        >>> print angle
         0.3000 + 0.4000i
        >>> coshangle = angle.cosh()
        >>> print coshangle
         0.9628 + 0.1186i
        >>> print coshangle.acosh()
         0.3000 + 0.4000i
        """

        if self.i == 0.0:
            return Complex(math.cosh(self.r), self.i)

        if self.r == 0.0:
            return Complex(math.cos(self.i), self.r)

        return Complex(math.cosh(self.r) * math.cos(self.i), math.sinh(self.r) * math.sin(self.i))

        """qaz
        """

    def tanh(self):
        """
        >>> angle = Complex(0.30,0.40)
        >>> print angle
         0.3000 + 0.4000i
        >>> sinhangle = angle.sinh()
        >>> coshangle = angle.cosh()
        >>> tanhangle = angle.tanh()
        >>> print tanhangle, sinhangle.divide(coshangle)
         0.3383 + 0.3811i  0.3383 + 0.3811i
        >>> print tanhangle.atanh()
         0.3000 + 0.4000i
        """

        if self.i == 0.0:
            return Complex(math.tanh(self.r), self.i)

        if self.r == 0.0:
            return Complex(self.r, math.tan(self.i))

        denom = sqr(math.sinh(self.r)) + sqr(math.cos(self.i))
        if denom == 0.0:
            raise ZeroDivisionError("Division by zero in Complex tanh() call")

        return Complex(math.cosh(self.r) * (math.sinh(self.r) / denom), math.sin(self.i) * (math.cos(self.i) / denom))


    def sech(self):
        '''
        >>> a = Complex(.5,1.2)
        >>> print a.cosh()
         0.4086 + 0.4857i
        >>> print a.cosh().multiply(a.sech())
         1.0000 + 0.0000i
        '''
        if self.i == 0.0:
            return Complex(1.0 / math.cosh(self.r), self.i)

        if self.r == 0.0:
            return Complex(1.0 / math.cos(self.i), self.r)

        denom = sqr(math.sinh(self.r)) + sqr(math.cos(self.i))
        if denom == 0.0:
            raise ZeroDivisionError("Division by zero in Complex sech() call")

        return Complex(math.cosh(self.r) * (math.cos(self.i) / denom), (- math.sin(self.i)) * (math.sinh(self.r) / denom))


    def csch(self):
        '''
        >>> a = Complex(.5,1.2)
        >>> print a.sinh()
         0.1888 + 1.0510i
        >>> print a.sinh().multiply(a.csch())
         1.0000 + 0.0000i
        '''
        if self.i == 0.0:
            return Complex(1.0 / math.sinh(self.r), self.i)

        if self.r == 0.0:
            return Complex(self.r, -1.0 / math.sin(self.i))

        denom = sqr(math.sinh(self.r)) + sqr(math.sin(self.i))
        if denom == 0.0:
            raise ZeroDivisionError("Division by zero in Complex csch() call")

        return Complex(math.cos(self.i) * (math.sinh(self.r) / denom), (- math.cosh(self.r)) * (math.sin(self.i) / denom))


    def coth(self):
        '''
        >>> a = Complex(.5,1.2)
        >>> print a.coth()
         0.5153 -0.2962i
        >>> print a.coth().multiply(a.tanh())
         1.0000 + 0.0000i
        '''
        if self.i == 0.0:
            return Complex(1.0 / math.tanh(self.r), self.i)

        if self.r == 0.0:
            return Complex(self.r, -1.0 / math.tan(self.i))

        denom = sqr(math.sinh(self.r)) + sqr(math.sin(self.i))
        if denom == 0.0:
            raise ZeroDivisionError("Division by zero in Complex coth() call")

        return Complex(math.cosh(self.r) * (math.sinh(self.r) / denom), (- math.cos(self.i)) * (math.sin(self.i) / denom))

    # 6 arc trig functions
    def asin(self):
        '''
        >>> a = Complex(0.42,0.2723)
        >>> print a.asin()
         0.4143 + 0.2932i
        >>> print a.asin().sin()
         0.4200 + 0.2723i
        '''
        if self.i == 0.0:
            if self.r < -1.0 or self.r > 1.0:
                v1 = 0.5 * math.sqrt(self.r * self.r + 2.0 * self.r + 1.0)
                v2 = 0.5 * math.sqrt(self.r * self.r - 2.0 * self.r + 1.0)
                return Complex(1.5707963267948966, Complex(self.i, - self.r).csgn() * math.log(v1 + v2 + math.sqrt(sqr(v1 + v2) - 1.0)))

            return Complex(math.asin(self.r), 0.0)

        v1 = 0.5 * math.sqrt(self.r * self.r + 2.0 * self.r + 1.0 + self.i * self.i)
        v2 = 0.5 * math.sqrt(self.r * self.r - 2.0 * self.r + 1.0 + self.i * self.i)
        try:
            rv = Complex(math.asin(v1 - v2), Complex(self.i, - self.r).csgn() * math.log(v1 + v2 + math.sqrt(sqr(v1 + v2) - 1.0)))
        except ValueError as ve:
            rv = Complex(0.0,0.0)
        return rv

    def acos(self):
        '''
        >>> a = Complex(0.2723,0.42)
        >>> print a.acos()
         1.3183 -0.4212i
        >>> print a.acos().cos()
         0.2723 + 0.4200i
        '''
        if self.i == 0.0:
            if self.r < -1.0 or self.r > 1.0:
                v1 = 0.5 * math.sqrt(self.r * self.r + 2.0 * self.r + 1.0)
                v2 = 0.5 * math.sqrt(self.r * self.r - 2.0 * self.r + 1.0)
                return Complex(1.5707963267948966, (- Complex(self.i, - self.r).csgn()) * math.log(v1 + v2 + math.sqrt(sqr(v1 + v2) - 1.0)))

            return Complex(math.acos(self.r), 0.0)

        v1 = 0.5 * math.sqrt(self.r * self.r + 2.0 * self.r + 1.0 + self.i * self.i)
        v2 = 0.5 * math.sqrt(self.r * self.r - 2.0 * self.r + 1.0 + self.i * self.i)
        try:
            rv = Complex(math.acos(v1 - v2), (- Complex(self.i, - self.r).csgn()) * math.log(v1 + v2 + math.sqrt(sqr(v1 + v2) - 1.0)))
        except ValueError as ve:
            rv = Complex(0.0,0.0)
        return rv

    def atan(self):
        '''
        >>> a = Complex(47.8,23.0)
        >>> print a.atan()
         1.5538 + 0.0082i
        >>> print a.atan().tan()
         47.8000 + 23.0000i
        '''
        if self.r == 0.0:
            if self.i == 1.0:
                return Complex(NaN, Infinity)

            if self.i == -1.0:
                return Complex(NaN, -Infinity)

            if self.i <= -1.0 or self.i >= 1.0:
                tf = Complex.sign(self.r) # java code has complex.isPositiveZero(self.r), looks wrong
                if tf:
                    fac = 1.0
                else: fac = -1.0
                return Complex(fac * 1.5707963267948966, 0.25 * math.log(sqr(self.i + 1.0) / sqr(self.i - 1.0)))

            return Complex(0.0, 0.25 * math.log(sqr(self.i + 1.0) / sqr(self.i - 1.0)))

        return Complex(0.5 * (math.atan2(self.r, 1.0 - self.i) - math.atan2(- self.r, self.i + 1.0)), 0.25 * math.log((self.r * self.r + sqr(self.i + 1.0)) / (self.r * self.r + sqr(self.i - 1.0))))


    def asec(self):
        '''
        >>> a = Complex(0.73,0.22)
        >>> print a.asec()
         0.4146 + 0.8379i
        >>> print a.asec().sec()
         0.7300 + 0.2200i
        '''
        if self.r == 0.0 and self.i == 0.0:
            return Complex(Infinity, Infinity)

        if self.i == 0.0:
            c = 0.5 * abs((self.r + 1.0) / self.r)
            d = 0.5 * abs((self.r - 1.0) / self.r)
            if self.r > -1.0 and self.r < 0.0:
                return Complex(3.141592653589793, Complex(self.i, self.r).csgn() * math.log(c + d + math.sqrt(sqr(c + d) - 1.0)))

            if self.r > 0.0 and self.r < 1.0:
                return Complex(0.0, Complex(self.i, self.r).csgn() * math.log(c + d + math.sqrt(sqr(c + d) - 1.0)))

            return Complex(math.acos(1.0 / self.r), Complex(self.i, self.r).csgn() * 0.0)

        a = sqr(self.i / self.abs2())
        b = self.r / self.abs2()
        c = 0.5 * math.sqrt(sqr(b + 1.0) + a)
        d = 0.5 * math.sqrt(sqr(b - 1.0) + a)
        return Complex(math.acos(c - d), Complex(self.i, self.r).csgn() * math.log(c + d + math.sqrt(sqr(c + d) - 1.0)))


    def acsc(self):
        '''
        >>> a = Complex(0.73,0.22)
        >>> print a.acsc()
         1.1562 -0.8379i
        >>> print a.acsc().csc()
         0.7300 + 0.2200i
        '''

        if self.r == 0.0 and self.i == 0.0:
            return Complex(Infinity, Infinity)

        if self.i == 0.0:
            c = 0.5 * abs((self.r + 1.0) / self.r)
            d = 0.5 * abs((self.r - 1.0) / self.r)
            if self.r > -1.0 and self.r < 0.0:
                return Complex(-1.5707963267948966, (- Complex(self.i, self.r).csgn()) * math.log(c + d + math.sqrt(sqr(c + d) - 1.0)))

            if self.r > 0.0 and self.r < 1.0:
                return Complex(1.5707963267948966, (- Complex(self.i, self.r).csgn()) * math.log(c + d + math.sqrt(sqr(c + d) - 1.0)))

            return Complex(math.asin(1.0 / self.r), (- Complex(self.i, self.r).csgn()) * 0.0)

        a = sqr(self.i / self.abs2())
        b = self.r / self.abs2()
        c = 0.5 * math.sqrt(sqr(b + 1.0) + a)
        d = 0.5 * math.sqrt(sqr(b - 1.0) + a)
        return Complex(math.asin(c - d), (- Complex(self.i, self.r).csgn()) * math.log(c + d + math.sqrt(sqr(c + d) - 1.0)))


    def acot(self):
        '''
        >>> a = Complex(0.73,0.22)
        >>> print a.acot()
         0.9250 -0.1429i
        >>> print a.acot().cot()
         0.7300 + 0.2200i
        '''
        if self.r == 0.0:
            if self.i == 1.0:
                return Complex(NaN, -Infinity)

            if self.i == -1.0:
                return Complex(NaN, Infinity)

            if self.i <= -1.0 or self.i >= 1.0:
                if Complex.sign(self) == 1.0:
                    fig = 0.0
                else:
                    fig = 3.141592653589793
                return Complex(fig, -0.25 * math.log(sqr(self.i + 1.0) / sqr(self.i - 1.0)))

            return Complex(1.5707963267948966, -0.25 * math.log(sqr(self.i + 1.0) / sqr(self.i - 1.0)))

        return Complex(0.5 * (3.141592653589793 - math.atan2(self.r, 1.0 - self.i) + math.atan2(- self.r, self.i + 1.0)), -0.25 * math.log((self.r * self.r + sqr(self.i + 1.0)) / (self.r * self.r + sqr(self.i - 1.0))))


    # 6 arc hyper trig functions
    def asinh(self):
        """
        >>> a = Complex(0.73,0.22)
        >>> print a.asinh()
         0.6865 + 0.1776i
        >>> print a.asinh().sinh()
         0.7300 + 0.2200i
        """
        if self.r == 0.0:
            if self.i < -1.0 or self.i > 1.0:
                return Complex(self.csgn() * math.log(abs(self.i) + math.sqrt(self.i * self.i - 1.0)), 1.5707963267948966 * Complex.sign(self.i))

            return Complex(self.r, math.asin(self.i))

        a = 0.5 * math.sqrt(self.r * self.r + self.i * self.i + 2.0 * self.i + 1.0)
        b = 0.5 * math.sqrt(self.r * self.r + self.i * self.i - 2.0 * self.i + 1.0)
        return Complex(self.csgn() * math.log(a + b + math.sqrt(sqr(a + b) - 1.0)), math.asin(a - b))


    def acosh(self):
        """
        >>> a = Complex(0.73,0.22)
        >>> print a.acosh()
         0.3025 + 0.7984i
        >>> print a.acosh().cosh()
         0.7300 + 0.2200i
        """
        if self.i == 0.0:
            a = 0.5 * abs(self.r + 1.0)
            b = 0.5 * abs(self.r - 1.0)
            if self.r <= -1.0:
                return Complex(Complex(self.i, 1.0 - self.r).csgn() * Complex(self.i, - self.r).csgn() * math.log(abs(self.r) + math.sqrt(self.r * self.r - 1.0)), Complex.sign(self.i) * 3.141592653589793)

            if self.r < 1.0:
                return Complex(0.0, Complex(self.i, 1.0 - self.r).csgn() * math.acos(a - b))

            return Complex(Complex(self.i, 1.0 - self.r).csgn() * Complex(self.i, - self.r).csgn() * math.log(abs(self.r) + math.sqrt(self.r * self.r - 1.0)), Complex(self.i, 1.0 - self.r).csgn() * math.acos(a - b))

        a = 0.5 * math.sqrt(self.r * self.r + 2.0 * self.r + 1.0 + self.i * self.i)
        b = 0.5 * math.sqrt(self.r * self.r - 2.0 * self.r + 1.0 + self.i * self.i)
        return Complex(Complex(self.i, 1.0 - self.r).csgn() * Complex(self.i, - self.r).csgn() * math.log(a + b + math.sqrt(sqr(a + b) - 1.0)), Complex(self.i, 1.0 - self.r).csgn() * math.acos(a - b))


    def atanh(self):
        """
        >>> a = Complex(0.73,0.22)
        >>> print a.atanh()
         0.8054 + 0.4051i
        >>> print a.atanh().tanh()
         0.7300 + 0.2200i
        """
        if self.i == 0.0:
            if self.r == -1.0:
                return Complex(-Infinity, NaN)

            if self.r == 1.0:
                return Complex(Infinity, NaN)

            if self.r < -1.0 or self.r > 1.0:
                return Complex(0.25 * math.log(sqr(self.r + 1.0) / sqr(self.r - 1.0)), Complex.sign(self.i) * 1.5707963267948966)

            return Complex(0.25 * math.log(sqr(self.r + 1.0) / sqr(self.r - 1.0)), 0.5 * (math.atan2(self.i, 1.0 + self.r) - math.atan2(- self.i, 1.0 - self.r)))

        return Complex(0.25 * math.log((self.i * self.i + sqr(self.r + 1.0)) / (self.i * self.i + sqr(self.r - 1.0))), 0.5 * (math.atan2(self.i, 1.0 + self.r) - math.atan2(- self.i, 1.0 - self.r)))


    def asech(self):
        """
        >>> a = Complex(0.73,0.22)
        >>> print a.asech()
         0.8379 -0.4146i
        >>> print a.asech().sech()
         0.7300 + 0.2200i
        """
        if self.i == 0.0:
            c = 0.5 * abs((1.0 + self.r) / self.r)
            d = 0.5 * abs((1.0 - self.r) / self.r)
            if self.r < -1.0 or self.r > 1.0:
                return Complex(0.0, Complex(- self.i, self.r * self.r - self.r).csgn() * math.acos(c - d))

            if self.r == 0.0:
                return Complex(Infinity, NaN)

            if self.r < 0.0:
                return Complex((- Complex(- self.i, - self.r + self.r * self.r).csgn()) * Complex(self.i, self.r).csgn() * math.log(c + d + math.sqrt(sqr(c + d) - 1.0)), (- Complex.sign(self.i)) * 3.141592653589793)

            return Complex((- Complex(- self.i, - self.r + self.r * self.r).csgn()) * Complex(self.i, self.r).csgn() * math.log(c + d + math.sqrt(sqr(c + d) - 1.0)), - self.i)

        a = self.r / self.abs2()
        b = sqr(self.i / self.abs2())
        c = 0.5 * math.sqrt(sqr(a + 1.0) + b)
        d = 0.5 * math.sqrt(sqr(a - 1.0) + b)
        return Complex((- Complex(- self.i, - self.r + self.r * self.r + self.i * self.i).csgn()) * Complex(self.i, self.r).csgn() * math.log(c + d + math.sqrt(sqr(c + d) - 1.0)), Complex(- self.i, self.r * self.r - self.r + self.i * self.i).csgn() * math.acos(c - d))


    def acsch(self):
        """
        >>> a = Complex(0.73,0.22)
        >>> print a.acsch()
         1.0729 -0.2339i
        >>> print a.acsch().csch()
         0.7300 + 0.2200i
        """
        if self.r == 0.0:
            c = 0.5 * abs((1.0 - self.i) / self.i)
            d = 0.5 * abs((1.0 + self.i) / self.i)
            if self.i == 0.0:
                return Complex(Infinity, -Infinity)

            if self.i > -1.0 and self.i < 1.0:
                return Complex(Complex(self.r, - self.i).csgn() * math.log(c + d + math.sqrt(sqr(c + d) - 1.0)), (- Complex.sign(self.i)) * 1.5707963267948966)

            return Complex(Complex.sign(self.r) * 0.0, math.asin(c - d))

        a = sqr(self.r / self.abs2())
        b = self.i / self.abs2()
        c = 0.5 * math.sqrt(sqr(b - 1.0) + a)
        d = 0.5 * math.sqrt(sqr(b + 1.0) + a)
        return Complex(Complex(self.r, - self.i).csgn() * math.log(c + d + math.sqrt(sqr(c + d) - 1.0)), math.asin(c - d))


    def acoth(self):
        """
        >>> a = Complex(0.73,0.22)
        >>> print a.acoth()
         0.8054 -1.1657i
        >>> print a.acoth().coth()
         0.7300 + 0.2200i
        """
        if self.i == 0.0:
            if self.r == -1.0:
                return Complex(-Infinity, NaN)

            if self.r == 1.0:
                return Complex(Infinity, NaN)

            if self.r > -1.0 and self.r < 1.0:
                return Complex(0.25 * math.log(sqr(self.r + 1.0) / sqr(self.r - 1.0)), (- Complex.sign(self.i)) * 1.5707963267948966)

            return Complex(0.25 * math.log(sqr(self.r + 1.0) / sqr(self.r - 1.0)), (- Complex.sign(self.i)) * 0.0)

        return Complex(0.25 * math.log((self.i * self.i + sqr(self.r + 1.0)) / (self.i * self.i + sqr(self.r - 1.0))), 0.5 * (math.atan2(self.i, 1.0 + self.r) - math.atan2(self.i, self.r - 1.0)))


    def ceil(self):
        '''
        >>> bizz = Complex(123.456, -888.888)
        >>> print bizz.ceil()
         124.0000 -888.0000i
        '''
        return Complex(math.ceil(self.r), math.ceil(self.i))


    def floor(self):
        '''
        >>> bizz = Complex(123.456, -888.888)
        >>> print bizz.floor()
         123.0000 -889.0000i
        '''
        return Complex(math.floor(self.r), math.floor(self.i))


    def rint(self):
        '''
        >>> bizz = Complex(123.456, -888.888)
        >>> print bizz.rint()
         123.0000 -889.0000i
        '''
        'Complex round-to-nearest-integer. Pointwise.'
        return Complex(round(self.r), round(self.i))


    @staticmethod
    def sign(flt):
        return math.copysign(1,flt)


    @staticmethod
    def unittest():
        'Run all the unit tests found in comments after the "def" statements'
        import doctest
        doctest.testmod()



## END of class Complex

ZERO  = Complex(0.0, 0.0)
ONE   = Complex(1.0, 0.0)
I     = Complex(0.0, 1.0)
C_NaN = Complex(NaN, NaN)



if __name__ == '__main__':

    # Main line

    Complex.unittest()


