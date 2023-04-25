import collections

from deepcoder.dsl.function import Function
from deepcoder.dsl.types import INT, BOOL, LIST, FunctionType

# firstorder functions
def head(xs):
    return xs[0] if xs else None

def tail(xs):
    return xs[-1] if xs else None

def minimum(xs):
    return min(xs) if xs else None

def maximum(xs):
    return max(xs) if xs else None

def reverse(xs):
    return xs[::-1]

def _sort(xs):
    return sorted(xs)

def _sum(xs):
    return sum(xs)

def take(n, xs):
    return xs[:n]

def drop(n, xs):
    return xs[n:]

def access(n, xs):
    return xs[n] if n >= 0 and len(xs) > n else None

HEAD = Function('HEAD', head, LIST, INT)
TAIL = Function('TAIL', tail, LIST, INT)
MINIMUM = Function('MINIMUM', minimum, LIST, INT)
MAXIMUM = Function('MAXIMUM', maximum, LIST, INT)
REVERSE = Function('REVERSE', reverse, LIST, LIST)
SORT = Function('SORT', _sort, LIST, LIST)
SUM = Function('SUM', _sum, LIST, INT)

TAKE = Function('TAKE', take, (INT, LIST), LIST)
DROP = Function('DROP', drop, (INT, LIST), LIST)
ACCESS = Function('ACCESS', access, (INT, LIST), INT)


# named functions
def plus1(x):
    return x + 1

def minus1(x):
    return x - 1

def times2(x):
    return x * 2

def div2(x):
    return int(x / 2)

def times_neg1(x):
    return -x

def pow2(x):
    return x ** 2

def times3(x):
    return x * 3

def div3(x):
    return int(x / 3)

def times4(x):
    return x * 4

def div4(x):
    return int(x / 4)

def gt0(x):
    return x > 0

def lt0(x):
    return x < 0

def even(x):
    return x % 2 == 0

def odd(x):
    return x % 2 == 1

def lplus(x, y):
    return x + y

def lminus(x, y):
    return x - y

def ltimes(x, y):
    return x * y

def lmin(x, y):
    return min(x, y)

def lmax(x, y):
    return max(x, y)


# lambda functions
PLUS1 = Function('+1', plus1, INT, INT)
MINUS1 = Function('-1', minus1, INT, INT)
TIMES2 = Function('*2', times2, INT, INT)
DIV2 = Function('/2', div2, INT, INT)
TIMESNEG1 = Function('*-1', times_neg1, INT, INT)
POW2 = Function('**2', pow2, INT, INT)
TIMES3 = Function('*3', times3, INT, INT)
DIV3 = Function('/3', div3, INT, INT)
TIMES4 = Function('*4', times4, INT, INT)
DIV4 = Function('/4', div4, INT, INT)

GT0 = Function('>0', gt0, INT, BOOL)
LT0 = Function('<0', lt0, INT, BOOL)
EVEN = Function('EVEN', even, INT, BOOL)
ODD = Function('ODD', odd, INT, BOOL)

LPLUS = Function('+', lplus, (INT, INT), INT)
LMINUS = Function('-', lminus, (INT, INT), INT)
LTIMES = Function('*', ltimes, (INT, INT), INT)
LMIN = Function('min', lmin, (INT, INT), INT)
LMAX = Function('max', lmax, (INT, INT), INT)

# higher order functions
def _scan1l(f, xs):
    ys = [0] * len(xs)
    for i, x in enumerate(xs):
        if i:
            ys[i] = f(ys[i - 1], x)
        else:
            ys[i] = x
    return ys


def map_function(f, xs):
    return [f(x) for x in xs]

def filter_function(f, xs):
    return [x for x in xs if f(x)]

def count_function(f, xs):
    return len([x for x in xs if f(x)])

def scan1l_function(f, xs):
    ys = [0] * len(xs)
    for i, x in enumerate(xs):
        if i:
            ys[i] = f(ys[i - 1], x)
        else:
            ys[i] = x
    return ys

def zipwith_function(f, xs, ys):
    return [f(x, y) for x, y in zip(xs, ys)]

MAP = Function('MAP', map_function, (FunctionType(INT, INT), LIST), LIST)
FILTER = Function('FILTER', filter_function, (FunctionType(INT, BOOL), LIST), LIST)
COUNT = Function('COUNT', count_function, (FunctionType(INT, BOOL), LIST), INT)
SCAN1L = Function('SCAN1L', scan1l_function, (FunctionType((INT, INT), INT), LIST), LIST)
ZIPWITH = Function('ZIPWITH', zipwith_function, (FunctionType((INT, INT), INT), LIST, LIST), LIST)


LAMBDAS = [
    PLUS1,
    MINUS1,
    TIMES2,
    DIV2,
    TIMESNEG1,
    POW2,
    TIMES3,
    DIV3,
    TIMES4,
    DIV4,

    GT0,
    LT0,
    EVEN,
    ODD,

    LPLUS,
    LMINUS,
    LTIMES,
    LMIN,
    LMAX,
]

FUNCTIONS = [
    HEAD,
    TAIL,
    MINIMUM,
    MAXIMUM,
    REVERSE,
    SORT,
    SUM,
    TAKE,
    DROP,
    ACCESS,

    MAP,
    FILTER,
    COUNT,
    SCAN1L,
    ZIPWITH,
] + LAMBDAS

NAME2FUNC = {x.name: x for x in FUNCTIONS}