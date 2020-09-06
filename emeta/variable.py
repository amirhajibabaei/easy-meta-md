# +
class Variable:

    def eval(self, *eval_args, **eval_kwargs):
        raise RuntimeError('implement in a subclass')

    def _eval(self, var, *eval_args, **eval_kwargs):
        if issubclass(var.__class__, Variable):
            return var(*eval_args, **eval_kwargs)
        else:
            return var

    def __call__(self, *eval_args, **eval_kwargs):
        return self.eval(*eval_args, **eval_kwargs)

    def __add__(self, other):
        return binary_op(self, other, Sum)

    def __radd__(self, other):
        return binary_op(other, self, Sum)

    def __sub__(self, other):
        return binary_op(self, other, Sub)

    def __rsub__(self, other):
        return binary_op(other, self, Sub)

    def __mul__(self, other):
        return binary_op(self, other, Mul)

    def __rmul__(self, other):
        return binary_op(other, self, Mul)

    def __truediv__(self, other):
        return binary_op(self, other, Div)

    def __rtruediv__(self, other):
        return binary_op(other, self, Div)

    def __pow__(self, other):
        return binary_op(self, other, Pow)

    def __neg__(self):
        return Neg(self)

    def __repr__(self):
        try:
            args = ', '.join((str(arg) for arg in self.args))
            try:
                if len(self.kwargs):
                    kwargs = ', '.join(
                        (f'{a}={b}' for a, b in self.kwargs.items()))
                    args = ', '.join((args, kwargs))
            except:
                pass
        except:
            raise RuntimeError('define args [and kwargs] or __repr__')
        return f'{self.__class__.__name__}({args})'


def binary_op(self, other, Cls):
    s = self.__class__ == Cls
    o = other.__class__ == Cls
    if s and o:
        return Cls(*self.args, *other.args)
    elif s and not o:
        return Cls(*self.args, other)
    elif not s and o:
        return Cls(self, *other.args)
    elif not s and not o:
        return Cls(self, other)


class Binary(Variable):

    def __init__(self, *args):
        self.args = args
        self.symbol = None

    def __repr__(self):
        return self.symbol.join([str(arg) for arg in self.args])

    def eval(self, *eval_args, **eval_kwargs):
        result = self._eval(self.args[0], *eval_args, **eval_kwargs)
        for arg in self.args[1:]:
            a = self._eval(arg, *eval_args, **eval_kwargs)
            result = self.op(result, a)
        return result


class Sum(Binary):

    def __init__(self, *args):
        super().__init__(*args)
        self.symbol = ' + '

    def op(self, a, b):
        return a + b


class Sub(Binary):

    def __init__(self, *args):
        super().__init__(*args)
        self.symbol = ' - '

    def op(self, a, b):
        return a - b


class Mul(Binary):

    def __init__(self, *args):
        super().__init__(*args)
        self.symbol = '*'

    def op(self, a, b):
        return a*b


class Div(Binary):

    def __init__(self, *args):
        super().__init__(*args)
        self.symbol = '/'

    def op(self, a, b):
        return a/b


class Pow(Binary):

    def __init__(self, *args):
        super().__init__(*args)
        self.symbol = '**'

    def op(self, a, b):
        return a**b


class Neg(Variable):

    def __init__(self, arg):
        self.arg = arg

    def __repr__(self):
        return f'-{self.arg}'

    def eval(self, *eval_args, **eval_kwargs):
        return -self._eval(self.arg, *eval_args, **eval_kwargs)


class Ext(Variable):

    def __init__(self, *args, **kwargs):
        """first arg should be a function"""
        self._func = args[0]
        self._args = args[1:]
        self.args = (self._func.__name__, *self._args)
        self.kwargs = kwargs

    def eval(self, *eval_args, **eval_kwargs):
        args = (self._eval(arg, *eval_args, **eval_kwargs)
                for arg in self._args)
        return self._func(*args, **self.kwargs)
