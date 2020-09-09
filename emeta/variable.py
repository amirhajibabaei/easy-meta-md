# +
class Variable:

    def __init__(self, *init_args, **init_kwargs):
        self.init_args = init_args
        self.init_kwargs = init_kwargs

    def eval(self, *eval_args, **eval_kwargs):
        raise RuntimeError('implement in a subclass')

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

    def __mod__(self, other):
        return binary_op(self, other, Mod)

    def __rmod__(self, other):
        return binary_op(other, self, Mod)

    def __pow__(self, other):
        return binary_op(self, other, Pow)

    def __neg__(self):
        return Neg(self)

    def __repr__(self):
        try:
            args = ', '.join((str(arg) for arg in self.init_args))
            try:
                if len(self.init_kwargs):
                    kwargs = ', '.join(
                        (f'{a}={b}' for a, b in self.init_kwargs.items()))
                    args = ', '.join((args, kwargs))
            except:
                pass
        except:
            raise RuntimeError('whaaat?!')
        return f'{self.__class__.__name__}({args})'

    def __getattr__(self, attr):
        return LazyGen(self, attr)


def eval(var, *eval_args, **eval_kwargs):
    if var is None:
        return None
    elif issubclass(var.__class__, Variable):
        return var(*eval_args, **eval_kwargs)
    else:
        return var


class LazyGen:

    def __init__(self, _self, attr):
        self.self = _self
        self.attr = attr

    def __call__(self, *args, **kwargs):
        return Lazy(self.self, self.attr, *args, **kwargs)


class Lazy(Variable):

    def __init__(self, var, attr, *args, **kwargs):
        super().__init__(var, attr, *args, **kwargs)
        self.var = var
        self.attr = attr
        self.args = args
        self.kwargs = kwargs

    def eval(self, *eval_args, **eval_kwargs):
        val = eval(self.var, *eval_args, **eval_kwargs)
        return getattr(val, self.attr)(*self.args, **self.kwargs)


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
        super().__init__(*args)
        self.args = args
        self.symbol = None

    def __repr__(self):
        return self.symbol.join([str(arg) for arg in self.args])

    def eval(self, *eval_args, **eval_kwargs):
        result = eval(self.args[0], *eval_args, **eval_kwargs)
        for arg in self.args[1:]:
            a = eval(arg, *eval_args, **eval_kwargs)
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


class Mod(Binary):

    def __init__(self, *args):
        super().__init__(*args)
        self.symbol = '%'

    def op(self, a, b):
        return a % b


class Pow(Binary):

    def __init__(self, *args):
        super().__init__(*args)
        self.symbol = '**'

    def op(self, a, b):
        return a**b


class Neg(Variable):

    def __init__(self, arg):
        super().__init__(arg)
        self.arg = arg

    def __repr__(self):
        return f'-{self.arg}'

    def eval(self, *eval_args, **eval_kwargs):
        return -eval(self.arg, *eval_args, **eval_kwargs)


class Ext(Variable):

    def __init__(self, func, *args, **kwargs):
        super().__init__(self, func, *args, **kwargs)
        """first arg should be a function"""
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def eval(self, *eval_args, **eval_kwargs):
        args = (eval(arg, *eval_args, **eval_kwargs)
                for arg in self.args)
        return self.func(*args, **self.kwargs)
