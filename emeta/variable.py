# +
import torch


class Variable:

    def __init__(self, *init_args, **init_kwargs):
        self.init_args = init_args
        self.init_kwargs = init_kwargs
        self.dependencies = set()
        self.params = set()
        for arg in (*init_args, *init_kwargs.values()):
            if isinstance(arg, Variable):
                self.dependencies.add(arg)
                arg.dependants.add(self)
                self.params = self.params.union(arg.params)
        self.dependants = set()
        self.value = None

    def eval(self, *eval_args, **eval_kwargs):
        raise RuntimeError('implement in a subclass')

    def __call__(self, *eval_args, **eval_kwargs):
        if self.value is None:
            self.value = self.eval(*eval_args, **eval_kwargs)
        return self.value

    def _forward(self):
        self.value = None
        for dep in self.dependants:
            dep._forward()

    def _backward(self):
        self.value = None
        for dep in self.dependencies:
            dep._backward()

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
        args = argstr(*self.init_args, **self.init_kwargs)
        return f'{self.__class__.__name__}({args})'

    def __getattr__(self, attr):
        return GetAttr(self, attr)


def asstr(arg):
    if type(arg) == str:
        return f'"{arg}"'
    else:
        return f'{arg}'


def argstr(*args, **kwargs):
    ar = (*(asstr(arg) for arg in args),
          *(f'{a}={asstr(b)}' for a, b in kwargs.items()))
    return ', '.join(ar)


def eval(var, *eval_args, **eval_kwargs):
    if isinstance(var, Variable):
        return var(*eval_args, **eval_kwargs)
    else:
        return var


class GetAttr:

    def __init__(self, _self, attr):
        self.self = _self
        self.attr = attr

    def __call__(self, *args, **kwargs):
        return Attr(self.self, self.attr, *args, **kwargs)


class Attr(Variable):

    def __init__(self, var, attr, *args, **kwargs):
        super().__init__(var, attr, *args, **kwargs)
        self.var = var
        self.attr = attr
        self.args = args
        self.kwargs = kwargs

    def eval(self, *eval_args, **eval_kwargs):
        val = eval(self.var, *eval_args, **eval_kwargs)
        return getattr(val, self.attr)(*self.args, **self.kwargs)

    def __repr__(self):
        args = argstr(*self.args, **self.kwargs)
        return f'{self.var}.{self.attr}({args})'


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
        return f'({self.symbol.join([str(arg) for arg in self.args])})'

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
        return f'(-{self.arg})'

    def eval(self, *eval_args, **eval_kwargs):
        return -eval(self.arg, *eval_args, **eval_kwargs)


def Param(name):
    try:
        return Par.instances[name]
    except KeyError:
        return Par(name)


class Par(Variable):

    instances = {}

    def __init__(self, name):
        assert name not in Par.instances
        super().__init__(name)
        self.name = name
        Par.instances[name] = self
        self.params.add(self)

    def eval(self):
        return self.data

    def __repr__(self):
        return f'Param("{self.name}")'

    def set(self, data, rg=True):
        self.data = torch.as_tensor(data)
        self.data.requires_grad = rg
        self._forward()

    def add(self, data):
        self.data.data += data
        self.data.grad = None
        self._forward()

    @property
    def force(self):
        f = self.data.grad
        if f is not None:
            return -f

    @property
    def dot(self):
        return self.dot_data

    def dot_(self, data):
        self.dot_data = data

    def _dot(self, data):
        self.dot_data += data
