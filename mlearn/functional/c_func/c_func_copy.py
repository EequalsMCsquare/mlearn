# This file was automatically generated by SWIG (http://www.swig.org).
# Version 4.0.1
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info
if _swig_python_version_info < (2, 7, 0):
    raise RuntimeError("Python 2.7 or later required")

# Import the low-level C/C++ module
if __package__ or "." in __name__:
    from . import _c_func
else:
    import _c_func

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)


def _swig_setattr_nondynamic_instance_variable(set):
    def set_instance_attr(self, name, value):
        if name == "thisown":
            self.this.own(value)
        elif name == "this":
            set(self, name, value)
        elif hasattr(self, name) and isinstance(getattr(type(self), name), property):
            set(self, name, value)
        else:
            raise AttributeError("You cannot add instance attributes to %s" % self)
    return set_instance_attr


def _swig_setattr_nondynamic_class_variable(set):
    def set_class_attr(cls, name, value):
        if hasattr(cls, name) and not isinstance(getattr(cls, name), property):
            set(cls, name, value)
        else:
            raise AttributeError("You cannot add class attributes to %s" % cls)
    return set_class_attr


def _swig_add_metaclass(metaclass):
    """Class decorator for adding a metaclass to a SWIG wrapped class - a slimmed down version of six.add_metaclass"""
    def wrapper(cls):
        return metaclass(cls.__name__, cls.__bases__, cls.__dict__.copy())
    return wrapper


class _SwigNonDynamicMeta(type):
    """Meta class to enforce nondynamic attributes (no new attributes) for a class"""
    __setattr__ = _swig_setattr_nondynamic_class_variable(type.__setattr__)


def new_doublep():
    return _c_func.new_doublep()

def copy_doublep(value):
    return _c_func.copy_doublep(value)

def delete_doublep(obj):
    return _c_func.delete_doublep(obj)

def doublep_assign(obj, value):
    return _c_func.doublep_assign(obj, value)

def doublep_value(obj):
    return _c_func.doublep_value(obj)

import numpy as np
from ctypes import c_double

def sample_conv2d(inputs, weights, bias):
    _shape = (weights.shape[1], inputs.shape[0], inputs.shape[1])
    ptr = _c_func.sample_conv2d(inputs, weights, bias)
    out = (c_double * (inputs.shape[1]*inputs.shape[0]*weights.shape[1])).from_address(int(ptr))
    arr = np.ctypeslib.as_array(out).reshape(*_shape)
    return arr

def matmulAdd(inputs, w, b):
    _shape = (inputs.shape[0], w.shape[-1])
    ptr = _c_func.matmulAdd(inputs, w.T, b)
    out = (c_double * (inputs.shape[0]*w.shape[-1])).from_address(int(ptr))
    arr = np.ctypeslib.as_array(out).reshape(*_shape)
    return arr
