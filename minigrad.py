import numpy as np

class Value:
  def __init__(self, data, _children=(), _op='', label=''):
    self.data = data
    self.grad = 0
    self._backward = lambda: None
    self._prev = set(_children)
    self._op = _op
    self.label = label
  
  def __repr__(self):
    return f"Value(data={self.data})"

  def __add__(self, other):
    # So you can add a number to self
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')
    def _backward():
      self.grad += out.grad
      other.grad += out.grad
    self._backward = _backward    
    return out

  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')
    
    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    
    self._backward = _backward
    
    return out

  def tanh(self):
    x = self.data
    t = (np.exp(2*x) - 1) / (np.exp(2*x) + 1)
    out = Value(data=t, _children = (self, ), _op='tanh')
    
    def _backward():
      self.grad += (1-t**2) * out.grad
    
    out._backward = _backward 
    
    return out

  def backward(self):
    # Do the backpropagation in reversed order
    topo = []
    visited = set()

    def build_topo(v):
      if v not in visited:
        visited.add(v)

        # Append you to the topo list only after all your children are appended
        for child in v._prev:
          build_topo(child)
        topo.append(v)

    build_topo(self)

    self.grad = 1.0
    for node in reversed(topo):
      node._backward()
