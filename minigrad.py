import numpy as np

class Value:
  def __init__(self, data, _children=(), _op='', label=''):
    self.data = data
    self.grad = 0.0
    self._backward = lambda: None
    self._prev = set(_children)
    self._op = _op
    self.label = label
  
  def __repr__(self):
    return f"Value(data={self.data:.2f}, grad={self.grad:.2f}), operation={self._op}"

  def __add__(self, other):
    # So you can add a number to self
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')
    def _backward():
      self.grad += 1.0 * out.grad
      other.grad += 1.0 * out.grad
    out._backward = _backward
    return out

  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')
    
    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    
    out._backward = _backward
    
    return out
  
  def __pow__(self, other):
    assert isinstance(other, (int, float)), "only supporting int/float powers"
    out = Value(self.data**other, (self,), f'^{other}')
    
    def _backward():
      self.grad += other * out.data / self.data * out.grad
    
    out._backward = _backward
    
    return out
  
  def __rmul__(self, other): # other * self
    return self * other
  
  def __truediv__(self, other): # self / other
    return self * other**-1
  
  def __neg__(self): # -self
    return self * -1

  def __sub__(self, other):
    return self + (-other)

  def __radd__(self, other): # other + self
    return self + other
  
  # def __rsub__(self, other): # other - self
  #   return -self + other

  # def __rtruediv__(self, other): # other / self
  #   return other * self**-1

  def tanh(self):
    x = self.data
    t = (np.exp(2*x) - 1) / (np.exp(2*x) + 1)
    out = Value(data=t, _children = (self, ), _op='tanh')
    
    def _backward():
      self.grad += (1-t**2) * out.grad
    
    out._backward = _backward 
    
    return out
  
  def exp(self):
    x = self.data
    out = Value(np.exp(x), (self, ), 'exp')
    def _backward():
      self.grad += out.data * out.grad

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
      # print(f"Backward propagate:", node)
      node._backward()
