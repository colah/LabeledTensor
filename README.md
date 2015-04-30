
I love NumPy. I love Theano. But keeping axes straight in my head can be a real pain.

Is X my first axis, or my second? What in the wide world is the fourth axis? Which axes align when multiplying n-dimensional arrays? What about when I convolve them? How does dimension broadcasting work again? ...

I don't want to deal with that nonsense and you probably don't either.

I think there's an alternative framework for thinking about tensors that solves this problems: labeled axes.

Instead of having a list of axes, we have a set of axes with human labels. Instead of remembering that X is the first axis, Y is the second, C the third, and N the fourth... Well, you don't have to remember everything, because axes just have human labels!

Before we go on, a quick example to make sure we're comfortable with this:

```python
>>> X = Axis("X", 4)
>>> Y = Axis("Y", 3)
>>> 
>>> arr = zeros(X, Y)
>>>
>>> arr.axes
set([Y~3, X~4])
>>> 
>>> arr[Y: 1] = 5
>>> arr[X: 3] = 7
>>>
>>>
>>> 
>>> arr

array([[ 0,  0,  0,  7],
       [ 5,  5,  5,  7],
       [ 0,  0,  0,  7]])
```

Broadcasting is dirt simple. When we do pointwise operations, we broadcast on dimensions that are not shared. The axes of the output are the union of the axes of the input.

Let's consider a few examples:

```python
>>> a = zeros(X)
>>> a[X: 2] = 4
>>> a
array([ 0,  0,  4,  0])

>>> b = zeros(Y)
>>> b[Y: 1] = 3
>>> b
array([ 0,  3,  0])

>>> a+a
array([ 0,  0,  8,  0])

>>> b*b
array([ 0,  9,  0])

>>> a+b
array([[ 0,  0,  4,  0],
       [ 3,  3,  7,  3],
       [ 0,  0,  4,  0]])

>>> arr*b 
array([[ 0,  0,  0,  7],
       [15, 15, 15, 21],
       [ 0,  0,  0,  7]])

```





```python

class tanh_layer:
    def __init__(X, H):
        self.W  = 0.1*randn(X,H)
        self.gW = zeros(X,H)
        self.b  = zeros(H)
        self.gb = zeros(H)

    def feedforward(x):
        self.x = x
        self.y = tanh(dot(self.W, x) + self.b)
        return 

    def backprop(gy):
        self.gb = gy*(1-self.y**2)
        self.gW += self.x*self.gb
        gx = 
        

```
