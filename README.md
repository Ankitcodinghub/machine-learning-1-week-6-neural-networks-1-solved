# machine-learning-1-week-6-neural-networks-1-solved
**TO GET THIS SOLUTION VISIT:** [Machine Learning 1 Week 6-Neural Networks 1 Solved](https://www.ankitcodinghub.com/product/machine-learning-1-week-6-neural-networks-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;98757&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;1&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (1 vote)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;Machine Learning 1 Week 6-Neural Networks 1 Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (1 vote)    </div>
    </div>
<div class="page" title="Page 1">
<div class="layoutArea">
<div class="column">
Exercise 1: Designing a Neural Network

We would like to implement a neural network that classifies data points in R2 according to decision boundary given in the figure below.

class B

We consider as an elementary computation the threshold neuron whose relation between inputs (ai)i and output aj is given by

zj =􏰌iaiwij +bj aj =1zj&gt;0.

(a) Design at hand a neural network that takes x1 and x2 as input and produces the output “1” if the input belongs to class A, and “0” if the input belongs to class B. Draw the neural network model and write down the weights wij and bias bj of each neuron.

Exercise 2: Backward Propagation (5 + 20 P)

We consider a neural network that takes two inputs x1 and x2 and produces an output y based on the following set of computations:

</div>
</div>
<table>
<tbody>
<tr>
<td></td>
<td>
<div class="layoutArea">
<div class="column">
class A

</div>
</div>
</td>
</tr>
<tr>
<td></td>
<td></td>
</tr>
</tbody>
</table>
<div class="layoutArea">
<div class="column">
z3 =x1·w13+x2·w23 a3 = tanh(z3)

z4 =x1 ·w14 +x2 ·w24 a4 = tanh(z4)

</div>
<div class="column">
z5 =a3·w35+a4·w45 a5 = tanh(z5)

z6 =a3 ·w36 +a4 ·w46 a6 = tanh(z6)

</div>
<div class="column">
y=a5+a6

</div>
</div>
<div class="layoutArea">
<div class="column">
<ol>
<li>(a) &nbsp;Draw the neural network graph associated to this set of computations.</li>
<li>(b) &nbsp;Write the set of backward computations that leads to the evaluation of the partial derivative ∂y/∂w13. Youranswer should avoid redundant computations. Hint: tanh′(t) = 1 − (tanh(t))2.Exercise 3: Programming (50 P)
Download the programming files on ISIS and follow the instructions.
</li>
</ol>
</div>
</div>
</div>
<div class="page" title="Page 2">
<div class="section">
<div class="layoutArea">
<div class="column">
Exercise sheet 6 (programming) [WiSe 2021/22] Machine Learning 1

</div>
</div>
<div class="layoutArea">
<div class="column">
Training a Neural Network

In this homework, our objective is to implement a simple neural network from scratch, in particular, error backpropagation and the gradient descent optimization procedure. We first import some useful libraries.

In [1]:

import numpy

import matplotlib

%matplotlib inline

from matplotlib import pyplot as plt na = numpy.newaxis numpy.random.seed(0)

We consider a two-dimensional moon dataset on which to train the network. We also create a grid dataset which we will use to visualize the decision functionintwodimensions.Wedenoteourtwoinputsasx1andx2andusethesuffix d and g todesignatetheactualdatasetandthegriddataset.

In [2]:

<pre># Create a moon dataset on which to train the neural network
</pre>
import sklearn,sklearn.datasets

Xd,Td = sklearn.datasets.make_moons(n_samples=100) Xd = Xd*2-1

Td = Td * 2 – 1

X1d = Xd[:,0]

X2d = Xd[:,1]

<pre># Creates a grid dataset on which to inspect the decision function
</pre>
<pre>l = numpy.linspace(-4,4,100)
X1g,X2g = numpy.meshgrid(l,l)
</pre>
The moon dataset is plotted below along with some dummy decision function x1 + x2 = 0. In [3]:

def plot(Yg,title=None):

plt.figure(figsize=(3,3))

plt.scatter(*Xd[Td==-1].T,color=’#0088FF’) plt.scatter(*Xd[Td==1].T,color=’#FF8800′) plt.contour(X1g,X2g,Yg,levels=[0],colors=’black’,linestyles=’dashed’) plt.contourf(X1g,X2g,Yg,levels=[-100,0,100],colors=[‘#0088FF’,’#FF8800′],alpha=0.1) if title is not None: plt.title(title)

plt.show()

plot(X1g+X2g) # plot the dummy decision function

Part 1: Implementing Error Backpropagation (30 P)

∀25 :z =xw +xw +b j=1j 11j22jj

</div>
</div>
<div class="layoutArea">
<div class="column">
∀25 : a = max We would like to implement the neural network with the equations: j = 1 j

</div>
</div>
<div class="layoutArea">
<div class="column">
where x_1,x_2 are the two input variables and y is the output of the network. The parameters of the neural network are initialized randomly using the normal distributions w_{ij} \sim \mathcal{N}

</div>
</div>
<div class="layoutArea">
<div class="column">
(\mu=0,\sigma^2=1/2), b_{j} \sim \mathcal{N}(\mu=0,\sigma^2=1), v_{j} \sim \mathcal{N}(\mu=0,\sigma^2=1/25). The following code initializes the parameters of the network and implements the forward pass defined above. The neural network is composed of 50 neurons.

</div>
</div>
</div>
</div>
<div class="page" title="Page 3">
<div class="section">
<div class="layoutArea">
<div class="column">
In [4]:

import numpy

NH = 50

<pre>W = numpy.random.normal(0,1/2.0**.5,[2,NH])
B = numpy.random.normal(0,1,[NH])
V = numpy.random.normal(0,1/NH**.5,[NH])
</pre>
def forward(X1,X2):

X = numpy.array([X1.flatten(),X2.flatten()]).T # Convert meshgrid into dataset Z = X.dot(W)+B

A = numpy.maximum(0,Z)

Y = A.dot(V)

return Y.reshape(X1.shape) # Reshape output into meshgrid

We now consider the task of training the neural network to classify the data. For this, we define the error function: \mathcal{E}(\theta) = \sum_{k=1}^N \max(0,-y^{(k)} t^{(k)}) where N is the number of data points, y is the output of the network and t is the label.

Task:

Complete the function below so that it returns the gradient of the error w.r.t. the parameters of the model.

In [5]:

def backprop(X1,X2,T):

X = numpy.array([X1.flatten(),X2.flatten()]).T

<pre>    # Compute activations
</pre>
<pre>    Z = X.dot(W)+B
    A = numpy.maximum(0,Z)
    Y = A.dot(V)
</pre>
<pre>    # Compute backward pass
</pre>
<pre>    DY = (-Y*T&gt;0)*(-T)
    DZ = numpy.outer(DY,V)*(Z&gt;0)
</pre>
<pre>    # Compute parameter gradients (averaged over the whole dataset)
</pre>
# ———————————– # TODO: replace by your code

# ———————————– import solution

<pre>    DW,DB,DV = solution.gradients(X,Z,A,Y,DY,DZ)
</pre>
<pre>    # -----------------------------------
</pre>
return DW,DB,DV

Exercise 2: Training with Gradient Descent (20 P)

We would like to use error backpropagation to optimize the parameters of the neural network. The code below optimizes the network for 128 iterations and at some chosen iterations plots the decision function along with the current error.

Task:

Complete the procedure above to perform at each iteration a step along the gradient in the parameter space. A good choice of learning rate is \eta=0.1.

In [6]:

for i in range(128):

if i in [0,1,3,7,15,31,63,127]:

Yg = forward(X1g,X2g)

Yd = forward(X1d,X2d)

Ed = numpy.maximum(0,-Yd*Td).mean() plot(Yg,title=”It: %d, Error: %.3f”%(i,Ed))

# ———————————– # TODO: replace by your code

# ———————————– import solution

<pre>        W,B,V = solution.descent(X1d,X2d,Td,W,B,V,backprop)
</pre>
<pre>        # -----------------------------------
</pre>
</div>
</div>
</div>
</div>
<div class="page" title="Page 4"></div>
<div class="page" title="Page 5"></div>
<div class="page" title="Page 6"></div>
<div class="page" title="Page 7"></div>
