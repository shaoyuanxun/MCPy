# McCormick_Relaxation_Library_with_Subgradient
#### Author: Yuanxun Shao, Jingjing Wang

### 1. Introduction
**MCPy** is a python library for automatically calculate convex\concave relaxations and subgradients of factorable nonconvex functions according to McCormick relaxation rules and interval arithmetic. The library could be quite useful for prototyping and testing new global optimization algorithms.

Theory and implementation for the global optimization of a wide class of algorithms is presented via convex/affine relaxations [1]. Similar to the convex\concave relaxation, the subgradient propagation relies on the recursive application of specific rules, which could provide us affine relaxations of McCormick relaxations. This library automatedly implements those theorems based on normal and reverse operator overloading.

### 2. Instruction
Supported rules so far: +, -, /, \*, sqrt, log, exp, \*\*(restricted to integer powers).
<br />
Upcoming rules: sin, cos.

There are three instance variables in the class, **MCPy.IA**, **MCPy.MC**, **MCPy.SG**.
<br />
**MCPy.IA** 
<br />1-D numpy array of two elements [LB, UB]. <br />LB/UB are the lower/upper bound the function calculated by the intervarl arithmetic.
<br />
**MCPy.MC**
<br />1-D numpy array of two elements [cv, cc]. <br />cv/cc are the convex underestimator/concave overestimator of the function calculated by the McCormick rules.
<br />
**MCPy.SG**
<br /> 2-D numpy n-by-2 matrix [**SG_cv**, **SG_cc**]. <br /> **SG_cv**/**SG_cc** are n-by-1 column verctors of subgradients for convex/concave relaxations.

### 3. Example and Illustration
(a) Import the MCPy class
<br />
(b) Initialize the variables
<br />
(c) Do the calculation
<br />
See [examples](https://github.com/shaoyuanxun/McCormick_Relaxation_Library_with_Subgradient/blob/master/examples.ipynb) file for more details, including the usage, plots, and animations.

### 4. Convergence [2]
See [convergence](https://github.com/shaoyuanxun/McCormick_Relaxation_Library_with_Subgradient/blob/master/convergence.ipynb) file.

### References:
[1] Mitsos, A., B. Chachuat, P.I. Barton, [McCormick-based relaxations of algorithms](http://epubs.siam.org/doi/abs/10.1137/080717341), SIAM Journal on Optimization, 20(2):573-601, 2009
<br />
[2] Bompadre, A., A. Mitsos, [Convergence rate of McCormick relaxations](https://link.springer.com/article/10.1007%2Fs10898-011-9685-2), Journal of Global Optimization 52(1):1-28, 2012
<br />
[3] Y. Shao and J. K. Scott. [Convex relaxations for global optimization under uncertainty described by continuous random variables](https://arxiv.org/abs/1709.08780), arXiv preprint. 2017.

```
Citation of our recent work:
@Article{Shao:2017,
  author = {Shao, Y. and Scott, J. K.},
  title  = {Convex Relaxations for Global Optimization Under Uncertainty Described by Continuous Random Variables (preprint: arXiv:1709.08780)},
  year   = {2017},
  url    = {arXiv:1709.08780},
}
```
