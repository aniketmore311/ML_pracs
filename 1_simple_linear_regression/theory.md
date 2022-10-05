## linear regression with gradient descent

### Theory

the relationship between dependant and independant variables is given by

$\hat{y} = \theta_0 + \theta_1 * x_1 + \theta_2 * x_2 + ... + \theta_n * x_n$  

where 

* $\hat{y}$ is the predicted value
* $n$ is the number of features
* $x_i$ is the ith feature value
* $\theta_j$ is the jth model parameter
* $\theta_0$ is the bias weight while $\theta_1$ to $\theta_n$ are feature weights

in simple linear regression we have only one independant variable x1

#### vectorized / linear algebra form

$\hat{y} = \theta^{T}.x$

where

* $\theta$ is the model parameter vector containing bias term $\theta_0$ and feature weights $\theta_1$ to $\theta_n$
* $\theta^{T}$ is the transpose of parameter vector (row vector instead of column vector)
* $x$ is that instance's feature vector containing $x_0$ to $x_n$ where $x_0$ is always 

#### MSE cost function

$MSE(X) = \frac{1}{m} * \sum_{i=1}^{m}{(\theta^T.x^{(i)} - y^{(i)})}$

where

* $X$ is the entire dataset
* $m$ is the number if instances in the dataset
* $x^{(i)}$ is the feature vector of ith instance in the dataset
* $y^{(i)}$ is the desired output for that instance

#### Normal equation / analytical method

$\hat{\theta} = (X^T.X)^{-1}.X^T.y$

where

* $\hat{\theta}$ is the value of $\theta$ minimizing the cost function

#### gradient descent

for gradient descent we have to calculate partial derivate of cost function wrt each model parameter

$\frac{\partial{MSE(\theta)}}{\partial{\theta_j}} = \frac{2}{m}\sum_{i=1}^{m}{(\theta^T.x^{(i)}-y^{(i)})}x^{(i)}_{j}$

we can calculate gradients of all model parameters wrt cost function in one computation using the equation

$\nabla MSE(\theta) = \frac{2}{m}X^{T}.(X.\theta - y) $

where

* $\nabla MSE(\theta)$ is the gradient vector containing gradients of all model parameters wrt cost function

to calculate next value of parameter vectors

$\theta^{(next step)} = \theta - \eta\nabla MSE(\theta)$ 

where

* $\eta$ is the learning rate