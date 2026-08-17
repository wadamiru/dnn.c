# Learning Representations by Back-Propagating Errors

**Original Paper:** Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). *Nature*, 323(6088), 533–536.

---

## 1. Overview

This seminal paper introduces the **Back-Propagation algorithm** for training multi-layer neural networks (multilayer perceptrons). Prior to this work, single-layer networks (like Perceptrons) could not learn non-linear decision boundaries (e.g., XOR) because they lacked internal representations (hidden units). Back-propagation solves this by computing gradients of an error function with respect to internal weights using the **chain rule**.

---

## 2. Network Architecture

Consider a feedforward network with:
- Input units $i$
- Hidden units $j$
- Output units $k$

### Forward Pass

For any unit $j$, the total input $x_j$ is a weighted sum of outputs $y_i$ from the preceding layer plus a bias $\theta_j$:

$$x_j = \sum_{i} w_{ji} y_i + \theta_j$$

The output $y_j$ is obtained by applying a smooth, differentiable activation function $f(x_j)$, typically the sigmoid:

$$y_j = f(x_j) = \frac{1}{1 + e^{-x_j}}$$

---

## 3. Loss Function

For a given training pattern $p$, the error measure $E_p$ is the squared error over output units $k$:

$$E_p = \frac{1}{2} \sum_{k} (t_{pk} - y_{pk})^2$$

where $t_{pk}$ is the target output and $y_{pk}$ is the actual output. The total error is $E = \sum_p E_p$.

---

## 4. Back-Propagation Math (Gradient Derivation)

To update weight $w_{ji}$ via gradient descent, we need $\frac{\partial E_p}{\partial w_{ji}}$. By the chain rule:

$$\frac{\partial E_p}{\partial w_{ji}} = \frac{\partial E_p}{\partial x_j} \cdot \frac{\partial x_j}{\partial w_{ji}}$$

Since $x_j = \sum_i w_{ji} y_i + \theta_j$, we have:

$$\frac{\partial x_j}{\partial w_{ji}} = y_i$$

Define the error signal (delta) for unit $j$ as:

$$\delta_j = -\frac{\partial E_p}{\partial x_j}$$

Thus, the gradient becomes:

$$-\frac{\partial E_p}{\partial w_{ji}} = \delta_j y_i$$

---

### Case 1: Output Layer Units ($k$)

Using the chain rule:

$$\delta_k = -\frac{\partial E_p}{\partial x_k} = -\frac{\partial E_p}{\partial y_k} \cdot \frac{\partial y_k}{\partial x_k}$$

1. Derivative of error wrt output $y_k$:
   $$\frac{\partial E_p}{\partial y_k} = -(t_k - y_k)$$

2. Derivative of sigmoid output wrt input $x_k$:
   $$\frac{\partial y_k}{\partial x_k} = f'(x_k) = y_k (1 - y_k)$$

Combining these:

$$\delta_k = (t_k - y_k) y_k (1 - y_k)$$

---

### Case 2: Hidden Layer Units ($j$)

For a hidden unit $j$, its output affects $E_p$ through all units $k$ in the subsequent layer:

$$\delta_j = -\frac{\partial E_p}{\partial x_j} = -\frac{\partial E_p}{\partial y_j} \cdot \frac{\partial y_j}{\partial x_j} = \left( \sum_{k} -\frac{\partial E_p}{\partial x_k} \frac{\partial x_k}{\partial y_j} \right) f'(x_j)$$

Since $x_k = \sum_j w_{kj} y_j + \theta_k$, we have $\frac{\partial x_k}{\partial y_j} = w_{kj}$. Substituting $\delta_k = -\frac{\partial E_p}{\partial x_k}$:

$$\delta_j = f'(x_j) \sum_{k} \delta_k w_{kj} = y_j (1 - y_j) \sum_{k} \delta_k w_{kj}$$

### Final Gradients
**Weight Gradient:**
$$\frac{\partial E_p}{\partial w_{ji}} = \delta_j \, y_i$$

**Bias Gradient:**
$$\frac{\partial E_p}{\partial \theta_j} = \delta_j$$

---

## 5. Weight & Bias Update Rules

Using gradient descent, weights are updated proportional to the negative gradient:

$$\Delta w_{ji}(t+1) = \epsilon \, \delta_j y_i + \alpha \, \Delta w_{ji}(t)$$

where:
- $\epsilon$ is the learning rate.
- $\alpha$ is a momentum term ($0 \le \alpha < 1$) added to smooth out oscillations and accelerate convergence across flat regions.

Similarly, biases are updated as:

$$\Delta \theta_j(t+1) = \epsilon \, \delta_j + \alpha \, \Delta \theta_j(t)$$

---

## 6. Summary of Algorithm

1. **Initialize:** Set weights and biases to small random values.
2. **Forward Pass:** Compute inputs $x_j$ and outputs $y_j$ layer by layer up to the output layer.
3. **Compute Output Errors:** Calculate $\delta_k = (t_k - y_k) y_k (1 - y_k)$ for all output units.
4. **Backward Pass:** Propagate error backward to calculate $\delta_j = y_j (1 - y_j) \sum_k \delta_k w_{kj}$ for hidden units.
5. **Update Weights:** Adjust weights $w_{ji} \leftarrow w_{ji} + \Delta w_{ji}$ and biases $\theta_j \leftarrow \theta_j + \Delta \theta_j$.
6. **Iterate:** Repeat until global error $E$ converges.
