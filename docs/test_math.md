# Math Rendering Test

This page tests LaTeX math rendering with MathJax.

## Inline Math

Here's Einstein's famous equation: $E = mc^2$

The quadratic formula: $x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$

Greek letters: $\alpha, \beta, \gamma, \Delta, \Sigma, \Omega$

## Display Math

### Basic Equation

$$
f(x) = \int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}
$$

### Matrix Example

$$
\begin{bmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33}
\end{bmatrix}
$$

### Multi-line Alignment

$$
\begin{align}
\nabla \cdot \mathbf{E} &= \frac{\rho}{\epsilon_0} \\
\nabla \cdot \mathbf{B} &= 0 \\
\nabla \times \mathbf{E} &= -\frac{\partial \mathbf{B}}{\partial t} \\
\nabla \times \mathbf{B} &= \mu_0 \mathbf{J} + \mu_0 \epsilon_0 \frac{\partial \mathbf{E}}{\partial t}
\end{align}
$$

### SpliceAI Example

The SpliceAI loss function:

$$
\mathcal{L} = -\sum_{i=1}^{L} \sum_{c \in \{A, D, N\}} y_{i,c} \log(\hat{y}_{i,c})
$$

where:
- $L$ is the sequence length
- $c$ represents splice site categories (Acceptor, Donor, Neither)
- $y_{i,c}$ is the true label
- $\hat{y}_{i,c}$ is the predicted probability

### Probability Distribution

$$
P(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}
$$

## Advanced Examples

### Summation and Product

$$
\sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{\pi^2}{6} \quad \text{and} \quad \prod_{p \text{ prime}} \frac{1}{1-p^{-s}} = \zeta(s)
$$

### Conditional Expression

$$
f(n) = \begin{cases}
n/2 & \text{if } n \text{ is even} \\
3n+1 & \text{if } n \text{ is odd}
\end{cases}
$$

### Fraction and Binomial

$$
\binom{n}{k} = \frac{n!}{k!(n-k)!} \quad \text{for } 0 \leq k \leq n
$$

---

If all equations above render properly (as typeset math, not raw LaTeX), then MathJax is working correctly! 🎉
