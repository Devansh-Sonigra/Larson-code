# Poisson Equation
This folder contains the implementation of finding solution to model problem with variable coefficient which is describe in chapter 4 of the book. The problem solved is:

\begin{align*}
  -\nabla \cdot ( a \nabla u) u &= f \quad \text{in } \Omega,\\
    -n \cdot (a \nabla u) &= \kappa ( u - g_D) - g_N, \quad \text{on } \partial \Omega.
\end{align*}



where:
- $\Omega$ is the computational domain (typically 2D),
- $f$ is a given source term, $a>0$, $\kappa > 0$, $g_D \text{ and }   g_N$ are given functions.

Steps to run

```bash
cmake .
make
./var_problem
```

After running, the numerical error will be written to a LaTeX file named error.tex.

To compile it to PDF:
```bash
pdflatex error.tex
```
