# Poisson Equation
This folder contains the implementation of eigenvalue problem which is describe in chapter 4 of the book. The problem solved is:

\begin{align*}
  -\Delta u &= \lambda u,\quad \text{in } \Omega,\\
  n \cdot \nabla u &= 0 \quad \text{on } \partial\Omega,
\end{align*}



where:
- $\Omega$ is the computational domain (typically 2D),

Steps to run

```bash
cmake .
make
./eigen_value
```

After running, the eigenvectors will be stored in eigenvectors.vtu 

Following is the command to view the file in VisIt
```bash
visit -o eigenvectors.vtu
```
