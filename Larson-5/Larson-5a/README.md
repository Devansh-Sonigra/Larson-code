# Poisson Equation
This folder contains the implementation of finding solution Heat equation which is describe in chapter 5 of the book. The problem solved is:

$$
  u_t-\Delta u &= f \quad \text{in } \Omega,\\
  u &= 0 \quad \text{on } \partial\Omega,\, \forall t, \\
  u(x,0) = u_0(x)
$$


where:
- $\Omega$ is the computational domain (typically 2D),
- $f$ is a given source term.

Steps to run

```bash
cmake .
make
./heat
```

The output will be stored in solution.vtu

Following is the command to view the file in VisIt
```bash
visit -o solution.vtu
```
