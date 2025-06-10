# $L^2$ Projection
This folder contains the implementation of finding solution Poisson's equation which is describe in chapter 4 of the book. The problem solved is:

\begin{align*}
  -\Delta u &= f \quad \text{in } \Omega,\\
  u &= 0 \quad \text{on } \partial\Omega,
\end{align*}

where:
- \(\Omega\) is the computational domain (typically 2D),
- \(f\) is a given source term.

Steps to run

```bash
cmake .
make
./poisson
```

After running, the numerical error will be written to a LaTeX file named error.tex.

To compile it to PDF:
```bash
pdflatex error.tex
```
