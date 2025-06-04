# $L^2$ Projection
This folder contains the implementation of $L^2$ projection which is describe in chapter 3 of the book. The goal is to project given $L^2$ function onto space of piecewise linear polynomial.

Steps to run

```bash
cmake .
make
./projection
```

After running, the numerical error will be written to a LaTeX file named error.tex.

To compile it to PDF:
```bash
pdflatex error.tex
```
