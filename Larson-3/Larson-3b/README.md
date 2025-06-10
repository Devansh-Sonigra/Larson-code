# $L^2$ Projection
This folder contains the implementation of $L^2$ projection which is describe in chapter 3 of the book. The goal is to project given $L^2$ function onto space of piecewise linear polynomial. This program can also run on quadrilateral mesh. To change the mesh type between triangle and quadrilateral, change the variable named "mesh_type" in the program appropriately.

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
