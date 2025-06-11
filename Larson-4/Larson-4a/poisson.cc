#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/lac/sparse_decomposition.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/sparse_mic.h>

#include <iostream>
#include <fstream>
#include <cmath>

using namespace dealii;

template<int dim>
class ExactSolution : public Function<dim>
{
public:
    ExactSolution () : Function<dim>() {}

    double value (const Point<dim>   &p,
                  const unsigned int  component = 0) const override;
    Tensor<1, dim> gradient (const Point<dim>   &p,
                             const unsigned int  component = 0) const override;
};

template<>
double ExactSolution<2>::value (const Point<2> &p,
                                const unsigned int /*component*/) const
{
    return p[0] * (1 - p[0]) * p[1] * ( 1 - p[1]);
}


template<>
double ExactSolution<3>::value (const Point<3> &p,
                                const unsigned int /*component*/) const
{
    return p[0] * (1 - p[0]) * p[1] * ( 1 - p[1]) * p[2] * (1 - p[2]);
}

template<>
Tensor<1, 2> ExactSolution<2>::gradient (const Point<2>   &p,
                                         const unsigned int) const
{
    Tensor<1, 2> values;
    values[0] = (1 - 2 * p[0]) *  p[1] * ( 1 - p[1]);
    values[1] = p[0] * (1 - p[0])  * ( 1 - 2 * p[1]);
    return values;
}


template<>
Tensor<1, 3> ExactSolution<3>::gradient (const Point<3>   &p,
                                         const unsigned int) const
{
    Tensor<1, 3> values;
    values[0] = (1 - 2 * p[0]) *  p[1] * ( 1 - p[1]) * p[2] * (1 - p[2]);
    values[1] = p[0] * (1 - p[0])  * ( 1 - 2 * p[1]) * p[2] * (1 - p[2]);
    values[2] = p[0] * (1 - p[0]) * p[1] * ( 1 - p[1]) * ( 1 - 2 * p[2]);
    return values;
}

template<int dim>
class RHS_function : public Function<dim>
{
public:
    RHS_function () : Function<dim>() {}

    double value (const Point<dim>   &p,
                  const unsigned int  component = 0) const override;
};

template<>
double RHS_function<2>::value (const Point<2> &p,
                               const unsigned int /*component*/) const
{
    return 2 * ( p[0] + p[1] - p[0] * p[0] - p[1] * p[1]);
}


template<>
double RHS_function<3>::value (const Point<3> &p,
                               const unsigned int /*component*/) const
{
    return 2 * (p[1] * ( 1 - p[1] ) * p[2] * (1 - p[2]) + p[0] *
                ( 1 - p[0] ) * p[1] * (1 - p[1]) + p[0] * ( 1 - p[0] ) * p[2] * (1 - p[2]) );
}

template<int dim>
class PoissonSolver
{
public:
    PoissonSolver( unsigned int nrefine, unsigned int degree);
    void run(std::vector<int> &ncell,
             std::vector<int> &ndofs,
             std::vector<double> &L2_error,
             std::vector<double> &H1_error,
             std::vector<int> &niterations);

private:
    void make_grid_and_dofs();
    void setup_system();
    void assemble_system();
    void solve(int &niteration);
    void compute_error(double &L2_error, double &H1_error);
    void refine_grid();

    unsigned int            nrefine;
    Triangulation<dim>      triangulation;
    const FE_SimplexP<dim>  fe;
    MappingFE<dim>          mapping;
    DoFHandler<dim>         dof_handler;

    SparsityPattern         sparsity_pattern;
    SparseMatrix<double>    stiffness_matrix;

    Vector<double>          system_rhs;
    Vector<double>          solution;
};

template<int dim>
PoissonSolver<dim>::PoissonSolver( unsigned int nrefine, unsigned int degree):
    nrefine(nrefine),
    fe(degree),
    mapping(FE_SimplexP<dim>(1)),
    dof_handler(triangulation)
{}

template<int dim>
void PoissonSolver<dim>::make_grid_and_dofs()
{
    // std::cout << "make grid " << std::endl;
    Triangulation<dim> tria;
    GridGenerator::hyper_cube(tria);
    //  tria.refine_global(1);

    GridGenerator::convert_hypercube_to_simplex_mesh(tria, triangulation);

}

template<int dim>
void PoissonSolver<dim>::setup_system()
{
    // std::cout << "setup system " << std::endl;
    dof_handler.distribute_dofs(fe);

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    //  Initializing the all the matrices
    stiffness_matrix.reinit(sparsity_pattern);
    system_rhs.reinit(dof_handler.n_dofs());
    solution.reinit(dof_handler.n_dofs());
}


template<int dim>
void PoissonSolver<dim>::assemble_system()
{
    // std::cout << "Solver " << std::endl;
    stiffness_matrix = 0;
    system_rhs = 0;

    // This takes input of what polynomial degree to be integrated exactly
    QGaussSimplex<dim> quadrature_formula(fe.degree + 1);
    FEValues<dim> fe_values (mapping, fe, quadrature_formula,
                             update_values | update_gradients | update_quadrature_points |
                             update_JxW_values);

    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    const unsigned int   n_q_points    = quadrature_formula.size();
    // ExactSolution<dim> exact_solution;
    RHS_function<dim> rhs_function;

    FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       cell_rhs (dofs_per_cell);
    std::vector<unsigned int> local_dof_indices (dofs_per_cell);

    for (const auto  &cell : dof_handler.active_cell_iterators()) {
        fe_values.reinit(cell);
        cell_matrix = 0;
        cell_rhs = 0;

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
            // float temp = exact_solution.value(fe_values.quadrature_point(q_point));
            float temp = rhs_function.value(fe_values.quadrature_point(q_point));
            for (unsigned int i = 0; i < dofs_per_cell ; ++i) {
                for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                    cell_matrix(i, j) += fe_values.shape_grad(i, q_point) * fe_values.shape_grad(j,
                                         q_point) * fe_values.JxW(q_point);
                }
                cell_rhs(i) += temp * fe_values.shape_value(i,
                                                            q_point) * fe_values.JxW(q_point);
            }
        }

        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell ; ++i) {
            for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                stiffness_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i,
                                     j));
            }
            system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }

    }

    // boundary condition
    std::map<unsigned int, double> boundary_values;
    VectorTools::interpolate_boundary_values (mapping,
                                              dof_handler,
                                              0,
                                              ExactSolution<dim>(),
                                              boundary_values);
    MatrixTools::apply_boundary_values (boundary_values,
                                        stiffness_matrix,
                                        solution,
                                        system_rhs);
}

template<int dim>
void PoissonSolver<dim>::solve(int &niteration)
{
    // std::cout << "Solver " << std::endl;
    SolverControl           solver_control (1000, 1e-12 * system_rhs.l2_norm());
    SolverCG<Vector<double>>              cg (solver_control);

    // SparseILU<double> preconditioner;
    // preconditioner.initialize(stiffness_matrix);
    // PreconditionJacobi<SparseMatrix<double>> preconditioner;
    // preconditioner.initialize(stiffness_matrix);

    // cg.solve takes solution as initial guess
    // cg.solve(stiffness_matrix, solution, system_rhs, preconditioner);
    cg.solve(stiffness_matrix, solution, system_rhs, PreconditionIdentity());
    niteration = solver_control.last_step();
}

template<int dim>
void PoissonSolver<dim>::compute_error(double &L2_error, double &H1_error)
{
    // std::cout << "Error ";
    ExactSolution<dim> exact_solution;

    Vector<double> difference_per_cell (triangulation.n_active_cells());
    VectorTools::integrate_difference (mapping, dof_handler, solution,
                                       exact_solution, difference_per_cell, QGaussSimplex<dim>(2 * fe.degree + 1),
                                       VectorTools::L2_norm);

    L2_error = difference_per_cell.l2_norm();

    VectorTools::integrate_difference (mapping, dof_handler, solution,
                                       exact_solution, difference_per_cell, QGaussSimplex<dim>(2 * fe.degree + 1),
                                       VectorTools::H1_seminorm);

    H1_error = difference_per_cell.l2_norm();
}

template<int dim>
void PoissonSolver<dim>::refine_grid()
{
    triangulation.refine_global(1);
}

template<int dim>
void PoissonSolver<dim>::run(std::vector<int> &ncell,
                             std::vector<int> &ndofs,
                             std::vector<double> &L2_error,
                             std::vector<double> &H1_error,
                             std::vector<int> &niterations)
{
    for (unsigned int n = 0; n < nrefine; ++n) {
        if (n == 0) {
            // std::cout << "hello " << std::endl;
            make_grid_and_dofs();
        } else {
            refine_grid();
        }

        setup_system();
        assemble_system();
        solve(niterations[n]);
        compute_error(L2_error[n], H1_error[n]);

        ncell[n] = triangulation.n_active_cells();
        ndofs[n] = dof_handler.n_dofs();
    }
}

int main ()
{
    deallog.depth_console (0);
    // unsigned int nrefine = 10;
    unsigned int nrefine = 7;
    unsigned int degree = 1;

    // std::cout << "hello " << std::endl;
    PoissonSolver<2> problem (nrefine, degree);
    std::vector<int> ncell(nrefine), ndofs(nrefine), niterations(nrefine);
    std::vector<double> L2_error(nrefine), H1_error(nrefine);
    // std::cout << "hello " << std::endl;
    problem.run (ncell, ndofs, L2_error, H1_error, niterations);
    ConvergenceTable  convergence_table;
    for (unsigned int n = 0; n < nrefine; ++n) {
        // std::cout<< n << std::endl;
        convergence_table.add_value("cells", ncell[n]);
        convergence_table.add_value("dofs",  ndofs[n]);
        convergence_table.add_value("iterations",  niterations[n]);
        convergence_table.add_value("L2",    L2_error[n]);
        convergence_table.add_value("H1",    H1_error[n]);
    }

    convergence_table.set_precision("L2", 3);
    convergence_table.set_scientific("L2", true);

    convergence_table.set_precision("H1", 3);
    convergence_table.set_scientific("H1", true);

    convergence_table.set_tex_caption("cells", "\\# cells");
    convergence_table.set_tex_caption("dofs", "\\# dofs");
    convergence_table.set_tex_caption("iterations", "\\# iterations");
    convergence_table.set_tex_caption("L2", "$L^2$-error");
    convergence_table.set_tex_caption("H1", "$H^1$-error");

    convergence_table.set_tex_format("cells", "r");
    convergence_table.set_tex_format("dofs",  "r");
    convergence_table.set_tex_format("iterations",  "r");

    convergence_table.evaluate_convergence_rates
    ("L2", ConvergenceTable::reduction_rate_log2);
    convergence_table.evaluate_convergence_rates
    ("H1", ConvergenceTable::reduction_rate_log2);

    std::cout << std::endl;
    convergence_table.write_text(std::cout);

    std::ofstream error_table_file("error_identity.tex");
    convergence_table.write_tex(error_table_file);

    return 0;
}
