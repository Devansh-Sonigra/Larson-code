// Here we have solved a problem given in book which does not contains the exact solution
// And hence I have commented out the things that require exact solution which is calculating error for testing convergence rates.
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
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
#include <deal.II/lac/solver_gmres.h>

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
    // return 2 * ( p[0] + p[1] - p[0] * p[0] - p[1] * p[1]);
    return 0.0;
}


template<>
double RHS_function<3>::value (const Point<3> &p,
                               const unsigned int /*component*/) const
{
    return 2 * (p[1] * ( 1 - p[1] ) * p[2] * (1 - p[2]) + p[0] *
                ( 1 - p[0] ) * p[1] * (1 - p[1]) + p[0] * ( 1 - p[0] ) * p[2] * (1 - p[2]) );
}

template<int dim>
class coeff_function: public Function<dim>
{
public:
    coeff_function() : Function<dim>() {}
    double value( const Point<dim> &p,
                  const unsigned int component = 0) const override;
};

template<>
double coeff_function<2>::value( const Point<2> &p, const unsigned int ) const
{
    return 1.0;
}

template<>
double coeff_function<3>::value( const Point<3> &p, const unsigned int ) const
{
    return 1.0;
}

template<int dim>
class kappa: public Function<dim>
{
public:
    kappa() : Function<dim>() {}
    double value( const Point<dim> &p,
                  const unsigned int component = 0) const override;
};

template<>
double kappa<2>::value( const Point<2> &p, const unsigned int ) const
{
    // return 1.0;
    if ( p[0] > 1.99) {
        return 1e+6;
    } else {
        return 0;
    }
}

template<>
double kappa<3>::value( const Point<3> &p, const unsigned int ) const
{
    return 1.0;
}

template<int dim>
class G_Dirichlet: public Function<dim>
{
public:
    G_Dirichlet() : Function<dim>() {}
    double value( const Point<dim> &p,
                  const unsigned int component = 0) const override;
};

template<>
double G_Dirichlet<2>::value( const Point<2> &p, const unsigned int ) const
{
    return 0.0;
}

template<>
double G_Dirichlet<3>::value( const Point<3> &p, const unsigned int ) const
{
    return 1.0;
}

template<int dim>
class G_Neumann: public Function<dim>
{
public:
    G_Neumann() : Function<dim>() {}
    double value( const Point<dim> &p,
                  const unsigned int component = 0) const override;
};

template<>
double G_Neumann<2>::value( const Point<2> &p, const unsigned int ) const
{
    // if (p[0] == 0 || p[0] == 1) {
    //     return - p[1] * ( 1- p[1]) - 1;
    // } else {
    //     return - p[0] * ( 1- p[0]) - 1;
    // }
    if ( p[0] < -0.9999) {
        return 1.0;
    } else  {
        return 0.0;
    }
}

template<>
double G_Neumann<3>::value( const Point<3> &p, const unsigned int ) const
{
    // For 3d example
    if (p[0] == 0 || p[0] == 1) {
        return - p[1] * ( 1 - p[1]) * p[2] * ( 1 - p[2]) - 1;
    } else if (p[1] == 0 || p[1] == 1) {
        return - p[0] * ( 1 - p[0]) * p[2] * ( 1 - p[2]) - 1;
    } else {
        return - p[0] * ( 1 - p[0]) * p[1] * ( 1 - p[1]) - 1;
    }
}

template<int dim>
class Solver
{
public:
    Solver( unsigned int nrefine, unsigned int degree);
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
    void output_results();
    void compute_error(double &L2_error, double &H1_error);
    void refine_grid();

    unsigned int            nrefine;
    Triangulation<dim>      triangulation;
    const FE_SimplexP<dim>  fe;
    MappingFE<dim>          mapping;
    DoFHandler<dim>         dof_handler;

    SparsityPattern         sparsity_pattern;
    SparseMatrix<double>    system_matrix;

    Vector<double>          system_rhs;
    Vector<double>          solution;
};

template<int dim>
Solver<dim>::Solver( unsigned int nrefine, unsigned int degree):
    nrefine(nrefine),
    fe(degree),
    mapping(FE_SimplexP<dim>(1)),
    dof_handler(triangulation)
{}

template<int dim>
void Solver<dim>::make_grid_and_dofs()
{
    // Triangulation<dim> tria;
    // GridGenerator::hyper_cube(tria);
    // //  tria.refine_global(1);

    // GridGenerator::convert_hypercube_to_simplex_mesh(tria, triangulation);
    GridIn<2> gridin;
    gridin.attach_triangulation(triangulation);

    std::ifstream input_file("naca.msh");
    gridin.read_msh(input_file);
}

template<int dim>
void Solver<dim>::setup_system()
{
    dof_handler.distribute_dofs(fe);

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    //  Initializing the all the matrices
    system_matrix.reinit(sparsity_pattern);
    system_rhs.reinit(dof_handler.n_dofs());
    solution.reinit(dof_handler.n_dofs());
}


template<int dim>
void Solver<dim>::assemble_system()
{
    system_matrix = 0;
    system_rhs = 0;

    // This takes input of what polynomial degree to be integrated exactly
    QGaussSimplex<dim> cell_quadrature_formula(fe.degree + 1);
    FEValues<dim> fe_values (mapping, fe, cell_quadrature_formula,
                             update_values | update_gradients | update_quadrature_points |
                             update_JxW_values);

    QGaussSimplex < dim - 1 > face_quadrature_formula(fe.degree + 1);
    FEFaceValues<dim> face_fe_values(mapping, fe, face_quadrature_formula,
                                     update_values | update_normal_vectors | update_quadrature_points |
                                     update_JxW_values);



    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    const unsigned int   n_q_points    = cell_quadrature_formula.size();
    const unsigned int   face_n_q_points = face_quadrature_formula.size();

    ExactSolution<dim> exact_solution;
    RHS_function<dim> rhs_function;
    coeff_function<dim> coeff_a;
    kappa<dim> kap;
    G_Dirichlet<dim> g_dir;
    G_Neumann<dim> g_neu;

    FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       cell_rhs (dofs_per_cell);
    std::vector<unsigned int> local_dof_indices (dofs_per_cell);

    for (const auto  &cell : dof_handler.active_cell_iterators()) {
        fe_values.reinit(cell);
        cell_matrix = 0;
        cell_rhs = 0;

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
            const double rhs_value = rhs_function.value(fe_values.quadrature_point(
                                                            q_point));
            const double a_value = coeff_a.value(fe_values.quadrature_point(q_point));
            for (unsigned int i = 0; i < dofs_per_cell ; ++i) {
                for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                    cell_matrix(i, j) +=  a_value * fe_values.shape_grad(i,
                                                                         q_point) * fe_values.shape_grad(j, q_point) * fe_values.JxW(q_point);
                }
                cell_rhs(i) += rhs_value * fe_values.shape_value(i,
                                                                 q_point) * fe_values.JxW(q_point);
            }
        }

        for (unsigned int f = 0; f < cell->n_faces(); ++f) {
            if (cell->face(f)->at_boundary()) {
                face_fe_values.reinit(cell, f);
                for (unsigned int q_point = 0; q_point < face_n_q_points; ++q_point) {
                    const double kappa_value = kap.value(face_fe_values.quadrature_point(q_point));
                    const double g_dir_value = g_dir.value(face_fe_values.quadrature_point(
                                                               q_point));
                    const double g_neu_value = g_neu.value(face_fe_values.quadrature_point(
                                                               q_point));

                    for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                        cell_rhs(i) += ( kappa_value * g_dir_value + g_neu_value) *
                                       face_fe_values.shape_value(i, q_point) * face_fe_values.JxW(q_point);
                        for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                            cell_matrix(i, j) += kappa_value * face_fe_values.shape_value(i,
                                                 q_point) * face_fe_values.shape_value(j,
                                                                                       q_point) * face_fe_values.JxW(q_point);
                        }
                    }

                }
            }
        }

        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell ; ++i) {
            for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                system_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i,
                                  j));
            }
            system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
    }
}

template<int dim>
void Solver<dim>::solve(int &niteration)
{
    SolverControl           solver_control (1000, 1e-12 * system_rhs.l2_norm());
    SolverCG<Vector<double>>   cg(solver_control);

    SparseILU<double> preconditioner;
    preconditioner.initialize(system_matrix);

    cg.solve(system_matrix, solution, system_rhs, preconditioner);
    niteration = solver_control.last_step();
}

template <int dim>
void Solver<dim>::output_results ()
{
    DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (solution, "solution");
    data_out.build_patches (fe.degree);
    std::string fname = "solution-" + Utilities::int_to_string(nrefine,
                                                               2) + ".vtu";
    std::ofstream output (fname);
    data_out.write_vtu (output);
}

template<int dim>
void Solver<dim>::compute_error(double &L2_error, double &H1_error)
{
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
void Solver<dim>::refine_grid()
{
    triangulation.refine_global(1);
}

template<int dim>
void Solver<dim>::run(std::vector<int> &ncell,
                      std::vector<int> &ndofs,
                      std::vector<double> &L2_error,
                      std::vector<double> &H1_error,
                      std::vector<int> &niterations)
{
    for (unsigned int n = 0; n < nrefine; ++n) {
        if (n == 0) {
            make_grid_and_dofs();
        } else {
            refine_grid();
        }

        setup_system();
        assemble_system();
        solve(niterations[n]);
        // compute_error(L2_error[n], H1_error[n]);

        ncell[n] = triangulation.n_active_cells();
        ndofs[n] = dof_handler.n_dofs();
    }
    output_results();
}

int main ()
{
    deallog.depth_console (0);
    // We are running only 1 iteration as we want solution of provided mesh
    unsigned int nrefine = 1;
    unsigned int degree = 1;

    Solver<2> problem (nrefine, degree);
    std::vector<int> ncell(nrefine), ndofs(nrefine), niterations(nrefine);
    std::vector<double> L2_error(nrefine), H1_error(nrefine);
    problem.run (ncell, ndofs, L2_error, H1_error, niterations);
    // ConvergenceTable  convergence_table;
    // for (unsigned int n = 0; n < nrefine; ++n) {
    //     convergence_table.add_value("cells", ncell[n]);
    //     convergence_table.add_value("dofs",  ndofs[n]);
    //     convergence_table.add_value("iterations",  niterations[n]);
    //     convergence_table.add_value("L2",    L2_error[n]);
    //     convergence_table.add_value("H1",    H1_error[n]);
    // }

    // convergence_table.set_precision("L2", 3);
    // convergence_table.set_scientific("L2", true);

    // convergence_table.set_precision("H1", 3);
    // convergence_table.set_scientific("H1", true);

    // convergence_table.set_tex_caption("cells", "\\# cells");
    // convergence_table.set_tex_caption("dofs", "\\# dofs");
    // convergence_table.set_tex_caption("iterations", "\\# iterations");
    // convergence_table.set_tex_caption("L2", "$L^2$-error");
    // convergence_table.set_tex_caption("H1", "$H^1$-error");

    // convergence_table.set_tex_format("cells", "r");
    // convergence_table.set_tex_format("dofs",  "r");
    // convergence_table.set_tex_format("iterations",  "r");

    // convergence_table.evaluate_convergence_rates
    // ("L2", ConvergenceTable::reduction_rate_log2);
    // convergence_table.evaluate_convergence_rates
    // ("H1", ConvergenceTable::reduction_rate_log2);

    // std::cout << std::endl;
    // convergence_table.write_text(std::cout);

    // std::ofstream error_table_file("error.tex");
    // convergence_table.write_tex(error_table_file);

    return 0;
}
