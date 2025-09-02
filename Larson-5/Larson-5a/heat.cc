#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/lac/sparse_decomposition.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/sparse_mic.h>
#include <deal.II/fe/fe_q.h>

#include <filesystem>
#include <iostream>
#include <fstream>
#include <cmath>

using namespace dealii;

template<int dim>
class initial_function : public Function<dim>
{
public:
    initial_function() : Function<dim>() {}

    double value(const Point<dim>&   p,
                 const unsigned int  component = 0) const override;
};

template<>
double
initial_function<2>::value(const Point<2>& p,
                           const unsigned int /*component*/) const
{
    return 2 * M_PI * M_PI * sin(M_PI * p[0]) * sin(M_PI * p[1]);
}


template<int dim>
class RHS_function : public Function<dim>
{
public:
    RHS_function() : Function<dim>() {}

    double value(double& time, const Point<dim>&   p,
                 const unsigned int  component = 0) const ;
};

template<>
double
RHS_function<2>::value(double& time, const Point<2>& p,
                       const unsigned int /*component*/) const
{
    return 0.0;
}

template<int dim>
class HeatSolver
{
public:
    HeatSolver(unsigned int degree);
    void run();

private:
    void make_grid_and_dofs();
    void setup_system();
    void assemble_system();
    void assemble_rhs(unsigned int& k);
    void solve(unsigned int& n);
    void output_results(unsigned int& n) const;


    double                  final_time = 1.0;
    double                  del_t = 1e-2;
    double                  theta = 0.5;

    Triangulation<dim>      triangulation;
    const FE_Q<dim>         fe;
    DoFHandler<dim>         dof_handler;

    SparsityPattern         sparsity_pattern;
    SparseMatrix<double>    system_matrix;
    SparseMatrix<double>    mass_matrix;
    SparseMatrix<double>    stiffness_matrix;

    Vector<double>          system_rhs;
    Vector<double>          solution;
};

template<int dim>
HeatSolver<dim>::HeatSolver(unsigned int degree):
    fe(degree),
    dof_handler(triangulation)
{}

template<int dim>
void
HeatSolver<dim>::make_grid_and_dofs()
{
    GridGenerator::hyper_cube(triangulation, 0, 1);
    triangulation.refine_global(8);
}

template<int  dim>
void
HeatSolver<dim>::setup_system()
{
    // std::cout << "setup system " << std::endl;
    dof_handler.distribute_dofs(fe);

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    //  Initializing the all the matrices
    system_matrix.reinit(sparsity_pattern);
    mass_matrix.reinit(sparsity_pattern);
    stiffness_matrix.reinit(sparsity_pattern);
    system_rhs.reinit(dof_handler.n_dofs());
    solution.reinit(dof_handler.n_dofs());
}

template<int dim>
void
HeatSolver<dim>::assemble_system()
{
    system_matrix = 0;
    mass_matrix = 0;
    stiffness_matrix = 0;

    system_rhs = 0;

    QGauss<dim> quadrature_formula(2 * fe.degree + 1);
    FEValues<dim>  fe_values(fe, quadrature_formula,
                             update_values | update_gradients | update_quadrature_points |
                             update_JxW_values);

    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    const unsigned int   n_q_points    = quadrature_formula.size();
    initial_function<dim> rhs_function;

    FullMatrix<double>   cell_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double>   cell_matrix_stiff(dofs_per_cell, dofs_per_cell);
    Vector<double>       cell_rhs(dofs_per_cell);
    std::vector<unsigned int> local_dof_indices(dofs_per_cell);

    double temp;
    for(const auto&  cell : dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        cell_matrix = 0;
        cell_matrix_stiff = 0;
        cell_rhs = 0;
        for(unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
            temp = rhs_function.value(fe_values.quadrature_point(q_point));
            for(unsigned int i = 0; i < dofs_per_cell ; ++i)
            {
                for(unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    cell_matrix(i, j) += fe_values.shape_value(i, q_point) *  fe_values.shape_value(j, q_point) * fe_values.JxW(q_point);
                    cell_matrix_stiff(i, j) += fe_values.shape_grad(i, q_point) *  fe_values.shape_grad(j, q_point) * fe_values.JxW(q_point);
                }
                cell_rhs(i) += temp * fe_values.shape_value(i, q_point) * fe_values.JxW(q_point);
            }
        }

        cell->get_dof_indices(local_dof_indices);
        for(unsigned int i = 0; i < dofs_per_cell ; ++i)
        {
            for(unsigned int j = 0; j < dofs_per_cell; ++j)
            {
                system_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i, j));
                mass_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i, j));
                stiffness_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix_stiff(i, j));
            }
            system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
    }

    system_matrix.add(del_t * theta, stiffness_matrix);
    std::map<unsigned int, double> boundary_values;
    VectorTools::interpolate_boundary_values( //mapping,
        dof_handler,
        0,
        Functions::ZeroFunction<dim>(),
        boundary_values);
    MatrixTools::apply_boundary_values(boundary_values,
                                       stiffness_matrix,
                                       solution,
                                       system_rhs);
}

template<int dim>
void
HeatSolver<dim>::assemble_rhs(unsigned int& k)
{
    system_rhs = 0;

    QGauss<dim> quadrature_formula(2 * fe.degree + 1);
    FEValues<dim> fe_values(fe, quadrature_formula,
                            update_values | update_gradients | update_quadrature_points |
                            update_JxW_values);

    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    const unsigned int   n_q_points    = quadrature_formula.size();
    RHS_function<dim> rhs_function;
    double temp_1 = k * del_t;
    double temp_2 = (k - 1) * del_t;
    double rhs_1;
    double rhs_2;

    Vector<double>       cell_rhs(dofs_per_cell);
    std::vector<unsigned int> local_dof_indices(dofs_per_cell);

    Vector<double>          temp_solution_1;
    temp_solution_1.reinit(dof_handler.n_dofs());
    mass_matrix.vmult(temp_solution_1, solution);

    Vector<double>          temp_solution_2;
    temp_solution_2.reinit(dof_handler.n_dofs());
    stiffness_matrix.vmult(temp_solution_2, solution);
    temp_solution_2 *= -del_t * (1 - theta);

    for(const auto&  cell : dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        cell_rhs = 0;

        for(unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
            rhs_1 = rhs_function.value(temp_1, fe_values.quadrature_point(q_point));
            rhs_2 = rhs_function.value(temp_2, fe_values.quadrature_point(q_point));
            for(unsigned int i = 0; i < dofs_per_cell ; ++i)
            {
                cell_rhs(i) += del_t * (theta * rhs_1 + (1 - theta) * rhs_2 )* fe_values.shape_value(i, q_point) *  fe_values.JxW(q_point);
            }
        }

        cell->get_dof_indices(local_dof_indices);
        for(unsigned int i = 0; i < dofs_per_cell ; ++i)
        {
            system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }

    }

    // boundary condition
    system_rhs += temp_solution_1;
    system_rhs += temp_solution_2;

    std::map<unsigned int, double> boundary_values;
    VectorTools::interpolate_boundary_values( //mapping,
        dof_handler,
        0,
        Functions::ZeroFunction<dim>(),
        boundary_values);
    MatrixTools::apply_boundary_values(boundary_values,
                                       system_matrix,
                                       solution,
                                       system_rhs);
}

template<int dim>
void
HeatSolver<dim>::solve(unsigned int& n)
{
    SolverControl               solver_control(1000, 1e-12 * system_rhs.l2_norm());
    SolverCG<Vector<double>> cg(solver_control);

    if(n == 0)
    {
        cg.solve(stiffness_matrix, solution, system_rhs, PreconditionIdentity());
    }
    else
    {
        cg.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
    }
}

template <int dim>
void
HeatSolver<dim>::output_results(unsigned int& n) const
{
    DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
    data_out.build_patches(fe.degree);

    std::ofstream output(std::string("heat_sol_") + Utilities::int_to_string(n) + std::string(".vtu"));
    data_out.write_vtu(output);
}

template<int dim>
void
HeatSolver<dim>::run()
{
    unsigned int noi = std::floor(final_time / del_t);
    make_grid_and_dofs();
    setup_system();
    assemble_system();
    for(unsigned int n = 1; n < noi; ++n)
    {
        unsigned int k = n - 1;
        solve(k);
        assemble_rhs(n);
        output_results(k);
    }
}

int
main()
{
    deallog.depth_console(0);
    unsigned int degree = 1;

    HeatSolver<2> problem(degree);
    problem.run();
    return 0;
}
