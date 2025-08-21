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

// template<int dim>
// class ExactSolution : public Function<dim>
// {
// public:
//     ExactSolution () : Function<dim>() {}

//     double value (const Point<dim>   &p,
//                   const unsigned int  component = 0) const override;
//     Tensor<1, dim> gradient (const Point<dim>   &p,
//                              const unsigned int  component = 0) const override;
// };

// template<>
// double ExactSolution<2>::value (const Point<2> &p,
//                                 const unsigned int /*component*/) const
// {
//     return p[0] * (1 - p[0]) * p[1] * ( 1 - p[1]);
// }


// template<>
// double ExactSolution<3>::value (const Point<3> &p,
//                                 const unsigned int /*component*/) const
// {
//     return p[0] * (1 - p[0]) * p[1] * ( 1 - p[1]) * p[2] * (1 - p[2]);
// }

// template<>
// Tensor<1, 2> ExactSolution<2>::gradient (const Point<2>   &p,
//                                          const unsigned int) const
// {
//     Tensor<1, 2> values;
//     values[0] = (1 - 2 * p[0]) *  p[1] * ( 1 - p[1]);
//     values[1] = p[0] * (1 - p[0])  * ( 1 - 2 * p[1]);
//     return values;
// }


// template<>
// Tensor<1, 3> ExactSolution<3>::gradient (const Point<3>   &p,
//                                          const unsigned int) const
// {
//     Tensor<1, 3> values;
//     values[0] = (1 - 2 * p[0]) *  p[1] * ( 1 - p[1]) * p[2] * (1 - p[2]);
//     values[1] = p[0] * (1 - p[0])  * ( 1 - 2 * p[1]) * p[2] * (1 - p[2]);
//     values[2] = p[0] * (1 - p[0]) * p[1] * ( 1 - p[1]) * ( 1 - 2 * p[2]);
//     return values;
// }
template<int dim>
class initial_function : public Function<dim>
{
public:
    initial_function () : Function<dim>() {}

    double value (const Point<dim>   &p,
                  const unsigned int  component = 0) const override;
};

template<>
double initial_function<2>::value (const Point<2> &p,
                               const unsigned int /*component*/) const
{
    // if (p[0] * p[0] + p[1] * p[1] < 1) {
    //     return(exp(1/(1 - p[0] * p[0] - p[1] * p[1])));
    // } else {
    //     return 0.0;
    // }
    return 2*M_PI*M_PI*sin(M_PI*p[0])*sin(M_PI*p[1]);
}


template<int dim>
class RHS_function : public Function<dim>
{
public:
    RHS_function () : Function<dim>() {}

    double value (double &time, const Point<dim>   &p,
                  const unsigned int  component = 0) const ;
};

template<>
double RHS_function<2>::value (double &time, const Point<2> &p,
                               const unsigned int /*component*/) const
{
    // return 2 * ( p[0] + p[1] - p[0] * p[0] - p[1] * p[1]);
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
        void assemble_rhs(unsigned int &k);
        void solve(unsigned int &n);
        void output_results(unsigned int &n) const;


        double                  final_time = 1.0;
        double                  del_t = 1e-2;

        Triangulation<dim>      triangulation;
        const FE_Q<dim>         fe;
        DoFHandler<dim>         dof_handler;

        SparsityPattern         sparsity_pattern;
        SparseMatrix<double>    system_matrix;
        SparseMatrix<double>    stiffness_matrix;

        Vector<double>          system_rhs;
        Vector<double>          solution;
        Vector<double>          temp_solution;
};

template<int dim>
HeatSolver<dim>::HeatSolver(unsigned int degree):
    fe(degree),
    dof_handler(triangulation)
{}

template<int dim>
void HeatSolver<dim>::make_grid_and_dofs()
{
    // std::cout << "make grid " << std::endl;
    GridGenerator::hyper_cube(triangulation, 0, 1);
    triangulation.refine_global(8);
    //  tria.refine_global(1);
}

template<int  dim>
void HeatSolver<dim>::setup_system()
{
    // std::cout << "setup system " << std::endl;
    dof_handler.distribute_dofs(fe);

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    //  Initializing the all the matrices
    system_matrix.reinit(sparsity_pattern);
    stiffness_matrix.reinit(sparsity_pattern);
    system_rhs.reinit(dof_handler.n_dofs());
    solution.reinit(dof_handler.n_dofs());
    temp_solution.reinit(dof_handler.n_dofs());
}

template<int dim>
void HeatSolver<dim>::assemble_system()
{
    system_matrix = 0;
    stiffness_matrix = 0;

    system_rhs = 0;

    QGauss<dim> quadrature_formula(2 * fe.degree + 1);
    FEValues<dim>  fe_values(fe, quadrature_formula,
                                 update_values | update_gradients | update_quadrature_points |
                                 update_JxW_values);

    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    const unsigned int   n_q_points    = quadrature_formula.size();
    initial_function<dim> rhs_function;

    FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
    FullMatrix<double>   cell_matrix_stiff (dofs_per_cell, dofs_per_cell);
    Vector<double>       cell_rhs (dofs_per_cell);
    std::vector<unsigned int> local_dof_indices (dofs_per_cell);
    
    double temp;
    for(const auto  &cell : dof_handler.active_cell_iterators()) 
    {
        fe_values.reinit(cell);
        cell_matrix = 0;
        cell_matrix_stiff = 0;
        cell_rhs = 0;
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
            temp = rhs_function.value(fe_values.quadrature_point(q_point));
            for (unsigned int i = 0; i < dofs_per_cell ; ++i)
            {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    cell_matrix(i,j) += fe_values.shape_value(i, q_point) *  fe_values.shape_value(j, q_point) * fe_values.JxW(q_point);
                    cell_matrix_stiff(i, j) += fe_values.shape_grad(i, q_point) *  fe_values.shape_grad(j, q_point) * fe_values.JxW(q_point);
                }
                cell_rhs(i) += temp * fe_values.shape_value(i, q_point) * fe_values.JxW(q_point);
            }
        }

        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell ; ++i) 
        {
            for (unsigned int j = 0; j < dofs_per_cell; ++j) 
            {
                system_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i, j));
                stiffness_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix_stiff(i, j));
            }
            system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
    }

    system_matrix.add(del_t, stiffness_matrix);
    std::map<unsigned int, double> boundary_values;
    VectorTools::interpolate_boundary_values (//mapping,
                                              dof_handler,
                                              0,
                                              Functions::ZeroFunction<dim>(),
                                              boundary_values);
    MatrixTools::apply_boundary_values (boundary_values,
                                        stiffness_matrix,
                                        solution,
                                        system_rhs);
}

template<int dim>
void HeatSolver<dim>::assemble_rhs(unsigned int &k)
{
    system_rhs = 0;

    // This takes input of what polynomial degree to be integrated exactly
    // QGaussSimplex<dim> quadrature_formula(fe.degree + 1);
    QGauss<dim> quadrature_formula(2 * fe.degree+1);
    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_values | update_gradients | update_quadrature_points |
                             update_JxW_values);

    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    const unsigned int   n_q_points    = quadrature_formula.size();
    // ExactSolution<dim> exact_solution;
    RHS_function<dim> rhs_function;
    double temp_1 = k * del_t;
    double temp;

    // FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       cell_rhs (dofs_per_cell);
    std::vector<unsigned int> local_dof_indices (dofs_per_cell);

    system_matrix.add(-del_t, stiffness_matrix);
    system_matrix.vmult(temp_solution, solution);
    for (const auto  &cell : dof_handler.active_cell_iterators()) {
        fe_values.reinit(cell);
        cell_rhs = 0;

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
            // float temp = exact_solution.value(fe_values.quadrature_point(q_point));
            temp = rhs_function.value(temp_1, fe_values.quadrature_point(q_point));
            for (unsigned int i = 0; i < dofs_per_cell ; ++i) {
                cell_rhs(i) += del_t * temp * fe_values.shape_value(i, q_point) * fe_values.JxW(q_point);
            }
        }

        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell ; ++i) {
            system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }

    }

    // boundary condition
    system_rhs += temp_solution;
    system_matrix.add(del_t, stiffness_matrix);
    std::map<unsigned int, double> boundary_values;
    VectorTools::interpolate_boundary_values (//mapping,
                                              dof_handler,
                                              0,
                                              Functions::ZeroFunction<dim>(),
                                              boundary_values);
    MatrixTools::apply_boundary_values (boundary_values,
                                        system_matrix,
                                        solution,
                                        system_rhs);
}

template<int dim>
void HeatSolver<dim>::solve(unsigned int &n)
{
    // std::cout << 1e-12 * system_rhs.l2_norm() << std::endl;
    SolverControl               solver_control(1000, 1e-12 * system_rhs.l2_norm());
    SolverGMRES<Vector<double>> cg(solver_control);

    if(n == 0) 
    {
        cg.solve(stiffness_matrix, solution, system_rhs, PreconditionIdentity());
    } else {
        cg.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
    }
}

template <int dim>
void HeatSolver<dim>::output_results(unsigned int &n) const
{
  DataOut<dim> data_out;

  data_out.attach_dof_handler(dof_handler);

  // Vector<double> temp_solution(dof_handler.n_dofs());
  // for (unsigned i = 0; i < dof_handler.n_dofs(); ++i) {
  //     temp_solution[i] = system_solution[i];
  // }

  data_out.add_data_vector(solution, "solution");
  // data_out.add_data_vector(solution_jx, std::string("jx_") + case_name);
  // data_out.add_data_vector(solution_jy, std::string("jy_") + case_name);
  // data_out.add_data_vector(solution_H, std::string("Hz_") + case_name);

  data_out.build_patches(fe.degree);

  std::ofstream output(std::string("heat_sol_") + Utilities::int_to_string(n) + std::string(".vtu"));
  data_out.write_vtu(output);
}

template<int dim>
void HeatSolver<dim>::run()
{
    unsigned int noi = std::floor(final_time/del_t);
    make_grid_and_dofs();
    setup_system();
    assemble_system();
    for (unsigned int n = 0; n < noi; ++n) {
        // if (n == 0) {
            // std::cout << "hello " << std::endl;
        // } else {
            // refine_grid();
        // }
        solve(n);
        assemble_rhs(n);
        output_results(n);
        // compute_error(L2_error[n], H1_error[n]);

        // ncell[n] = triangulation.n_active_cells();
        // ndofs[n] = dof_handler.n_dofs();
    }
}

int main ()
{
    deallog.depth_console (0);
    // unsigned int nrefine = 10;
    // unsigned int nrefine = 7;
    unsigned int degree = 1;

    // std::cout << "hello " << std::endl;
    HeatSolver<2> problem (degree);
    // std::vector<int> ncell(nrefine), ndofs(nrefine), niterations(nrefine);
    // std::vector<double> L2_error(nrefine), H1_error(nrefine);
    // std::cout << "hello " << std::endl;
    problem.run ();
    // ConvergenceTable  convergence_table;
    // for (unsigned int n = 0; n < nrefine; ++n) {
    //     // std::cout<< n << std::endl;
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

    // std::ofstream error_table_file("error_identity.tex");
    // convergence_table.write_tex(error_table_file);

    return 0;
}
