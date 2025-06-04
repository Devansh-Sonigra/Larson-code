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
#include <deal.II/lac/solver_cg.h>

#include <iostream>
#include <fstream>
#include <cmath>

using namespace dealii;


class ExactSolution : public Function<2>
{
    public:
        ExactSolution () : Function<2>() {}

        double value (const Point<2>   &p,
                const unsigned int  component = 0) const override;
        Tensor<1,2> gradient (const Point<2>   &p,
                const unsigned int  component = 0) const override;
};

//template<>
double ExactSolution::value (const Point<2> &p, const unsigned int /*component*/) const
{
//    return p[0] * p[0] + p[1] *p[1];
//   return p[0] + p[1];
    return sin(p[0] * p[1]);
}

//template<>
Tensor<1,2> ExactSolution::gradient (const Point<2>   &p, const unsigned int) const
{
    Tensor<1,2> values;
//    values[0] = 2 * p[0]; 
//    values[1] = 2 * p[1];
//    return values;

//    values[0] = 1;
//    values[1] = 1;
//    return values;

    values[0] = p[1] * cos(p[0] * p[1]);
    values[1] = p[0] * cos(p[0] * p[1]);
    return values;
}

int main()
{

    ExactSolution exact_solution;

    Triangulation<2> tria;
    Triangulation<2> triangulation;


    GridGenerator::hyper_cube(tria);
    //  tria.refine_global(1);

    GridGenerator::convert_hypercube_to_simplex_mesh(tria, triangulation);
    triangulation.refine_global(1);
    triangulation.refine_global(1);
    triangulation.refine_global(1);
//    triangulation.refine_global(1);

    DoFHandler<2> dof_handler(triangulation);
    const FE_SimplexP<2> fe(1);
    MappingFE<2> mapping(fe);

    QGaussSimplex<2> quadrature_formula(fe.degree + 1);
    FEValues<2> fe_values (mapping, fe, quadrature_formula, update_values | update_gradients | update_quadrature_points | update_JxW_values);

    dof_handler.distribute_dofs(fe);

//  Sparsity pattern for mass matrix
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    SparsityPattern sparsity_pattern;
    sparsity_pattern.copy_from(dsp);

//  Initializing the all the matrices
    SparseMatrix<double> mass_matrix(sparsity_pattern);
    Vector<double> solution;
    Vector<double> system_rhs;

//  To solve the system 
    SolverControl           solver_control (1000, 1e-12);
    SolverCG<Vector<double>>              cg (solver_control);

    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    const unsigned int   n_q_points    = quadrature_formula.size();

    FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       cell_rhs (dofs_per_cell);
    std::vector<unsigned int> local_dof_indices (dofs_per_cell);

    for(const auto  &cell : dof_handler.active_cell_iterators()){
        fe_values.reinit(cell);
        cell_matrix = 0;
        cell_rhs = 0;

        for(unsigned int q_point = 0; q_point< n_q_points; ++q_point) {
            for(unsigned int i = 0; i < dofs_per_cell ; ++i) {
                for(unsigned int j = 0; j < dofs_per_cell; ++j) {
                    cell_matrix(i, j) += fe_values.shape_value(i, q_point) * fe_values.shape_value(j, q_point) * fe_values.JxW(q_point);
                }
                cell_rhs(i) += exact_solution.value(fe_values.quadrature_point(q_point)) * fe_values.shape_value(i, q_point) * fe_values.JxW(q_point);
            }
        }

        cell->get_dof_indices(local_dof_indices);
        for(unsigned int i = 0; i < dofs_per_cell ; ++i) {
            for(unsigned int j = 0; j < dofs_per_cell; ++j) {
                mass_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i,j));
            }
            system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }

    }

    cg.solve(mass_matrix, solution, system_rhs, PreconditionIdentity());

    Vector<double> difference_per_cell (triangulation.n_active_cells());
    VectorTools::integrate_difference (mapping, dof_handler, solution, exact_solution, difference_per_cell, QGaussSimplex<2>(2*fe.degree+1),VectorTools::L2_norm);

    double L2_error = difference_per_cell.l2_norm();

//    VectorTools::interpolate(mapping, dof_handler,exact_solution, system_rhs);
//    VectorTools::project(mapping, dof_handler,exact_solution, system_rhs);
//    system_rhs -= solution;
//    std::cout << system_rhs.l2_norm() << " error" << std::endl;

//    for(int i = 0; i < dof_handler.n_dofs(); i++) {
//        std::cout << solution(i) << " ";
//    }

    std::cout<<"The error is "<< L2_error << std::endl;
    std::cout << "CG converged in " << solver_control.last_step() << " iterations." << std::endl;

    std::ofstream out("grid-1.vtu");
    GridOut       grid_out;
    grid_out.write_vtu(triangulation, out);
    std::cout << "Grid written to grid-1.vtu" << std::endl;

    return 0;
}
