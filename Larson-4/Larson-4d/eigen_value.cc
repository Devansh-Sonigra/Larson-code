#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/numerics/data_out.h>
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
#include <deal.II/base/index_set.h>
#include <deal.II/lac/arpack_solver.h>
#include <deal.II/lac/sparse_direct.h>

#include <iostream>
#include <fstream>
#include <cmath>

using namespace dealii;

template<int dim>
class EigenValueSolver
{
public:
  EigenValueSolver(unsigned int degree);
  void run();

private:
  void make_grid_and_dofs();
  void setup_system();
  void assemble_system();
  void solve();
  void output_results() const;
  void refine_grid();

  unsigned int            nrefine;
  Triangulation<dim>      triangulation;
  const FE_SimplexP<dim>  fe;
  MappingFE<dim>          mapping;
  DoFHandler<dim>         dof_handler;

  SparsityPattern         sparsity_pattern;
  SparseMatrix<double>    stiffness_matrix, mass_matrix;
  std::vector<Vector<double>>     eigenfunctions;

  std::vector<std::complex<double>> eigenvalues;
  int                     n_eigen_values;
};

template<int dim>
EigenValueSolver<dim>::EigenValueSolver(unsigned int degree):
  fe(degree),
  mapping(FE_SimplexP<dim>(1)),
  dof_handler(triangulation)
{}

template<int dim>
void
EigenValueSolver<dim>::make_grid_and_dofs()
{
  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria);

  GridGenerator::convert_hypercube_to_simplex_mesh(tria, triangulation);

}

template<int dim>
void
EigenValueSolver<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);

  DynamicSparsityPattern dsp_dofs(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp_dofs);

  sparsity_pattern.copy_from(dsp_dofs);

  stiffness_matrix.reinit(sparsity_pattern);
  mass_matrix.reinit(sparsity_pattern);

  n_eigen_values = 10;
  eigenfunctions.resize(n_eigen_values + 1);
  for(auto& eigenfunction : eigenfunctions)
  {
    eigenfunction.reinit(dof_handler.n_dofs());
  }

  eigenvalues.resize(n_eigen_values);
}


template<int dim>
void
EigenValueSolver<dim>::assemble_system()
{
  stiffness_matrix = 0;
  mass_matrix = 0;

  QGaussSimplex<dim> quadrature_formula(fe.degree + 1);
  FEValues<dim> fe_values(mapping, fe, quadrature_formula,
                          update_values | update_gradients | update_quadrature_points |
                          update_JxW_values);

  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();

  FullMatrix<double>   cell_stiffness_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double>   cell_mass_matrix(dofs_per_cell, dofs_per_cell);
  std::vector<unsigned int> local_dof_indices(dofs_per_cell);


  for(const auto&  cell : dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);
    cell_stiffness_matrix = 0;
    cell_mass_matrix = 0;

    for(unsigned int q_point = 0; q_point < n_q_points; ++q_point)
    {
      for(unsigned int i = 0; i < dofs_per_cell ; ++i)
      {
        for(unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          cell_stiffness_matrix(i, j) += fe_values.shape_grad(i, q_point) * fe_values.shape_grad(j, q_point) * fe_values.JxW(q_point);
          cell_mass_matrix(i, j) += fe_values.shape_value(i, q_point) * fe_values.shape_value(j, q_point) * fe_values.JxW(q_point);
        }
      }
    }

    cell->get_dof_indices(local_dof_indices);
    for(unsigned int i = 0; i < dofs_per_cell ; ++i)
    {
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        stiffness_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_stiffness_matrix(i, j));
        mass_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_mass_matrix(i, j));
      }
    }

  }
}

template<int dim>
void
EigenValueSolver<dim>::solve()
{
  SolverControl solver_control(1000, 1e-12);
  SparseDirectUMFPACK inverse;
  inverse.initialize(stiffness_matrix);

  const unsigned int num_arnoldi_vectors = 2 * eigenvalues.size() + 2;
  ArpackSolver::AdditionalData additional_data(num_arnoldi_vectors);

  ArpackSolver eigensolver(solver_control, additional_data);
  eigensolver.solve(stiffness_matrix,
                    mass_matrix,
                    inverse,
                    eigenvalues,
                    eigenfunctions,
                    n_eigen_values);
}

template <int dim>
void
EigenValueSolver<dim>::output_results() const
{
  DataOut<dim> data_out;

  data_out.attach_dof_handler(dof_handler);

  for(unsigned int i = 0; i < eigenfunctions.size(); ++i)
    data_out.add_data_vector(eigenfunctions[i],
                             std::string("eigenfunction_") +
                             Utilities::int_to_string(i));
  data_out.build_patches();

  std::ofstream output("eigenvectors.vtu");
  data_out.write_vtu(output);
}

template<int dim>
void
EigenValueSolver<dim>::refine_grid()
{
  triangulation.refine_global(6);
}

template<int dim>
void
EigenValueSolver<dim>::run()
{
  make_grid_and_dofs();
  refine_grid();
  setup_system();
  assemble_system();
  solve();
  output_results();
  std::cout << std::endl;
  for(unsigned int i = 0; i < eigenvalues.size(); ++i)
    std::cout << "      Eigenvalue " << i << " : " << eigenvalues[i] << std::endl;
}

int
main()
{
  deallog.depth_console(0);

  unsigned int degree = 1;
  EigenValueSolver<2> problem(degree);
  problem.run();

  return 0;
}
