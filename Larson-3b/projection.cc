#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
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
#include <deal.II/base/convergence_table.h>

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
       Tensor<1,dim> gradient (const Point<dim>   &p,
               const unsigned int  component = 0) const override;
};

template<>
double ExactSolution<2>::value (const Point<2> &p, const unsigned int /*component*/) const
{
    //    return p[0] * p[0] + p[1] *p[1];
    //   return p[0] + p[1];
    return sin(p[0] * p[1]);
}


template<>
double ExactSolution<3>::value (const Point<3> &p, const unsigned int /*component*/) const
{
       return p[0] * p[0] + p[1] *p[1] + p[2] * p[2];
    //   return p[0] + p[1];
    // return sin(p[0] * p[1]);
}

template<>
Tensor<1,2> ExactSolution<2>::gradient (const Point<2>   &p, const unsigned int) const
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


template<>
Tensor<1,3> ExactSolution<3>::gradient (const Point<3>   &p, const unsigned int) const
{
    Tensor<1,3> values;
       values[0] = 2 * p[0]; 
       values[1] = 2 * p[1];
       values[2] = 2 * p[2];
       return values;

    //    values[0] = 1;
    //    values[1] = 1;
    //    return values;

    // values[0] = p[1] * cos(p[0] * p[1]);
    // values[1] = p[0] * cos(p[0] * p[1]);
    return values;
}

template<int dim>
class Projection {
    public:
        Projection( unsigned int nrefine, unsigned int degree, std::string mesh_type);
        void run(std::vector<int> &ncell,
                std::vector<int> &ndofs,
                std::vector<double> &L2_error,
                std::vector<double> &H1_error);

    private:
        void make_grid_and_dofs();
        void setup_system();
        void assemble_system();
        void solve();
        void compute_error(double &L2_error, double &H1_error);
        void refine_grid();

        unsigned int            nrefine;
        std::string             mesh_type;
        Triangulation<dim>      triangulation;
        const FE_SimplexP<dim>  fe_1;
        const FE_Q<dim>         fe_2;
        MappingFE<dim>          mapping;
        DoFHandler<dim>         dof_handler;

        SparsityPattern         sparsity_pattern;
        SparseMatrix<double>    mass_matrix;

        Vector<double>          system_rhs;
        Vector<double>          solution;
};

template<int dim>
Projection<dim>::Projection( unsigned int nrefine, unsigned int degree, std::string mesh_type):
    nrefine(nrefine),
    mesh_type(mesh_type),
    fe_1(degree),
    fe_2(degree),
    mapping(FE_SimplexP<dim>(1)),
    dof_handler(triangulation)
{}

template<int dim>
void Projection<dim>::make_grid_and_dofs() {

    if (mesh_type == "triangle") {
        Triangulation<dim> tria;
        GridGenerator::hyper_cube(tria);
        //  tria.refine_global(1);

        GridGenerator::convert_hypercube_to_simplex_mesh(tria, triangulation);
    } else  {
        GridGenerator::hyper_cube(triangulation);  
        triangulation.refine_global(1);
    }
}

template<int dim>
void Projection<dim>::setup_system() {
    
    if (mesh_type == "triangle") {
        dof_handler.distribute_dofs(fe_1);
    } else {
        dof_handler.distribute_dofs(fe_2);
    }

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    //  Initializing the all the matrices
    mass_matrix.reinit(sparsity_pattern);
    system_rhs.reinit(dof_handler.n_dofs());
    solution.reinit(dof_handler.n_dofs());
}

template<int dim>
void Projection<dim>::assemble_system() {
    mass_matrix = 0;
    system_rhs = 0;
    unsigned int dofs_per_cell;

    if (mesh_type == "triangle") {
        QGaussSimplex<dim> quadrature_formula(fe_1.degree + 1);
        FEValues<dim> fe_values (mapping, fe_1, quadrature_formula, update_values | update_gradients | update_quadrature_points | update_JxW_values);

        dofs_per_cell = fe_1.dofs_per_cell;
        const unsigned int   n_q_points    = quadrature_formula.size();
        ExactSolution<dim> exact_solution;  

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
    } else {
        QGauss<dim> quadrature_formula(fe_2.degree + 1);
        FEValues<dim> fe_values( fe_2, quadrature_formula, update_values | update_gradients | update_quadrature_points | update_JxW_values);
        dofs_per_cell = fe_2.dofs_per_cell;

        const unsigned int   n_q_points    = quadrature_formula.size();
        ExactSolution<dim> exact_solution;  

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
    }

}

template<int dim>
void Projection<dim>::solve() {
    SolverControl           solver_control (1000, 1e-12);
    SolverCG<Vector<double>>              cg (solver_control);
    cg.solve(mass_matrix, solution, system_rhs, PreconditionIdentity());
}

template<int dim>
void Projection<dim>::compute_error(double &L2_error, double &H1_error) {
    ExactSolution<dim> exact_solution;  

    if (mesh_type == "triangle") {
        Vector<double> difference_per_cell (triangulation.n_active_cells());
        VectorTools::integrate_difference (mapping, dof_handler, solution, exact_solution, difference_per_cell, QGaussSimplex<dim>(2*fe_1.degree+1),VectorTools::L2_norm);

        L2_error = difference_per_cell.l2_norm();

        VectorTools::integrate_difference (mapping, dof_handler, solution, exact_solution, difference_per_cell, QGaussSimplex<dim>(2*fe_1.degree+1),VectorTools::H1_seminorm);

        H1_error = difference_per_cell.l2_norm();
    } else {
        Vector<double> difference_per_cell (triangulation.n_active_cells());
        VectorTools::integrate_difference (dof_handler, solution, exact_solution, difference_per_cell, QGauss<dim>(2*fe_2.degree+1),VectorTools::L2_norm);

        L2_error = difference_per_cell.l2_norm();

        VectorTools::integrate_difference (dof_handler, solution, exact_solution, difference_per_cell, QGauss<dim>(2*fe_2.degree+1),VectorTools::H1_seminorm);

        H1_error = difference_per_cell.l2_norm();
          
    }
}

template<int dim>
void Projection<dim>::refine_grid() {
    triangulation.refine_global(1);  
}

template<int dim>
void Projection<dim>::run(std::vector<int> &ncell,
        std::vector<int> &ndofs,
        std::vector<double> &L2_error,
        std::vector<double> &H1_error) {
    for (unsigned int n = 0; n < nrefine; ++n) {
        if (n == 0) {
            make_grid_and_dofs();  
        }
        else {
            refine_grid();
        }

        setup_system();
        assemble_system();
        solve();
        compute_error(L2_error[n], H1_error[n]);

        ncell[n] = triangulation.n_active_cells();
        ndofs[n] = dof_handler.n_dofs();
    }
}

int main ()
{
    deallog.depth_console (0);
    // unsigned int nrefine = 10;
    unsigned int nrefine = 4; 
    unsigned int degree = 1; 
    std::string mesh_type = "triangle";

    Projection<2> problem (nrefine, degree, mesh_type);
    std::vector<int> ncell(nrefine), ndofs(nrefine);
    std::vector<double> L2_error(nrefine), H1_error(nrefine);
    problem.run (ncell, ndofs, L2_error, H1_error);
    ConvergenceTable  convergence_table;
    for(unsigned int n=0; n<nrefine; ++n)
    {
        // std::cout<< n << std::endl;
        convergence_table.add_value("cells", ncell[n]);
        convergence_table.add_value("dofs",  ndofs[n]);
        convergence_table.add_value("L2",    L2_error[n]);
        convergence_table.add_value("H1",    H1_error[n]);
    }

    convergence_table.set_precision("L2", 3);
    convergence_table.set_scientific("L2", true);

    convergence_table.set_precision("H1", 3);
    convergence_table.set_scientific("H1", true);

    convergence_table.set_tex_caption("cells", "\\# cells");
    convergence_table.set_tex_caption("dofs", "\\# dofs");
    convergence_table.set_tex_caption("L2", "$L^2$-error");
    convergence_table.set_tex_caption("H1", "$H^1$-error");

    convergence_table.set_tex_format("cells", "r");
    convergence_table.set_tex_format("dofs",  "r");

    convergence_table.evaluate_convergence_rates
        ("L2", ConvergenceTable::reduction_rate_log2);
    convergence_table.evaluate_convergence_rates
        ("H1", ConvergenceTable::reduction_rate_log2);

    std::cout << std::endl;
    convergence_table.write_text(std::cout);

    std::ofstream error_table_file("error.tex");
    convergence_table.write_tex(error_table_file);

    return 0;
}
