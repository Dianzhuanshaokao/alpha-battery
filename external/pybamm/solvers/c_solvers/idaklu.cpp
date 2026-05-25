#include "idaklu/casadi_solver.hpp"
#include "idaklu/common.hpp"
#include "idaklu/python.hpp"

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

#include <vector>

Function generate_function(const std::string &data)
{
  return Function::deserialize(data);
}

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<np_array>);

PYBIND11_MODULE(idaklu, m)
{
  m.doc() = "sundials solvers"; // optional module docstring

  py::bind_vector<std::vector<np_array>>(m, "VectorNdArray");

  m.def("solve_python",
        [](np_array t, np_array y0, np_array yp0, residual_type res,
           jacobian_type jac, sensitivities_type sens, jac_get_type get_jac_data,
           jac_get_type get_jac_row_vals, jac_get_type get_jac_col_ptr,
           int nnz, event_type events, int number_of_events, int use_jacobian,
           np_array rhs_alg_id, np_array atol, double rtol, np_array inputs,
           int number_of_sensitivity_parameters) {
          return solve_python(
              t, y0, yp0, res, jac, sens, get_jac_data, get_jac_row_vals,
              get_jac_col_ptr, nnz, events, number_of_events, use_jacobian,
              rhs_alg_id, atol, rtol, inputs, number_of_sensitivity_parameters);
        },
        "The solve function for python evaluators", py::arg("t"), py::arg("y0"),
        py::arg("yp0"), py::arg("res"), py::arg("jac"), py::arg("sens"),
        py::arg("get_jac_data"), py::arg("get_jac_row_vals"),
        py::arg("get_jac_col_ptr"), py::arg("nnz"), py::arg("events"),
        py::arg("number_of_events"), py::arg("use_jacobian"),
        py::arg("rhs_alg_id"), py::arg("atol"), py::arg("rtol"),
        py::arg("inputs"), py::arg("number_of_sensitivity_parameters"));

  py::class_<CasadiSolver>(m, "CasadiSolver")
      .def("solve", &CasadiSolver::solve, "perform a solve", py::arg("t"),
           py::arg("y0"), py::arg("yp0"), py::arg("inputs"));

  m.def("create_casadi_solver", &create_casadi_solver,
        "Create a casadi idaklu solver object", py::arg("number_of_states"),
        py::arg("number_of_parameters"), py::arg("rhs_alg"),
        py::arg("jac_times_cjmass"), py::arg("jac_times_cjmass_colptrs"),
        py::arg("jac_times_cjmass_rowvals"), py::arg("jac_times_cjmass_nnz"),
        py::arg("jac_bandwidth_lower"), py::arg("jac_bandwidth_upper"),
        py::arg("jac_action"), py::arg("mass_action"), py::arg("sens"),
        py::arg("events"), py::arg("number_of_events"), py::arg("rhs_alg_id"),
        py::arg("atol"), py::arg("rtol"), py::arg("inputs"), py::arg("options"),
        py::return_value_policy::take_ownership);

  m.def(
      "generate_function",
      [](const std::string &data) { return generate_function(data); },
      "Generate a casadi function", py::arg("string"));

  py::class_<Function>(m, "Function");

  py::class_<Solution>(m, "solution")
      .def_readwrite("t", &Solution::t)
      .def_readwrite("y", &Solution::y)
      .def_readwrite("yS", &Solution::yS)
      .def_readwrite("flag", &Solution::flag);
}
