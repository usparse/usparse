#ifndef AMG
#define AMG

#include "amg_types.hpp"

namespace usparse {
namespace amg {

AMGHierarchy amg_setup(const CsrMatrix& A, const AMGOptions& options = AMGOptions());

void amg_vcycle(const AMGHierarchy& data, const Vector& b, Vector& x);

SolveResult amg_solve(const AMGHierarchy& data, const Vector& b, Vector& x,
                        int max_cycles = 1000, double rtol = 1e-8,double atol = 1e-12);
}
}
#endif
