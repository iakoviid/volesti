// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2021 Vissarion Fisikopoulos
// Copyright (c) 2018-2021 Apostolos Chalkis

#include "Eigen/Eigen"
#include "diagnostics/multivariate_psrf.hpp"
#include "ode_solvers/ode_solvers.hpp"
#include "preprocess/crhmc/crhmc_input.h"
#include "preprocess/crhmc/crhmc_problem.h"
#include "random_walks/random_walks.hpp"
#include "sampling/mixing_time_estimation_sampler.hpp"
#include <boost/random.hpp>
#include <chrono>
using NT = double;
using Kernel = Cartesian<NT>;
using RNGType = BoostRandomNumberGenerator<boost::mt19937, NT>;
using Point = typename Kernel::Point;
typedef HPolytope<Point> HPOLYTOPE;
using MT = Eigen::Matrix<NT, Eigen::Dynamic, Eigen::Dynamic>;
using VT = Eigen::Matrix<NT, Eigen::Dynamic, 1>;
using Func = GaussianFunctor::FunctionFunctor<Point>;
using Grad = GaussianFunctor::GradientFunctor<Point>;
using Hess = GaussianFunctor::HessianFunctor<Point>;
using func_params = GaussianFunctor::parameters<NT, Point>;
using Input = crhmc_input<MT, Point, Func, Grad, Hess>;
using CrhmcProblem = crhmc_problem<Point, Input>;
using Solver = ImplicitMidpointODESolver<Point, NT, CrhmcProblem, Grad>;
void sample_gaussian(HPOLYTOPE HP, int numpoints = 1000, int nburns = 0, int num_threads=1) {
  int dimension = HP.dimension();
  NT variance = 1.0;
  func_params params = func_params(Point(dimension), variance, 1);
  Func f(params);
  Grad g(params);
  Hess h(params);
  Input input = Input(dimension, f, g, h);
  input.Aineq = HP.get_mat();
  input.bineq = HP.get_vec();
  Opts options;
  CrhmcProblem P = CrhmcProblem(input, options);

  MT randPoints;
  RNGType rng(1);
  unsigned int walkL = 1;
  #ifdef TIME_KEEPING
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    start = std::chrono::system_clock::now();
  #endif
  parallel_crhmc_sampling<MT, CrhmcProblem, RNGType, CRHMCWalk, NT, Point, Input, Solver,
                 Opts>(randPoints, P, rng, walkL, numpoints, Point(P.center),
                       nburns, input, options, num_threads);
  #ifdef TIME_KEEPING
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> total_time = end - start;
    std::cerr << "Total time: " << total_time.count() << "\n";
  #endif
  std::cout << randPoints.transpose();
  std::cerr << "PSRF: " << multivariate_psrf<NT, VT, MT>(randPoints)
            << std::endl;
}


HPOLYTOPE visualization_example() {
  MT A=MT(5,2);
  A<<1, 0, -0.25, -1, 2.5, 1,0.4, -1,-0.9, 0.5;
  VT b = 10*VT::Ones(5);
  return HPOLYTOPE(2, A, b);
}

int main(int argc, char *argv[]) {
  HPOLYTOPE HP = visualization_example();
  if (argc == 4) {
    sample_gaussian(HP, atoi(argv[1]), atoi(argv[2]),atoi(argv[3]));
  } else {
    std::cerr << "Example Usage: ./sampler [n_samples] [initial_burns] [num_threads]"
              <<  "\n";
  }
  return 0;
}
