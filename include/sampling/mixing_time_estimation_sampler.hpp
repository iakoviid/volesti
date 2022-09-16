// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2020 Vissarion Fisikopoulos
// Copyright (c) 2018-2020 Apostolos Chalkis
// Copyright (c) 2022-2022 Ioannis Iakovidis

// Contributed and/or modified by Ioannis Iakovidis, as part of Google Summer of
// Code 2022 program.

// Licensed under GNU LGPL.3, see LICENCE file

// References
// Yunbum Kook, Yin Tat Lee, Ruoqi Shen, Santosh S. Vempala. "Sampling with
// Riemannian Hamiltonian

#ifndef MIXING_TIME_ESTIMATION_SAMPLER_HPP
#define MIXING_TIME_ESTIMATION_SAMPLER_HPP
#include "diagnostics/effective_sample_size.hpp"
#include <omp.h>
double global_sampling_rate=0;
double global_est_num_samples=0;
bool global_terminate=false;
template <typename Walk>
class mixing_time_estimation_sampler
{
  using NT = typename Walk::NT;
  using MT = typename Walk::MT;
  using VT = typename Walk::VT;
  using Opts = typename Walk::Opts;

public:
  bool removedInitial = false;
  NT sampling_rate = 0;
  NT sampling_rate_outside = 0;
  NT est_num_samples = 0;
  NT est_num_samples_outside = 0;
  int totalNumSamples = 0;
  Opts &options;
  NT nextEstimateStep;
  MT &chains;
  bool terminate = false;
  int N;
  int dimension;
  mixing_time_estimation_sampler(Walk &s, int num_samples, MT &randPoints, int dim)
      : options(s.params.options), chains(randPoints)
  {
    nextEstimateStep = options.initialStepEst;
    N = num_samples;
    dimension = dim;
  }

  void estimation_step(Walk &s)
  {
    if (s.nEffectiveStep.mean() > nextEstimateStep)
    {
      unsigned int min_eff_samples = 1;
      effective_sample_size<NT, VT, MT>(chains, min_eff_samples);
      std::cerr<<min_eff_samples<<"\n";
      if (removedInitial == false &&
          min_eff_samples > 2 * options.nRemoveInitialSamples)
      {
        int num_batches = chains.cols() / s.simdLen;
        int k = std::ceil(options.nRemoveInitialSamples * (num_batches / min_eff_samples));
        s.num_runs = std::ceil(s.num_runs * (1 - k / num_batches));
        s.acceptedStep = s.acceptedStep * (1 - k / num_batches);
        int q = chains.cols() - (k - 1) * s.simdLen;
        chains.leftCols(q) = chains.rightCols(q);
        chains.conservativeResize(chains.rows(), q);
        removedInitial = true;
        effective_sample_size<NT, VT, MT>(chains, min_eff_samples);
      }
      int num_batches = chains.cols() / s.simdLen;
      NT mixingTime = num_batches / min_eff_samples;
      sampling_rate = s.simdLen / mixingTime;
      est_num_samples = s.num_runs * sampling_rate;
      estimation_update(s);
    }
  }
  void estimation_update(Walk &s)
  {
    NT total_sampling_rate = sampling_rate + sampling_rate_outside;
    NT totalNumSamples = est_num_samples + est_num_samples_outside;
    if (totalNumSamples > N && removedInitial)
    {
      terminate = true;
    }
    else if (removedInitial)
    {
      NT estimateEndingStep = N / total_sampling_rate * ((s.nEffectiveStep).mean() / s.num_runs);
      nextEstimateStep = std::min(nextEstimateStep * options.step_multiplier, estimateEndingStep);
    }
    else
    {
      NT estimateEndingStep = (2 * options.nRemoveInitialSamples * s.simdLen) / sampling_rate * ((s.nEffectiveStep).mean() / s.num_runs);
      nextEstimateStep = std::min(nextEstimateStep * options.step_multiplier, estimateEndingStep);
    }
  }
  template <typename RNGType>
  void apply(Walk &s, RNGType &rng)
  {
    while (!terminate)
    {
      s.apply(rng);
      estimation_step(s);
      if(options.num_threads > 1){
        sync();
      }
      MT sample = s.getPoints();
      chains.conservativeResize(dimension, chains.cols() + s.simdLen);
      chains.rightCols(s.simdLen) = sample;
    }
  }
  void clear(){
    chains.resize(dimension,0);
    nextEstimateStep = options.initialStepEst;
  }
  template <typename RNGType>
  void apply(Walk &s, RNGType &rng, int N)
  {
    for (int i = 0; i < N; i++)
    {
      s.apply(rng);
      estimation_step(s);
      if(options.num_threads > 1){
        sync();
      }
      MT sample = s.getPoints();
      chains.conservativeResize(dimension, chains.cols() + s.simdLen);
      chains.rightCols(s.simdLen) = sample;
    }
  }
  //Broadcast terminate, sampling_rate, est_num_samples
  void sync(){
    int thread_index = omp_get_thread_num();
    global_sampling_rate=0;
    global_est_num_samples=0;
    #pragma omp barrier
    #pragma omp critical
    {
        global_sampling_rate+=sampling_rate;
        global_est_num_samples+=est_num_samples;
        global_terminate= global_terminate || terminate;
    }
    #pragma omp barrier
    sampling_rate_outside=global_sampling_rate-sampling_rate;
    est_num_samples_outside=global_est_num_samples-est_num_samples;
    terminate=global_terminate;
  }
};

template <
    typename PointList,
    typename Polytope,
    typename RandomNumberGenerator,
    typename WalkTypePolicy,
    typename NT,
    typename Point,
    typename Input,
    typename Solver,
    typename Opts>
void crhmc_sampling(PointList &randPoints,
                    Polytope &P,
                    RandomNumberGenerator &rng,
                    const unsigned int &walk_len,
                    const unsigned int &rnum,
                    const Point &starting_point,
                    unsigned int const &nburns,
                    Input &input,
                    Opts &options)
{
  using NegativeGradientFunctor = typename Input::Grad;
  using NegativeLogprobFunctor = typename Input::Func;
  typedef typename WalkTypePolicy::template Walk<
      Point,
      Polytope,
      RandomNumberGenerator,
      NegativeGradientFunctor,
      NegativeLogprobFunctor,
      Solver>
      walk;

  typedef typename WalkTypePolicy::template parameters<
      NT,
      NegativeGradientFunctor>
      walk_params;

  // Initialize random walk parameters
  unsigned int dim = starting_point.dimension();
  walk_params params(input.df, dim, options);

  if (input.df.params.eta > 0)
  {
    params.eta = input.df.params.eta;
  }

  Point p = starting_point;

  walk crhmc_walk = walk(P, p, input.df, input.f, params);

  typedef mixing_time_estimation_sampler<walk> RandomPointGenerator;
  RandomPointGenerator r = RandomPointGenerator(crhmc_walk, rnum, randPoints, input.dimension);
  if(nburns > 0)
    {
      r.apply(crhmc_walk, rng, nburns);
      r.clear();
    }

  r.apply(crhmc_walk, rng);
}
template <
    typename PointList,
    typename Polytope,
    typename RandomNumberGenerator,
    typename WalkTypePolicy,
    typename NT,
    typename Point,
    typename Input,
    typename Solver,
    typename Opts>
void parallel_crhmc_sampling(PointList &randPoints,
                    Polytope &P,
                    RandomNumberGenerator &rng,
                    const unsigned int &walk_len,
                    const unsigned int &rnum,
                    const Point &starting_point,
                    unsigned int const &nburns,
                    Input &input,
                    Opts &options,
                    unsigned int const& num_threads)
{
  std::cerr<<"Number of threads "<<num_threads<<"\n";
  using NegativeGradientFunctor = typename Input::Grad;
  using NegativeLogprobFunctor = typename Input::Func;
  typedef typename WalkTypePolicy::template Walk<
      Point,
      Polytope,
      RandomNumberGenerator,
      NegativeGradientFunctor,
      NegativeLogprobFunctor,
      Solver>
      walk;

  typedef typename WalkTypePolicy::template parameters<
      NT,
      NegativeGradientFunctor>
      walk_params;
  omp_set_num_threads(num_threads);
  options.num_threads=num_threads;
  // Initialize random walk parameters
  unsigned int dim = starting_point.dimension();

  Point p = starting_point;

  typedef mixing_time_estimation_sampler<walk> RandomPointGenerator;
  std::vector<PointList> points(num_threads);
  #pragma omp parallel
  {
    int thread_index = omp_get_thread_num();
    walk_params params(input.df, dim, options);
    if (input.df.params.eta > 0)
    {
      params.eta = input.df.params.eta;
    }
    walk crhmc_walk= walk(P, p, input.df, input.f, params);
    RandomPointGenerator r=RandomPointGenerator(crhmc_walk, rnum, points[thread_index], input.dimension);
  if(nburns > 0)
    {
      r.apply(crhmc_walk, rng, nburns);
      r.clear();
    }

  r.apply(crhmc_walk, rng);
  }
  for(unsigned int i=0;i<num_threads;i++){
    randPoints.conservativeResize(input.dimension,randPoints.cols()+points[i].cols());
    randPoints.rightCols(points[i].cols())=points[i];
    points[i].resize(0,0);
  }
}
#endif
