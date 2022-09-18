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
  using pts= typename std::vector<MT>;
public:
  bool removedInitial = false;
  NT sampling_rate = 0;
  NT sampling_rate_outside = 0;
  NT est_num_samples = 0;
  NT est_num_samples_outside = 0;
  int totalNumSamples = 0;
  Opts &options;
  NT nextEstimateStep;
  pts chains;
  MT& randPoints;
  bool terminate = false;
  int N;
  int dimension;
  mixing_time_estimation_sampler(Walk &s, int num_samples, MT &output_matrix, int dim)
      : options(s.params.options), randPoints(output_matrix)
  {
    nextEstimateStep = options.initialStepEst;
    N = num_samples;
    dimension = dim;
    chains=pts(options.simdLen,MT(s.dim,0));

  }

  NT min_ess(pts const &samples){
    NT minimum=std::numeric_limits<NT>::max();
    unsigned int min_eff_samples=0;
    for(int i=0;i<samples.size();i++){
      VT ess= effective_sample_size<NT, VT, MT>(samples[i], min_eff_samples);
      minimum=ess.minCoeff()<minimum ? ess.minCoeff():minimum;
    }
    return minimum;
  }
void resize(pts& samples,unsigned int n){
  unsigned int m=samples[0].cols()-n;
  unsigned int rows=samples[0].rows();
  for(int i=0;i<samples.size();i++){
    samples[i].leftCols(n)=samples[i].rightCols(n);
    samples[i].conservativeResize(rows,m);
  }
}
  void estimation_step(Walk &s)
  {
    if (s.nEffectiveStep.mean() > nextEstimateStep)
    {
      NT min_eff_samples = min_ess(chains);
      int thread_index = omp_get_thread_num();
      std::cerr<<"thread "<< thread_index<<" miness "<<min_eff_samples<<"\n";
      int num_batches = chains[0].cols();
      if (removedInitial == false &&
          min_eff_samples > 2 * options.nRemoveInitialSamples)
      {
        int k = std::ceil(options.nRemoveInitialSamples * (num_batches / min_eff_samples));
        s.num_runs = std::ceil(s.num_runs * (1 - k / num_batches));
        s.acceptedStep = s.acceptedStep * (1 - k / num_batches);
        resize(chains,k);
        removedInitial = true;
        NT min_eff_samples = min_ess(chains);
      }
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
  void push_back_sample(pts &samples,MT const& x){
    for(int i=0;i<samples.size();i++){
      samples[i].conservativeResize(samples[i].rows(),samples[i].cols()+1);
      samples[i].col(samples[i].cols()-1)=x.col(i);
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
      MT sample = s.x;
      push_back_sample(chains,sample);
    }
    finalize(s, chains, randPoints,options.raw_output);
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
      MT sample = s.x;
      push_back_sample(chains,sample);
    }
    finalize(s, chains, randPoints,options.raw_output);
  }
  //Broadcast terminate, sampling_rate, est_num_samples
  void sync(){
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
  //Compress the vectorized sample
  void finalize(Walk &s, pts &chains,MT &randPoints,bool raw_output){
    unsigned int dimension=s.P.y.rows();
    if(raw_output){
    for(int i=0;i<chains.size();i++){
      randPoints.conservativeResize(dimension,randPoints.cols()+chains[i].cols());
      randPoints.rightCols(chains[i].cols())=(s.P.T*chains[i]).colwise()+s.P.y;
      chains[i].resize(0,0);
      }
    }else{
      int N=chains[0].cols();
      for(int i=0;i<chains.size();i++){
        unsigned int min_eff_samples=0;
          NT ess=effective_sample_size<NT, VT, MT>(chains[i],min_eff_samples).minCoeff();
          unsigned int gap=std::ceil(N/ess);
          MT points=chains[i](Eigen::all,Eigen::seq(0,N-1,gap));
          unsigned int n=points.cols();
          randPoints.conservativeResize(dimension,randPoints.cols()+n);
          randPoints.rightCols(n)=(s.P.T*points).colwise()+s.P.y;
          chains[i].resize(0,0);
      }
    }
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
  r.apply(crhmc_walk, rng);
  if(nburns>0 && nburns<randPoints.cols()){
    int n_output=randPoints.cols()-nburns;
    randPoints.leftCols(n_output) = randPoints.rightCols(n_output);
    randPoints.conservativeResize(randPoints.rows(), n_output);
  }
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
    r.apply(crhmc_walk, rng);
  }
  for(unsigned int i=0;i<num_threads;i++){
    randPoints.conservativeResize(input.dimension,randPoints.cols()+points[i].cols());
    randPoints.rightCols(points[i].cols())=points[i];
    points[i].resize(0,0);
  }
  if(nburns>0 && nburns<randPoints.cols()){
    int n_output=randPoints.cols()-nburns;
    randPoints.leftCols(n_output) = randPoints.rightCols(n_output);
    randPoints.conservativeResize(randPoints.rows(), n_output);
  }
}

#endif
