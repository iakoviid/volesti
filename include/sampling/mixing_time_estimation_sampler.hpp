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
template <typename Walk>
class mixing_time_estimation_sampler {
  using NT= typename Walk::NT;
  using MT= typename Walk::MT;
  using VT= typename Walk::VT;
  using Opts=typename Walk::Opts;
public:
  bool removedInitial = false;
  NT sampling_rate = 0;
  NT sampling_rate_outside = 0;
  NT est_num_samples = 0;
  NT est_num_samples_outside = 0;
  int totalNumSamples=0;
  Opts &options;
  NT nextEstimateStep;
  MT &chains;
  bool terminate=false;
  int N;
  int iba=0;
  int dimension;
  mixing_time_estimation_sampler(Walk &s,int num_samples, MT& randPoints, int dim)
      : options(s.params.options), chains(randPoints) {
    nextEstimateStep = options.initialStepEst;
    N=num_samples;
    dimension=dim;
  }

  void estimation_step(Walk &s) {
    if (s.nEffectiveStep.mean() > nextEstimateStep) {
      iba++;
      unsigned int min_eff_samples = 1;
      effective_sample_size<NT, VT, MT>(chains,min_eff_samples);
      //std::cerr<<min_eff_samples<<"\n";
      if (removedInitial == false &&
          min_eff_samples > 2 * options.nRemoveInitialSamples) {
        int num_batches=chains.cols()/s.simdLen;
        int k = std::ceil(options.nRemoveInitialSamples * (num_batches / min_eff_samples));
        std::cerr<<k<<"\n";
        s.num_runs = std::ceil(s.num_runs * (1 - k / num_batches));
        s.acceptedStep = s.acceptedStep * (1 - k / num_batches);
        int q=chains.cols()-(k-1)*s.simdLen;
        chains.leftCols(q) = chains.rightCols(q);
        chains.conservativeResize(chains.rows(),q);
        removedInitial = true;
        effective_sample_size<NT, VT, MT>(chains,min_eff_samples);
      }
      int num_batches=chains.cols()/s.simdLen;
      NT mixingTime =  num_batches / min_eff_samples;
      sampling_rate = s.simdLen / mixingTime;
      est_num_samples = s.num_runs * sampling_rate;
    }
  }
  void estimation_update(Walk &s){
    NT total_sampling_rate=sampling_rate+ sampling_rate_outside;
    NT totalNumSamples=est_num_samples+est_num_samples_outside;
    if(totalNumSamples> N && removedInitial){
      terminate=true;
    }else if(removedInitial){
      NT estimateEndingStep = N / total_sampling_rate * ((s.nEffectiveStep).mean() / s.num_runs);
      nextEstimateStep= std::min(nextEstimateStep*options.step_multiplier,estimateEndingStep);
    }else{
      NT estimateEndingStep = (2 * options.nRemoveInitialSamples * s.simdLen) / sampling_rate * ((s.nEffectiveStep).mean() / s.num_runs);
      nextEstimateStep= std::min(nextEstimateStep*options.step_multiplier,estimateEndingStep);
    }
  }
template<typename RNGType>
  void apply(Walk &s,RNGType& rng){
    while(!terminate){
      s.apply(rng);
      estimation_step(s);
      MT sample = s.getPoints();
      chains.conservativeResize(dimension, chains.cols()+s.simdLen);
      chains.rightCols(s.simdLen) = sample;
    }
  }
  template<typename RNGType>
    void apply(Walk &s,RNGType& rng, int N){
      for(int i=0;i<N;i++){
        if(i%1000==0){
          std::cerr<<i<<" out of "<<N<<" "<< iba<<"\n";
        }
        s.apply(rng);
        estimation_step(s);
        MT sample = s.getPoints();
        chains.conservativeResize(dimension, chains.cols()+s.simdLen);
        chains.rightCols(s.simdLen) = sample;
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
        typename Opts
        >
void crhmc_sampling(PointList &randPoints,
                       Polytope &P,
                       RandomNumberGenerator &rng,
                       const unsigned int &walk_len,
                       const unsigned int &rnum,
                       const Point &starting_point,
                       unsigned int const& nburns,
                       Input& input,
                       Opts& options)
{
    using NegativeGradientFunctor=typename Input::Grad;
    using NegativeLogprobFunctor=typename Input::Func;
    typedef typename WalkTypePolicy::template Walk
            <
                    Point,
                    Polytope,
                    RandomNumberGenerator,
                    NegativeGradientFunctor,
                    NegativeLogprobFunctor,
                    Solver
            > walk;

    typedef typename WalkTypePolicy::template parameters
            <
                    NT,
                    NegativeGradientFunctor
            > walk_params;

    // Initialize random walk parameters
    unsigned int dim = starting_point.dimension();
    walk_params params(input.df, dim, options);

    if (input.df.params.eta > 0) {
        params.eta = input.df.params.eta;
    }

    Point p = starting_point;

    walk crhmc_walk=walk(P, p, input.df, input.f, params);

    typedef mixing_time_estimation_sampler<walk> RandomPointGenerator;
    RandomPointGenerator r=RandomPointGenerator(crhmc_walk, rnum, randPoints, input.dimension);
    r.apply(crhmc_walk,rng, rnum);
}
#endif
