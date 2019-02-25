/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions Contact  
//
// ************************************************************************
//@HEADER
*/

#include <Kokkos_Core.hpp>
#include <vector>

#define TURN_OFF_MERGE_CHUNKS // for debugging - will be removed

template<class scalar_t>
struct ArrayType {
  scalar_t * ptr;
  KOKKOS_INLINE_FUNCTION
  ArrayType(scalar_t * pSetPtr) : ptr(pSetPtr) {};
};

template<class policy_t, class scalar_t, class part_t>
struct ArraySumReducer {

  typedef ArraySumReducer reducer;
  typedef ArrayType<scalar_t> value_type;
  value_type * value;
  size_t value_count;

  KOKKOS_INLINE_FUNCTION ArraySumReducer(
    value_type &val,
    const size_t & count) :
    value(&val), value_count(count)
  {}

  KOKKOS_INLINE_FUNCTION
  value_type& reference() const {
    return *value;
  }

  KOKKOS_INLINE_FUNCTION
  void join(value_type& dst, const value_type& src)  const {
    for(int n = 0; n < value_count; ++n) {
      dst.ptr[n] += src.ptr[n];
    }
  }

  KOKKOS_INLINE_FUNCTION
  void join (volatile value_type& dst, const volatile value_type& src) const {
    for(int n = 0; n < value_count; ++n) {
      dst.ptr[n] += src.ptr[n];
    }
  }

  KOKKOS_INLINE_FUNCTION void init (value_type& dst) const {
    for(int n = 0; n < value_count; ++n) {
      dst.ptr[n] = 0;
    }
  }
};

template<class scalar_t, class part_t, class index_t, class device_t>
struct ReduceWeightsFunctorInnerLoop {

#ifdef TURN_OFF_MERGE_CHUNKS
  part_t concurrent_current_part;
#endif
  part_t current_work_part;
  part_t current_concurrent_num_parts;
  Kokkos::View<index_t*, device_t> permutations;
  Kokkos::View<scalar_t *, device_t> coordinates;
  Kokkos::View<scalar_t**, device_t> weights;
  Kokkos::View<part_t*, device_t> parts;
  Kokkos::View<part_t*, device_t> info;
  Kokkos::View<scalar_t *, device_t> cut_coordinates;
  bool bUniformWeights;
  scalar_t sEpsilon;
  Kokkos::View<index_t *, device_t> part_xadj;
  Kokkos::View<part_t*, device_t> view_num_partitioning_in_current_dim;
  Kokkos::View<part_t*, device_t> kokkos_my_incomplete_cut_count;
  
#ifndef TURN_OFF_MERGE_CHUNKS
  Kokkos::View<part_t*, device_t> kokkos_prefix_sum_num_cuts;
#endif

  KOKKOS_INLINE_FUNCTION
  ReduceWeightsFunctorInnerLoop(
#ifdef TURN_OFF_MERGE_CHUNKS
    part_t concurrent_current_part,
#endif
    part_t current_work_part,
    part_t current_concurrent_num_parts,
    Kokkos::View<index_t*, device_t> permutations,
    Kokkos::View<scalar_t *, device_t> coordinates,
    Kokkos::View<scalar_t**, device_t> weights,
    Kokkos::View<part_t*, device_t> parts,
    Kokkos::View<part_t*, device_t> info,
    Kokkos::View<scalar_t *, device_t> cut_coordinates,
    bool bUniformWeights,
    scalar_t sEpsilon,
    Kokkos::View<index_t *, device_t> part_xadj,
    Kokkos::View<part_t*, device_t> view_num_partitioning_in_current_dim,
    Kokkos::View<part_t *, device_t> kokkos_my_incomplete_cut_count
#ifndef TURN_OFF_MERGE_CHUNKS
    , Kokkos::View<part_t *, device_t> kokkos_prefix_sum_num_cuts
#endif
    ) :
#ifdef TURN_OFF_MERGE_CHUNKS
      concurrent_current_part(concurrent_current_part),
#endif
      current_work_part(current_work_part),
      current_concurrent_num_parts(current_concurrent_num_parts),
      permutations(permutations),
      coordinates(coordinates),
      weights(weights),
      parts(parts),
      info(info),
      cut_coordinates(cut_coordinates),
      bUniformWeights(bUniformWeights),
      sEpsilon(sEpsilon),
      part_xadj(part_xadj),
      view_num_partitioning_in_current_dim(
        view_num_partitioning_in_current_dim),
      kokkos_my_incomplete_cut_count(kokkos_my_incomplete_cut_count)
#ifndef TURN_OFF_MERGE_CHUNKS
      , kokkos_prefix_sum_num_cuts(kokkos_prefix_sum_num_cuts)
#endif
  {
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const size_t ii, ArrayType<scalar_t>& threadSum) const {
  
    int i = permutations(ii);
    scalar_t coord = coordinates(i);
    scalar_t w = bUniformWeights ? 1 : weights(i,0);

#ifdef TURN_OFF_MERGE_CHUNKS
    part_t total_part_shift = 0;
    part_t concurrent_cut_shifts = 0;
#else
    part_t concurrent_current_part = info(i);
    int kk = concurrent_current_part - current_work_part;

    if(kokkos_my_incomplete_cut_count(kk) > 0) {
    
      part_t concurrent_cut_shifts =
        kokkos_prefix_sum_num_cuts(kk);
        
      part_t total_part_shift =
        concurrent_cut_shifts * 2 + kk;
#endif

      part_t num_cuts = view_num_partitioning_in_current_dim(
        concurrent_current_part) - 1;
      
      scalar_t b = -99999999.9; // TODO: Clean up bounds

      // now check each part and it's right cut
      for(index_t part = 0; part <= num_cuts; ++part) {
      
        scalar_t a = b;
        b = (part == num_cuts) ? 99999999.9 : // TODO: Clean up bounds
          cut_coordinates(concurrent_cut_shifts+part);

        if(coord >= a + sEpsilon && coord <= b - sEpsilon) {
          threadSum.ptr[total_part_shift+part*2] += w;
          parts(i) = part*2;
        }

        if(part != num_cuts) {
          if(coord < b + sEpsilon && coord > b - sEpsilon) {
            threadSum.ptr[total_part_shift+part*2+1] += w;
            parts(i) = part*2+1;
          }
        }        
      }
      
#ifndef TURN_OFF_MERGE_CHUNKS
    }
#endif

  }
};

template<class policy_t, class scalar_t, class part_t,
  class index_t, class device_t>
struct ReduceWeightsFunctor {
  typedef typename policy_t::member_type member_type;
  typedef Kokkos::View<scalar_t*> scalar_view_t;
  typedef scalar_t value_type[];

  bool bTest = false;

#ifdef TURN_OFF_MERGE_CHUNKS
  part_t concurrent_current_part;
#endif
  part_t current_work_part;
  part_t current_concurrent_num_parts;
  int value_count;
  Kokkos::View<index_t*, device_t> permutations;
  Kokkos::View<scalar_t *, device_t> coordinates;
  Kokkos::View<scalar_t**, device_t> weights;
  Kokkos::View<part_t*, device_t> parts;
  Kokkos::View<part_t*, device_t> info;
  Kokkos::View<scalar_t *, device_t> cut_coordinates;
  Kokkos::View<index_t *, device_t> part_xadj;
  Kokkos::View<bool*, device_t> uniform_weights;
  scalar_t sEpsilon;
  Kokkos::View<part_t*, device_t> view_num_partitioning_in_current_dim;
  Kokkos::View<part_t*, device_t> kokkos_my_incomplete_cut_count;
  
#ifndef TURN_OFF_MERGE_CHUNKS
  Kokkos::View<part_t*, device_t> kokkos_prefix_sum_num_cuts;
#endif

  ReduceWeightsFunctor(
#ifdef TURN_OFF_MERGE_CHUNKS
    part_t concurrent_current_part,
#endif
    part_t current_work_part,
    part_t current_concurrent_num_parts,
    const int & weight_array_size,
    Kokkos::View<index_t*, device_t> permutations,
    Kokkos::View<scalar_t *, device_t> coordinates,
    Kokkos::View<scalar_t**, device_t> weights,
    Kokkos::View<part_t*, device_t> parts,
    Kokkos::View<part_t*, device_t> info,
    Kokkos::View<scalar_t *, device_t> cut_coordinates,
    Kokkos::View<index_t *, device_t> part_xadj,
    Kokkos::View<bool*, device_t> uniform_weights,
    scalar_t sEpsilon,
    Kokkos::View<part_t*, device_t> view_num_partitioning_in_current_dim,
    Kokkos::View<part_t *, device_t> kokkos_my_incomplete_cut_count

#ifndef TURN_OFF_MERGE_CHUNKS
    , Kokkos::View<part_t *, device_t> kokkos_prefix_sum_num_cuts
#endif
    ) :
#ifdef TURN_OFF_MERGE_CHUNKS
      concurrent_current_part(concurrent_current_part),
#endif
      current_work_part(current_work_part),
      current_concurrent_num_parts(current_concurrent_num_parts),
      value_count(weight_array_size),
      permutations(permutations),
      coordinates(coordinates),
      weights(weights),
      parts(parts),
      info(info),
      cut_coordinates(cut_coordinates),
      part_xadj(part_xadj),
      uniform_weights(uniform_weights),
      sEpsilon(sEpsilon),
      view_num_partitioning_in_current_dim(view_num_partitioning_in_current_dim),
      kokkos_my_incomplete_cut_count(kokkos_my_incomplete_cut_count)
#ifndef TURN_OFF_MERGE_CHUNKS
      , kokkos_prefix_sum_num_cuts(kokkos_prefix_sum_num_cuts)
#endif
  {
  }

  size_t team_shmem_size (int team_size) const {
    return sizeof(scalar_t) * value_count * team_size;
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const member_type & teamMember, value_type teamSum) const {
    bool bUniformWeights = uniform_weights(0);

#ifdef TURN_OFF_MERGE_CHUNKS
    index_t all_begin = (concurrent_current_part == 0) ? 0 :
      part_xadj(concurrent_current_part - 1);
    index_t all_end = part_xadj(concurrent_current_part);
#else
    index_t all_begin = (current_work_part == 0) ? 0 :
      part_xadj(current_work_part-1);
    index_t all_end = part_xadj(
      current_work_part + current_concurrent_num_parts - 1);
#endif

    index_t num_working_points = all_end - all_begin;
    int num_teams = teamMember.league_size();
    
    index_t stride = num_working_points / num_teams;
    if((num_working_points % num_teams) > 0) {
      stride += 1; // make sure we have coverage for the final points
    }
        
    index_t begin = all_begin + stride * teamMember.league_rank();
    index_t end = begin + stride;
    if(end > all_end) {
      end = all_end; // the last team may have less work than the other teams
    }

#ifndef TURN_OFF_MERGE_CHUNKS
    Kokkos::parallel_for(
      Kokkos::TeamThreadRange(teamMember, current_concurrent_num_parts),
      [=] (const int & kk) {
      part_t sum_num_cuts = 0;
      for(int kk2 = 0; kk2 < kk; ++kk2) {
        part_t num_parts =
          view_num_partitioning_in_current_dim(current_work_part + kk2);
        sum_num_cuts += num_parts - 1;
      }
      kokkos_prefix_sum_num_cuts(kk) = sum_num_cuts;      
    });
    
    Kokkos::parallel_for(
      Kokkos::TeamThreadRange(teamMember, begin, end),
      [=] (const int & ii) {
      int i = permutations(ii);
      
      for(int kk = 0; kk < current_concurrent_num_parts; ++kk) {
        auto current_concurrent_work_part = current_work_part + kk;
        if(ii >= ((current_concurrent_work_part == 0) ? 0 : part_xadj(current_concurrent_work_part-1)) && ii < part_xadj(current_concurrent_work_part)) {
          info(i) = current_concurrent_work_part;
          break;
        }
      }
    });
#endif

    // create the team shared data - each thread gets one of the arrays
    scalar_t * shared_ptr = (scalar_t *) teamMember.team_shmem().get_shmem(
      sizeof(scalar_t) * value_count * teamMember.team_size());

    // select the array for this thread
    ArrayType<scalar_t>
      array(&shared_ptr[teamMember.team_rank() * value_count]);

    // create reducer which handles the ArrayType class
    ArraySumReducer<policy_t, scalar_t, part_t> arraySumReducer(
      array, value_count);

    if(!bTest) {
        // call the reduce
        ReduceWeightsFunctorInnerLoop<scalar_t, part_t,
          index_t, device_t> inner_functor(
#ifdef TURN_OFF_MERGE_CHUNKS
          concurrent_current_part,
#endif
          current_work_part,
          current_concurrent_num_parts,
          permutations,
          coordinates,
          weights,
          parts,
          info,
          cut_coordinates,
          bUniformWeights,
          sEpsilon,
          part_xadj,
          view_num_partitioning_in_current_dim,
          kokkos_my_incomplete_cut_count
#ifndef TURN_OFF_MERGE_CHUNKS
          ,kokkos_prefix_sum_num_cuts
#endif
          );

        Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(teamMember, begin, end),
          inner_functor, arraySumReducer);
    }

    teamMember.team_barrier();

    // collect all the team's results
    Kokkos::single(Kokkos::PerTeam(teamMember), [=] () {
      for(int n = 0; n < value_count; ++n) {
        teamSum[n] += array.ptr[n];
      }
    });
  }

  KOKKOS_INLINE_FUNCTION
  void join(value_type dst, const value_type src)  const {
    for(int n = 0; n < value_count; ++n) {
      dst[n] += src[n];
    }
  }

  KOKKOS_INLINE_FUNCTION
  void join (volatile value_type dst, const volatile value_type src) const {
    for(int n = 0; n < value_count; ++n) {
      dst[n] += src[n];
    }
  }

  KOKKOS_INLINE_FUNCTION void init (value_type dst) const {
    for(int n = 0; n < value_count; ++n) {
      dst[n] = 0;
    }
  }
};

// store results from a run
struct Result {
  int time_us;        // time in ms for actual partition
  int n_teams;        // how many teams we ran with
  int n_points;       // how many points we ran with
  int n_parts;        // how many parts we ran with
};

// main will call this simple_model test for varies n_teams
Result run_test(int n_teams, int n_points, int n_parts) {

  typedef double scalar_t;
  typedef int part_t;
  typedef int lno_t;
  typedef Kokkos::Cuda device_t;
  typedef Kokkos::Cuda execution_t;

  int n_cuts = n_parts - 1;

#ifdef TURN_OFF_MERGE_CHUNKS
  int kk = 0; // in real test we loop 
#endif

  // dummy values
  part_t current_work_part = 0;
  part_t current_concurrent_num_parts = 1;
  scalar_t sEpsilon = 0.0000001;

  Kokkos::View<lno_t*, device_t> kokkos_coordinate_permutations
    ("kokkos_coordinate_permutations", n_points);
  Kokkos::View<scalar_t *, device_t> kokkos_current_dim_coords
    ("kokkos_current_dim_coords", n_points);
  Kokkos::View<scalar_t**, device_t> kokkos_weights
    ("kokkos_weights", n_points, 1);
  Kokkos::View<part_t*, device_t> kokkos_assigned_part_ids
    ("kokkos_assigned_part_ids", n_points);
  Kokkos::View<part_t*, device_t> kokkos_info
    ("kokkos_info", n_points);
    
  Kokkos::parallel_for(n_points, KOKKOS_LAMBDA (const int i) {
      kokkos_coordinate_permutations(i) = i;
      kokkos_current_dim_coords(i) = (scalar_t) i / (scalar_t) (n_points-1); // scale 0-1
      kokkos_info(i) = -1; // currently not used
      kokkos_info(i,0) = 1.0;
  });
  
  Kokkos::View<scalar_t *, device_t> local_kokkos_temp_cut_coords
    ("local_kokkos_temp_cut_coords", n_cuts);
  Kokkos::parallel_for(n_cuts, KOKKOS_LAMBDA (const int i) {
    // divide 0-1 range into n_cuts+1 partitions so 1st cut > 0 and last cut < 1.0
    local_kokkos_temp_cut_coords(i) = (scalar_t) (i + 1) / (scalar_t) (n_cuts + 1);
  });
  
  Kokkos::View<lno_t *, device_t> kokkos_part_xadj
    ("kokkos_part_xadj", current_concurrent_num_parts);
  Kokkos::parallel_for(kokkos_part_xadj.size(), KOKKOS_LAMBDA (const int i) {
      kokkos_part_xadj(i) = n_points;
  });
  
  Kokkos::View<bool*, device_t> kokkos_uniform_weights
    ("kokkos_uniform_weights", 1); // currently just read 0 slot
  Kokkos::parallel_for(1, KOKKOS_LAMBDA (const int i) {
      kokkos_uniform_weights(0) = false;
  });
  
  Kokkos::View<part_t*, device_t> view_num_partitioning_in_current_dim
    ("view_num_partitioning_in_current_dim", current_concurrent_num_parts);
  Kokkos::parallel_for(current_concurrent_num_parts, KOKKOS_LAMBDA (const int i) {
      view_num_partitioning_in_current_dim(i) = n_parts; // in simple model we just set each the same
  });
  
  Kokkos::View<part_t *, device_t> local_kokkos_my_incomplete_cut_count
    ("local_kokkos_my_incomplete_cut_count", n_cuts);
  Kokkos::parallel_for(n_cuts, KOKKOS_LAMBDA (const int i) {
      local_kokkos_my_incomplete_cut_count(i) = 1;
  });
  
#ifndef TURN_OFF_MERGE_CHUNKS
  Kokkos::View<part_t *, device_t> kokkos_prefix_sum_num_cuts
    ("kokkos_prefix_sum_num_cuts", current_concurrent_num_parts, 0);
#endif

  // start clock
  typedef std::chrono::high_resolution_clock Clock;
  auto clock_start = Clock::now();

  typedef Kokkos::TeamPolicy<execution_t> policy_t;
  
  auto policy_ReduceWeightsFunctor =
    policy_t(n_teams, Kokkos::AUTO);

  int array_length = n_parts + n_cuts;
  scalar_t * part_weights = new scalar_t[array_length];

  ReduceWeightsFunctor<policy_t, scalar_t, part_t, lno_t, device_t>
    teamFunctor(
#ifdef TURN_OFF_MERGE_CHUNKS
      current_work_part + kk,
#endif
      current_work_part,
      current_concurrent_num_parts,
      array_length,
      kokkos_coordinate_permutations,
      kokkos_current_dim_coords,
      kokkos_weights,
      kokkos_assigned_part_ids,
      kokkos_info,
      local_kokkos_temp_cut_coords,
      kokkos_part_xadj,
      kokkos_uniform_weights,
      sEpsilon,
      view_num_partitioning_in_current_dim,
      local_kokkos_my_incomplete_cut_count
#ifndef TURN_OFF_MERGE_CHUNKS
      ,kokkos_prefix_sum_num_cuts
#endif
      );

  Kokkos::parallel_reduce(policy_ReduceWeightsFunctor,
    teamFunctor, part_weights);
  
  delete [] part_weights;

  // set the output values
  Result result;
  result.time_us = static_cast<int>(std::chrono::duration_cast<
    std::chrono::microseconds>(Clock::now() - clock_start).count());
  result.n_teams = n_teams;
  result.n_points = n_points;
  result.n_parts = n_parts;
  return result;

}

int main( int argc, char* argv[] )
{
  Kokkos::ScopeGuard kokkosScope(argc, argv); 

  int n_parts = 2;
  int n_points = 200000;
  
  std::vector<Result> results; // store the results for each run
  for(int n_teams = 1; n_teams <= n_points; n_teams *=2) {
    Result result = run_test(n_teams, n_points, n_parts);
    results.push_back(result); // add to vector for logging at end
  }

  // now loop and log each result - shows how n_teams impacts total time
  for(auto&& result : results) {
    printf("teams: %8d   time: %d us\n",
      result.n_teams, result.time_us);
  }
}
