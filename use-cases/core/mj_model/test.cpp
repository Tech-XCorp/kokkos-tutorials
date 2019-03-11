//*
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

#include <Kokkos_Core.hpp>
#include <vector>

template<class scalar_t>
struct ArrayType {
  scalar_t * ptr;
  KOKKOS_INLINE_FUNCTION
  ArrayType(scalar_t * pSetPtr) : ptr(pSetPtr) {};
};

template<class policy_t, class scalar_t, class part_t>
struct ArrayCombinationReducer {

  typedef ArrayCombinationReducer reducer;
  typedef ArrayType<scalar_t> value_type;
  scalar_t max_scalar;
  value_type * value;
  size_t value_count_rightleft;
  size_t value_count_weights;

  KOKKOS_INLINE_FUNCTION ArrayCombinationReducer(
    scalar_t mj_max_scalar,
    value_type &val,
    const size_t & mj_value_count_rightleft,
    const size_t & mj_value_count_weights) :
      max_scalar(mj_max_scalar),
      value(&val),
      value_count_rightleft(mj_value_count_rightleft),
      value_count_weights(mj_value_count_weights)
  {}

  KOKKOS_INLINE_FUNCTION
  value_type& reference() const {
    return *value;
  }

  KOKKOS_INLINE_FUNCTION
  void join(value_type& dst, const value_type& src)  const {
    for(int n = 0; n < value_count_weights; ++n) {
      dst.ptr[n] += src.ptr[n];
    }

    for(int n = value_count_weights + 2; n < value_count_weights + value_count_rightleft - 2; n += 2) {
      if(src.ptr[n] > dst.ptr[n]) {
        dst.ptr[n] = src.ptr[n];
      }
      if(src.ptr[n+1] < dst.ptr[n+1]) {
        dst.ptr[n+1] = src.ptr[n+1];
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void join (volatile value_type& dst, const volatile value_type& src) const {
    for(int n = 0; n < value_count_weights; ++n) {
      dst.ptr[n] += src.ptr[n];
    }

    for(int n = value_count_weights + 2; n < value_count_weights + value_count_rightleft - 2; n += 2) {
      if(src.ptr[n] > dst.ptr[n]) {
        dst.ptr[n] = src.ptr[n];
      }
      if(src.ptr[n+1] < dst.ptr[n+1]) {
        dst.ptr[n+1] = src.ptr[n+1];
      }
    }
  }

  KOKKOS_INLINE_FUNCTION void init (value_type& dst) const {
    for(int n = 0; n < value_count_weights; ++n) {
      dst.ptr[n] = 0;
    }
    
    for(int n = value_count_weights + 2; n < value_count_weights + value_count_rightleft - 2; n += 2) {
      dst.ptr[n]   = -max_scalar;
      dst.ptr[n+1] =  max_scalar;
    }
  }
};

template<class policy_t, class scalar_t, class part_t, class index_t,
  class device_t>
struct ReduceWeightsFunctorMerge {
  typedef typename policy_t::member_type member_type;
  typedef Kokkos::View<scalar_t*> scalar_view_t;
  typedef scalar_t value_type[];
  scalar_t max_scalar;
  
#ifdef TURN_OFF_MERGE_CHUNKS
  part_t concurrent_current_part;
  part_t num_cuts;
#endif
  part_t current_work_part;
  part_t current_concurrent_num_parts;
  int value_count_rightleft;
  int value_count_weights;
  int value_count;
  Kokkos::View<index_t*, device_t> permutations;
  Kokkos::View<scalar_t *, device_t> coordinates;
  Kokkos::View<scalar_t**, device_t> weights;
  Kokkos::View<part_t*, device_t> parts;
#ifndef TURN_OFF_MERGE_CHUNKS
  Kokkos::View<part_t*, device_t> info;
#endif
  Kokkos::View<scalar_t *, device_t> cut_coordinates;
  Kokkos::View<index_t *, device_t> part_xadj;
  Kokkos::View<bool*, device_t> uniform_weights;
  scalar_t sEpsilon;
  Kokkos::View<part_t*, device_t> view_num_partitioning_in_current_dim;
  Kokkos::View<part_t*, device_t> my_incomplete_cut_count;
  
#ifndef TURN_OFF_MERGE_CHUNKS
  Kokkos::View<part_t*, device_t> prefix_sum_num_cuts;
#endif

  ReduceWeightsFunctorMerge(
    scalar_t mj_max_scalar,
#ifdef TURN_OFF_MERGE_CHUNKS
    part_t mj_concurrent_current_part,
    part_t mj_num_cuts,
#endif
    part_t mj_current_work_part,
    part_t mj_current_concurrent_num_parts,
    part_t mj_left_right_array_size,
    part_t mj_weight_array_size,
    Kokkos::View<index_t*, device_t> mj_permutations,
    Kokkos::View<scalar_t *, device_t> mj_coordinates,
    Kokkos::View<scalar_t**, device_t> mj_weights,
    Kokkos::View<part_t*, device_t> mj_parts,
#ifndef TURN_OFF_MERGE_CHUNKS
    Kokkos::View<part_t*, device_t> mj_info,
#endif
    Kokkos::View<scalar_t *, device_t> mj_cut_coordinates,
    Kokkos::View<index_t *, device_t> mj_part_xadj,
    Kokkos::View<bool*, device_t> mj_uniform_weights,
    scalar_t mj_sEpsilon,
    Kokkos::View<part_t*, device_t> mj_view_num_partitioning_in_current_dim,
    Kokkos::View<part_t *, device_t> mj_my_incomplete_cut_count

#ifndef TURN_OFF_MERGE_CHUNKS
    , Kokkos::View<part_t *, device_t> mj_prefix_sum_num_cuts
#endif
    ) :
      max_scalar(mj_max_scalar),
#ifdef TURN_OFF_MERGE_CHUNKS
      concurrent_current_part(mj_concurrent_current_part),
      num_cuts(mj_num_cuts),
#endif
      current_work_part(mj_current_work_part),
      current_concurrent_num_parts(mj_current_concurrent_num_parts),
      value_count_rightleft(mj_left_right_array_size), 
      value_count_weights(mj_weight_array_size),
      value_count(mj_weight_array_size+mj_left_right_array_size),
      permutations(mj_permutations),
      coordinates(mj_coordinates),
      weights(mj_weights),
      parts(mj_parts),
#ifndef TURN_OFF_MERGE_CHUNKS
      info(mj_info),
#endif
      cut_coordinates(mj_cut_coordinates),
      part_xadj(mj_part_xadj),
      uniform_weights(mj_uniform_weights),
      sEpsilon(mj_sEpsilon),
      view_num_partitioning_in_current_dim(mj_view_num_partitioning_in_current_dim),
      my_incomplete_cut_count(mj_my_incomplete_cut_count)
#ifndef TURN_OFF_MERGE_CHUNKS
      ,prefix_sum_num_cuts(mj_prefix_sum_num_cuts)
#endif
  {
  }

  size_t team_shmem_size (int team_size) const {
    return sizeof(scalar_t) * (value_count_weights + value_count_rightleft)
      * team_size;
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
      prefix_sum_num_cuts(kk) = sum_num_cuts;      
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
    size_t sh_mem_size = sizeof(scalar_t) * (value_count_weights + value_count_rightleft) * teamMember.team_size();

    scalar_t * shared_ptr = (scalar_t *) teamMember.team_shmem().get_shmem(
      sh_mem_size);

    // select the array for this thread
    ArrayType<scalar_t>
      array(&shared_ptr[teamMember.team_rank() * (value_count_weights + value_count_rightleft)]);

    // create reducer which handles the ArrayType class
    ArrayCombinationReducer<policy_t, scalar_t, part_t> arraySumReducer(
      max_scalar, array,
      value_count_rightleft,
      value_count_weights);

    Kokkos::parallel_reduce(
      Kokkos::TeamThreadRange(teamMember, begin, end),
      [=] (const size_t ii, ArrayType<scalar_t>& threadSum) {
      
      
      int i = permutations(ii);
      scalar_t coord = coordinates(i);
      scalar_t w = bUniformWeights ? 1 : weights(i,0);

  #ifndef TURN_OFF_MERGE_CHUNKS
      part_t concurrent_current_part = info(i);
      int kk = concurrent_current_part - current_work_part;

      if(my_incomplete_cut_count(kk) > 0) {
      
        part_t concurrent_cut_shifts =
          prefix_sum_num_cuts(kk);
          
        part_t total_part_shift =
          concurrent_cut_shifts * 2 + kk;
          
        part_t num_cuts = view_num_partitioning_in_current_dim(
          concurrent_current_part) - 1;

  #endif

        scalar_t b = -max_scalar;

        // for the left/right closest part calculation
  #ifdef TURN_OFF_MERGE_CHUNKS
        scalar_t * p1 = &threadSum.ptr[value_count_weights + 2];
  #else
        scalar_t * p1 = &threadSum.ptr[value_count_weights + (concurrent_cut_shifts * 2) + kk * 4 + 2];
  #endif

        // now check each part and it's right cut
        for(index_t part = 0; part <= num_cuts; ++part) {
        
          scalar_t a = b;
          b = (part == num_cuts) ? max_scalar :
  #ifdef TURN_OFF_MERGE_CHUNKS
            cut_coordinates(part);
  #else
            cut_coordinates(concurrent_cut_shifts+part);
  #endif

          if(coord >= a + sEpsilon && coord <= b - sEpsilon) {
  #ifdef TURN_OFF_MERGE_CHUNKS
            threadSum.ptr[part*2] += w;
  #else
            threadSum.ptr[total_part_shift+part*2] += w;
  #endif
            parts(i) = part*2;
          }

          if(part != num_cuts) {
            if(coord < b + sEpsilon && coord > b - sEpsilon) {
  #ifdef TURN_OFF_MERGE_CHUNKS
              threadSum.ptr[part*2+1] += w;
  #else
              threadSum.ptr[total_part_shift+part*2+1] += w;
  #endif
              parts(i) = part*2+1;
            }

            // now handle the left/right closest part
            if(coord > b && coord < *(p1+1)) {
              *(p1+1) = coord;
            }
            if(coord < b && coord > *p1) {
              *p1 = coord;
            }
            p1 += 2;
          }        
        }
        
  #ifndef TURN_OFF_MERGE_CHUNKS
      }
  #endif      
    }, arraySumReducer);

    teamMember.team_barrier();

    // collect all the team's results
    Kokkos::single(Kokkos::PerTeam(teamMember), [=] () {
      for(int n = 0; n < value_count_weights; ++n) {
        teamSum[n] += array.ptr[n];
      }
      
      for(int n = 2 + value_count_weights; n < value_count_weights + value_count_rightleft - 2; n += 2) {
        if(array.ptr[n] > teamSum[n]) {
          teamSum[n] = array.ptr[n];
        }
        if(array.ptr[n+1] < teamSum[n+1]) {
          teamSum[n+1] = array.ptr[n+1];
        }
      }
    
    });
  }
  
  KOKKOS_INLINE_FUNCTION
  void join(value_type dst, const value_type src)  const {
    for(int n = 0; n < value_count_weights; ++n) {
      dst[n] += src[n];
    }

    for(int n = value_count_weights + 2; n < value_count_weights + value_count_rightleft - 2; n += 2) {
      if(src[n] > dst[n]) {
        dst[n] = src[n];
      }
      if(src[n+1] < dst[n+1]) {
        dst[n+1] = src[n+1];
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void join (volatile value_type dst, const volatile value_type src) const {
    for(int n = 0; n < value_count_weights; ++n) {
      dst[n] += src[n];
    }

    for(int n = value_count_weights + 2; n < value_count_weights + value_count_rightleft - 2; n += 2) {
      if(src[n] > dst[n]) {
        dst[n] = src[n];
      }
      if(src[n+1] < dst[n+1]) {
        dst[n+1] = src[n+1];
      }
    }
  }

  KOKKOS_INLINE_FUNCTION void init (value_type dst) const {
    for(int n = 0; n < value_count_weights; ++n) {
      dst[n] = 0;
    }
    
    for(int n = value_count_weights; n < value_count_weights + value_count_rightleft; n += 2) {
      dst[n]   = -max_scalar;
      dst[n+1] =  max_scalar;
    }
  }
};

#define TURN_OFF_MERGE_CHUNKS // for debugging - will be removed
template<class policy_t, class scalar_t, class part_t, class index_t,
  class device_t>
struct ReduceWeightsFunctor {
  typedef typename policy_t::member_type member_type;
  typedef Kokkos::View<scalar_t*> scalar_view_t;
  typedef scalar_t value_type[];
  scalar_t max_scalar;
  
#ifdef TURN_OFF_MERGE_CHUNKS
  part_t concurrent_current_part;
  part_t num_cuts;
#endif
  part_t current_work_part;
  part_t current_concurrent_num_parts;
  int value_count_rightleft;
  int value_count_weights;
  int value_count;
  Kokkos::View<index_t*, device_t> permutations;
  Kokkos::View<scalar_t *, device_t> coordinates;
  Kokkos::View<scalar_t**, device_t> weights;
  Kokkos::View<part_t*, device_t> parts;
#ifndef TURN_OFF_MERGE_CHUNKS
  Kokkos::View<part_t*, device_t> info;
#endif
  Kokkos::View<scalar_t *, device_t> cut_coordinates;
  Kokkos::View<index_t *, device_t> part_xadj;
  Kokkos::View<bool*, device_t> uniform_weights;
  scalar_t sEpsilon;
  Kokkos::View<part_t*, device_t> view_num_partitioning_in_current_dim;
  Kokkos::View<part_t*, device_t> my_incomplete_cut_count;
  
#ifndef TURN_OFF_MERGE_CHUNKS
  Kokkos::View<part_t*, device_t> prefix_sum_num_cuts;
#endif

  ReduceWeightsFunctor(
    scalar_t mj_max_scalar,
#ifdef TURN_OFF_MERGE_CHUNKS
    part_t mj_concurrent_current_part,
    part_t mj_num_cuts,
#endif
    part_t mj_current_work_part,
    part_t mj_current_concurrent_num_parts,
    part_t mj_left_right_array_size,
    part_t mj_weight_array_size,
    Kokkos::View<index_t*, device_t> mj_permutations,
    Kokkos::View<scalar_t *, device_t> mj_coordinates,
    Kokkos::View<scalar_t**, device_t> mj_weights,
    Kokkos::View<part_t*, device_t> mj_parts,
#ifndef TURN_OFF_MERGE_CHUNKS
    Kokkos::View<part_t*, device_t> mj_info,
#endif
    Kokkos::View<scalar_t *, device_t> mj_cut_coordinates,
    Kokkos::View<index_t *, device_t> mj_part_xadj,
    Kokkos::View<bool*, device_t> mj_uniform_weights,
    scalar_t mj_sEpsilon,
    Kokkos::View<part_t*, device_t> mj_view_num_partitioning_in_current_dim,
    Kokkos::View<part_t *, device_t> mj_my_incomplete_cut_count

#ifndef TURN_OFF_MERGE_CHUNKS
    , Kokkos::View<part_t *, device_t> mj_prefix_sum_num_cuts
#endif
    ) :
      max_scalar(mj_max_scalar),
#ifdef TURN_OFF_MERGE_CHUNKS
      concurrent_current_part(mj_concurrent_current_part),
      num_cuts(mj_num_cuts),
#endif
      current_work_part(mj_current_work_part),
      current_concurrent_num_parts(mj_current_concurrent_num_parts),
      value_count_rightleft(mj_left_right_array_size), 
      value_count_weights(mj_weight_array_size),
      value_count(mj_weight_array_size+mj_left_right_array_size),
      permutations(mj_permutations),
      coordinates(mj_coordinates),
      weights(mj_weights),
      parts(mj_parts),
#ifndef TURN_OFF_MERGE_CHUNKS
      info(mj_info),
#endif
      cut_coordinates(mj_cut_coordinates),
      part_xadj(mj_part_xadj),
      uniform_weights(mj_uniform_weights),
      sEpsilon(mj_sEpsilon),
      view_num_partitioning_in_current_dim(mj_view_num_partitioning_in_current_dim),
      my_incomplete_cut_count(mj_my_incomplete_cut_count)
#ifndef TURN_OFF_MERGE_CHUNKS
      ,prefix_sum_num_cuts(mj_prefix_sum_num_cuts)
#endif
  {
  }

  size_t team_shmem_size (int team_size) const {
    return sizeof(scalar_t) * (value_count_weights + value_count_rightleft)
      * team_size;
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
      prefix_sum_num_cuts(kk) = sum_num_cuts;      
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
    size_t sh_mem_size = sizeof(scalar_t) * (value_count_weights + value_count_rightleft) * teamMember.team_size();

    scalar_t * shared_ptr = (scalar_t *) teamMember.team_shmem().get_shmem(
      sh_mem_size);

    // select the array for this thread
    ArrayType<scalar_t>
      array(&shared_ptr[teamMember.team_rank() * (value_count_weights + value_count_rightleft)]);

    // create reducer which handles the ArrayType class
    ArrayCombinationReducer<policy_t, scalar_t, part_t> arraySumReducer(
      max_scalar, array,
      value_count_rightleft,
      value_count_weights);

    Kokkos::parallel_reduce(
      Kokkos::TeamThreadRange(teamMember, begin, end),
      [=] (const size_t ii, ArrayType<scalar_t>& threadSum) {
      
      
      int i = permutations(ii);
      scalar_t coord = coordinates(i);
      scalar_t w = bUniformWeights ? 1 : weights(i,0);

  #ifndef TURN_OFF_MERGE_CHUNKS
      part_t concurrent_current_part = info(i);
      int kk = concurrent_current_part - current_work_part;

      if(my_incomplete_cut_count(kk) > 0) {
      
        part_t concurrent_cut_shifts =
          prefix_sum_num_cuts(kk);
          
        part_t total_part_shift =
          concurrent_cut_shifts * 2 + kk;
          
        part_t num_cuts = view_num_partitioning_in_current_dim(
          concurrent_current_part) - 1;

  #endif

        scalar_t b = -max_scalar;

        // for the left/right closest part calculation
  #ifdef TURN_OFF_MERGE_CHUNKS
        scalar_t * p1 = &threadSum.ptr[value_count_weights + 2];
  #else
        scalar_t * p1 = &threadSum.ptr[value_count_weights + (concurrent_cut_shifts * 2) + kk * 4 + 2];
  #endif

        // now check each part and it's right cut
        for(index_t part = 0; part <= num_cuts; ++part) {
        
          scalar_t a = b;
          b = (part == num_cuts) ? max_scalar :
  #ifdef TURN_OFF_MERGE_CHUNKS
            cut_coordinates(part);
  #else
            cut_coordinates(concurrent_cut_shifts+part);
  #endif

          if(coord >= a + sEpsilon && coord <= b - sEpsilon) {
  #ifdef TURN_OFF_MERGE_CHUNKS
            threadSum.ptr[part*2] += w;
  #else
            threadSum.ptr[total_part_shift+part*2] += w;
  #endif
            parts(i) = part*2;
          }

          if(part != num_cuts) {
            if(coord < b + sEpsilon && coord > b - sEpsilon) {
  #ifdef TURN_OFF_MERGE_CHUNKS
              threadSum.ptr[part*2+1] += w;
  #else
              threadSum.ptr[total_part_shift+part*2+1] += w;
  #endif
              parts(i) = part*2+1;
            }

            // now handle the left/right closest part
            if(coord > b && coord < *(p1+1)) {
              *(p1+1) = coord;
            }
            if(coord < b && coord > *p1) {
              *p1 = coord;
            }
            p1 += 2;
          }        
        }
        
  #ifndef TURN_OFF_MERGE_CHUNKS
      }
  #endif      
    }, arraySumReducer);

    teamMember.team_barrier();

    // collect all the team's results
    Kokkos::single(Kokkos::PerTeam(teamMember), [=] () {
      for(int n = 0; n < value_count_weights; ++n) {
        teamSum[n] += array.ptr[n];
      }
      
      for(int n = 2 + value_count_weights; n < value_count_weights + value_count_rightleft - 2; n += 2) {
        if(array.ptr[n] > teamSum[n]) {
          teamSum[n] = array.ptr[n];
        }
        if(array.ptr[n+1] < teamSum[n+1]) {
          teamSum[n+1] = array.ptr[n+1];
        }
      }
    
    });
  }
  
  KOKKOS_INLINE_FUNCTION
  void join(value_type dst, const value_type src)  const {
    for(int n = 0; n < value_count_weights; ++n) {
      dst[n] += src[n];
    }

    for(int n = value_count_weights + 2; n < value_count_weights + value_count_rightleft - 2; n += 2) {
      if(src[n] > dst[n]) {
        dst[n] = src[n];
      }
      if(src[n+1] < dst[n+1]) {
        dst[n+1] = src[n+1];
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void join (volatile value_type dst, const volatile value_type src) const {
    for(int n = 0; n < value_count_weights; ++n) {
      dst[n] += src[n];
    }

    for(int n = value_count_weights + 2; n < value_count_weights + value_count_rightleft - 2; n += 2) {
      if(src[n] > dst[n]) {
        dst[n] = src[n];
      }
      if(src[n+1] < dst[n+1]) {
        dst[n+1] = src[n+1];
      }
    }
  }

  KOKKOS_INLINE_FUNCTION void init (value_type dst) const {
    for(int n = 0; n < value_count_weights; ++n) {
      dst[n] = 0;
    }
    
    for(int n = value_count_weights; n < value_count_weights + value_count_rightleft; n += 2) {
      dst[n]   = -max_scalar;
      dst[n+1] =  max_scalar;
    }
  }
};
#undef TURN_OFF_MERGE_CHUNKS

// store results from a run
struct Result {
  int time_basic_us;        // time in ms for actual partition
  int time_merge_us;        // time in ms for actual partition
  int num_teams;        // how many teams we ran with
  int num_points;       // how many points we ran with
  int num_parts;        // how many parts we ran with
};

// main will call this simple_model test for varies num_teams
Result run_test(int num_teams, int num_points, int num_parts) {

  // set the output values
  Result result;
  result.num_teams = num_teams;
  result.num_points = num_points;
  result.num_parts = num_parts;
  
  for(int runLoop = 0; runLoop <= 1; ++runLoop) {

  typedef double mj_scalar_t;
  typedef int mj_part_t;
  typedef int mj_lno_t;
  typedef Kokkos::Cuda device_t;
  typedef Kokkos::Cuda execution_t;

  int num_cuts = num_parts - 1;

  int kk = 0; // in real test we loop - for TURN_OFF_MERGE_CHUNKS only

  // dummy values
  mj_part_t current_work_part = 0;
  mj_part_t current_concurrent_num_parts = 1;
  mj_scalar_t sEpsilon = 0.0000001;

  Kokkos::View<mj_lno_t*, device_t> coordinate_permutations
    ("coordinate_permutations", num_points);
  Kokkos::View<mj_scalar_t *, device_t> mj_current_dim_coords
    ("mj_current_dim_coords", num_points);
  Kokkos::View<mj_scalar_t**, device_t> mj_weights
    ("mj_weights", num_points, 1);
  Kokkos::View<mj_part_t*, device_t> assigned_part_ids
    ("assigned_part_ids", num_points);
  Kokkos::View<mj_part_t*, device_t> info
    ("info", num_points);
    
  Kokkos::parallel_for(num_points, KOKKOS_LAMBDA (const int i) {
      coordinate_permutations(i) = i;
      mj_current_dim_coords(i) = (mj_scalar_t) i / (mj_scalar_t) (num_points-1); // scale 0-1
      info(i) = -1; // currently not used
      info(i,0) = 1.0;
  });
  
  Kokkos::View<mj_scalar_t *, device_t> local_temp_cut_coords
    ("local_temp_cut_coords", num_cuts);
  Kokkos::parallel_for(num_cuts, KOKKOS_LAMBDA (const int i) {
    // divide 0-1 range into num_cuts+1 partitions so 1st cut > 0 and last cut < 1.0
    local_temp_cut_coords(i) = (mj_scalar_t) (i + 1) / (mj_scalar_t) (num_cuts + 1);
  });
  
  Kokkos::View<mj_lno_t *, device_t> part_xadj
    ("part_xadj", current_concurrent_num_parts);
  Kokkos::parallel_for(part_xadj.size(), KOKKOS_LAMBDA (const int i) {
      part_xadj(i) = num_points;
  });
  
  Kokkos::View<bool*, device_t> mj_uniform_weights
    ("mj_uniform_weights", 1); // currently just read 0 slot
  Kokkos::parallel_for(1, KOKKOS_LAMBDA (const int i) {
      mj_uniform_weights(0) = false;
  });
  
  Kokkos::View<mj_part_t*, device_t> view_num_partitioning_in_current_dim
    ("view_num_partitioning_in_current_dim", current_concurrent_num_parts);
  Kokkos::parallel_for(current_concurrent_num_parts, KOKKOS_LAMBDA (const int i) {
      view_num_partitioning_in_current_dim(i) = num_parts; // in simple model we just set each the same
  });
  
  Kokkos::View<mj_part_t *, device_t> local_my_incomplete_cut_count
    ("local_my_incomplete_cut_count", num_cuts);
  Kokkos::parallel_for(num_cuts, KOKKOS_LAMBDA (const int i) {
      local_my_incomplete_cut_count(i) = 1;
  });
  
  Kokkos::View<mj_part_t *, device_t> prefix_sum_num_cuts
    ("prefix_sum_num_cuts", current_concurrent_num_parts, 0);

  typedef Kokkos::TeamPolicy<execution_t> policy_t;
  
  auto policy_ReduceWeightsFunctor =
    policy_t(num_teams, Kokkos::AUTO);

  int weight_array_length = num_parts + num_cuts;
  int right_left_array_length = (2 + num_cuts) * 2;

  int total_array_length =
    weight_array_length + right_left_array_length;

  mj_scalar_t * reduce_array =
    new mj_scalar_t[static_cast<size_t>(total_array_length)];
    
  if(runLoop == 0) {
    // start clock for merge
    typedef std::chrono::high_resolution_clock Clock;
    auto clock_start = Clock::now();
    
    ReduceWeightsFunctorMerge<policy_t, mj_scalar_t, mj_part_t, mj_lno_t,
      device_t>
      teamFunctor(
        std::numeric_limits<mj_scalar_t>::max(),
        current_work_part,
        current_concurrent_num_parts,
        right_left_array_length,
        weight_array_length,
        coordinate_permutations,
        mj_current_dim_coords,
        mj_weights,
        assigned_part_ids,
        info,
        local_temp_cut_coords,
        part_xadj,
        mj_uniform_weights,
        sEpsilon,
        view_num_partitioning_in_current_dim,
        local_my_incomplete_cut_count
        ,prefix_sum_num_cuts
        );
    Kokkos::parallel_reduce(policy_ReduceWeightsFunctor,
      teamFunctor, reduce_array);

    result.time_merge_us = static_cast<int>(std::chrono::duration_cast<
      std::chrono::microseconds>(Clock::now() - clock_start).count());
  }
  else {
    // start clock for basic
    typedef std::chrono::high_resolution_clock Clock;
    auto clock_start = Clock::now();
    
    ReduceWeightsFunctor<policy_t, mj_scalar_t, mj_part_t, mj_lno_t,
      device_t>
      teamFunctor(
        std::numeric_limits<mj_scalar_t>::max(),
        current_work_part + kk,
        num_cuts,
        current_work_part,
        current_concurrent_num_parts,
        right_left_array_length,
        weight_array_length,
        coordinate_permutations,
        mj_current_dim_coords,
        mj_weights,
        assigned_part_ids,
        local_temp_cut_coords,
        part_xadj,
        mj_uniform_weights,
        sEpsilon,
        view_num_partitioning_in_current_dim,
        local_my_incomplete_cut_count
        );
    Kokkos::parallel_reduce(policy_ReduceWeightsFunctor,
      teamFunctor, reduce_array);

    result.time_basic_us = static_cast<int>(std::chrono::duration_cast<
      std::chrono::microseconds>(Clock::now() - clock_start).count());
  }
    
  delete [] reduce_array;

  }

  return result;
  
  

}

int main( int argc, char* argv[] )
{
  Kokkos::ScopeGuard kokkosScope(argc, argv); 

  int num_parts = 2;
  int num_points = 200000;
  
  std::vector<Result> results; // store the results for each run
  for(int num_teams = 1; num_teams <= num_points; num_teams *=2) {
    Result result = run_test(num_teams, num_points, num_parts);
    results.push_back(result); // add to vector for logging at end
  }
  // now loop and log each result - shows how num_teams impacts total time
  for(auto&& result : results) {
    printf("teams: %8d   basic: %d   merge: %d\n",
      result.num_teams, result.time_basic_us, result.time_merge_us);
  }
}
