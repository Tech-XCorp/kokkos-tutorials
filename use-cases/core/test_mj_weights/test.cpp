#include <Kokkos_Core.hpp>
#include <chrono>

#define NUM_PARTS 2

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

  Kokkos::View<index_t*, device_t> permutations;
  Kokkos::View<scalar_t *, device_t> coordinates;
  Kokkos::View<scalar_t**, device_t> weights;
  Kokkos::View<part_t*, device_t> parts;
  Kokkos::View<scalar_t *, device_t> cut_coordinates;
  bool bUniformWeights;
  scalar_t sEpsilon;
  int num_cuts;

  KOKKOS_INLINE_FUNCTION
  ReduceWeightsFunctorInnerLoop(
    Kokkos::View<index_t*, device_t> mj_permutations,
    Kokkos::View<scalar_t *, device_t> mj_coordinates,
    Kokkos::View<scalar_t**, device_t> mj_weights,
    Kokkos::View<part_t*, device_t> mj_parts,
    Kokkos::View<scalar_t *, device_t> mj_cut_coordinates,
    bool mj_bUniformWeights,
    scalar_t mj_sEpsilon,
    int mj_num_cuts
  ) :
    permutations(mj_permutations),
    coordinates(mj_coordinates),
    weights(mj_weights),
    parts(mj_parts),
    cut_coordinates(mj_cut_coordinates),
    bUniformWeights(mj_bUniformWeights),
    sEpsilon(mj_sEpsilon),
    num_cuts(mj_num_cuts)
  {

  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const size_t ii, ArrayType<scalar_t>& threadSum) const {
#ifdef kill
    int i = permutations(ii);
    scalar_t coord = coordinates(i);
    scalar_t w = 1.0; // bUniformWeights ? 1 : weights(i,0);

    // check part 0
    scalar_t b = cut_coordinates(0);
    if(coord <= b - sEpsilon) {
      threadSum.ptr[0] += w;
      parts(i) = 0;
    }

    // check cut 0
    if( coord < b + sEpsilon && coord > b - sEpsilon) {
      threadSum.ptr[1] += w;
      parts(i) = 1;
    }

    scalar_t a;
    // now check each part and it's right cut
    for(index_t part = 1; part < num_cuts; ++part) {
      a = b;
      b = cut_coordinates(part);

      if(coord < b + sEpsilon && coord > b - sEpsilon) {
        threadSum.ptr[part*2+1] += w;
        parts(i) = part*2+1;
      }

      if(coord >= a + sEpsilon && coord <= b - sEpsilon) {
        threadSum.ptr[part*2] += w;
        parts(i) = part*2;
      }
    }
    // check last part
    if(coord >= b + sEpsilon) {
      threadSum.ptr[num_cuts*2] += w;
      parts(i) = num_cuts*2;
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

  index_t n_points;
  int value_count;
  Kokkos::View<index_t*, device_t> permutations;
  Kokkos::View<scalar_t *, device_t> coordinates;
  Kokkos::View<scalar_t**, device_t> weights;
  Kokkos::View<part_t*, device_t> parts;
  Kokkos::View<scalar_t *, device_t> cut_coordinates;
  Kokkos::View<bool*, device_t> uniform_weights;
  scalar_t sEpsilon;

  ReduceWeightsFunctor(
    index_t mj_n_points,
    const int & mj_weight_array_size,
    Kokkos::View<index_t*, device_t> mj_permutations,
    Kokkos::View<scalar_t *, device_t> mj_coordinates,
    Kokkos::View<scalar_t**, device_t> mj_weights,
    Kokkos::View<part_t*, device_t> mj_parts,
    Kokkos::View<scalar_t *, device_t> mj_cut_coordinates,
    Kokkos::View<bool*, device_t> mj_uniform_weights,
    scalar_t mj_sEpsilon) :
      n_points(mj_n_points),
      value_count(mj_weight_array_size),
      permutations(mj_permutations),
      coordinates(mj_coordinates),
      weights(mj_weights),
      parts(mj_parts),
      cut_coordinates(mj_cut_coordinates),
      uniform_weights(mj_uniform_weights),
      sEpsilon(mj_sEpsilon) {
  }

  size_t team_shmem_size (int team_size) const {
    return sizeof(scalar_t) * value_count * team_size;
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const member_type & teamMember, value_type teamSum) const {
    bool bUniformWeights = uniform_weights(0);

    index_t all_begin = 0;
    index_t all_end = n_points;

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

    // create the team shared data - each thread gets one of the arrays
    scalar_t * shared_ptr = (scalar_t *) teamMember.team_shmem().get_shmem(
      sizeof(scalar_t) * value_count * teamMember.team_size());

    // select the array for this thread
    ArrayType<scalar_t>
      array(&shared_ptr[teamMember.team_rank() * value_count]);

    // create reducer which handles the ArrayType class
    ArraySumReducer<policy_t, scalar_t, part_t> arraySumReducer(
      array, value_count);

    int num_cuts = value_count / 2;

    // call the reduce
    ReduceWeightsFunctorInnerLoop<scalar_t, part_t,
      index_t, device_t> inner_functor(
      permutations,
      coordinates,
      weights,
      parts,
      cut_coordinates,
      bUniformWeights,
      sEpsilon,
      num_cuts);

    Kokkos::parallel_reduce(
      Kokkos::TeamThreadRange(teamMember, begin, end),
      inner_functor, arraySumReducer);

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


// main will call this simple_model test for varies n_teams
double simple_model(int n_teams) {

  typedef Kokkos::Cuda device_t;

  typedef double scalar_t;
  typedef int part_t;
  typedef int lno_t;
 
  typedef Kokkos::View<scalar_t*, device_t> scalar_view_t;
  typedef Kokkos::View<scalar_t**, device_t> weights_view_t;
  typedef Kokkos::View<part_t*, device_t> part_view_t;
  typedef Kokkos::View<bool*, device_t> bool_view_t;

  int n_parts = NUM_PARTS;
  int n_points = 200000;

  if(n_points < n_teams) {
    throw std::logic_error("Simple test does not support n_points < n_teams.");
  }
  if(n_points < n_parts) {
    throw std::logic_error("Simple test does not support n_points < n_parts.");
  }
 
  // create parts - this will be filled by solution
  part_view_t parts(Kokkos::ViewAllocateWithoutInitializing("parts"), n_points);
  Kokkos::parallel_for("initialize parts", n_points, KOKKOS_LAMBDA (int i) {
    parts(i) = 0;
  });

  // create dummy permutations
  part_view_t permutations(Kokkos::ViewAllocateWithoutInitializing("permutations"), n_points);
  Kokkos::parallel_for("initialize permutations", n_points, KOKKOS_LAMBDA (int i) {
    permutations(i) = i; // just one to one for now
  });

  // create coordinates
  scalar_view_t coordinates(Kokkos::ViewAllocateWithoutInitializing("coordinates"), n_points);
  Kokkos::parallel_for("initialize coordinates", n_points, KOKKOS_LAMBDA (int i) {
    coordinates(i) = (double) i / (double) (n_points-1); // just 0 to 1 for now
  });

  // create cut_coordinates
  int n_cuts = n_parts - 1;
  scalar_view_t cut_coordinates(Kokkos::ViewAllocateWithoutInitializing("coordinates"), n_cuts);
  Kokkos::parallel_for("initialize cut_coordinates", n_cuts, KOKKOS_LAMBDA (int i) {
    cut_coordinates(i) = (double) (i+1) / (double) (n_cuts+1); // just 0 to 1 for now but first and last cut are inset
//    printf("init cut %d: %.2f\n", i, (float) cut_coordinates(i));
  });

  // create uniform_weights
  bool_view_t uniform_weights(Kokkos::ViewAllocateWithoutInitializing("uniform_weights"), 1);
  Kokkos::parallel_for("initialize uniform_weights", 1, KOKKOS_LAMBDA (int i) {
    uniform_weights(i) = false; // true for npw
  });

  // create weights
  weights_view_t weights(Kokkos::ViewAllocateWithoutInitializing("weights"), n_points,1);
  Kokkos::parallel_for("initialize weights", n_points, KOKKOS_LAMBDA (int i) {
    weights(i,0) = 1.0; // just 1.0 for now - may not be used
  });  

  scalar_t epsilon = 0.1;

  const int array_size = 2 * n_cuts + 1;
  typedef Kokkos::TeamPolicy<> policy_t;
  ReduceWeightsFunctor<policy_t, scalar_t, part_t,
    lno_t, device_t> teamFunctor(
      n_points,
      array_size,
      permutations,
      coordinates,
      weights,
      parts,
      cut_coordinates,
      uniform_weights,
      epsilon);

  scalar_t * pArray = new scalar_t[array_size];

  auto policy = policy_t(n_teams, Kokkos::AUTO);

  typedef std::chrono::high_resolution_clock Clock;
  auto clock_start = Clock::now();

  Kokkos::parallel_reduce(policy, teamFunctor, pArray);

  double time = static_cast<double>(std::chrono::duration<double, std::milli>(Clock::now() - clock_start).count());

  delete [] pArray;

  return time;
}

int main( int argc, char* argv[] )
{
  Kokkos::ScopeGuard kokkosScope(argc, argv);

  typedef std::chrono::high_resolution_clock Clock;
 
  const int n_teams = 60;

  double time = simple_model(n_teams);
  printf("time: %.2f      teams: %d\n", time, n_teams);

  return 0;
}




