#include <Kokkos_Core.hpp>

/*

This test was created to study intermittent failures observed after the following commit:
  6e32e1361c41f6c4ab25760627a18681080749dc  Tue Aug 28  CUDA: fix team.reduce()

The test does a double nested parallel_reduce sum on an array with
length determined at run-time.

Each thread has its array allocated from the shared memory pool for the team.

The inner loop adds 1.0 to one element in the array for each point and the final
result is validated by checking if the total sum of the array is equal to n_points.

Intermittent failures are seen for some combinations of n_teams and n_points.

Changing the array size (currently 22) will result in similar patterns but
larger array size means errors will occur for smaller total n_points.

Rolling back changes from the above commit will cause all checks to pass.

Changing the struct to be a fixed compile-time array (as in scalar_t ptr[22])
will cause all checks to pass. So the issue seem to be specific to the
special setup we had with a dynamic allocated array.

An example of test output is copied at the end of this file.

*/

template<class scalar_t>
struct ArrayType {
  scalar_t * ptr; // Note scalar_t ptr[22] will work (change constructor below to just ignore pSetPtr)
  KOKKOS_INLINE_FUNCTION
  ArrayType(scalar_t * pSetPtr) : ptr(pSetPtr) {};
};

// Reducer handles join across all array indices
template<class policy_t, class scalar_t>
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

template<class policy_t, class scalar_t, class device_t>
struct ReduceWeightsFunctor {
  typedef typename policy_t::member_type member_type;
  typedef scalar_t value_type[];
  int n_points;
  int value_count;

  ReduceWeightsFunctor(int array_size, int set_n_points) :
    value_count(array_size), n_points(set_n_points)  {}

  size_t team_shmem_size (int team_size) const {
    return sizeof(scalar_t) * value_count * team_size;
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const member_type & teamMember, value_type teamSum) const {

    // Note in this simple test assuming n_points a multiple of num teams
    int points_per_team = n_points / teamMember.league_size();
    int begin = teamMember.league_rank() * points_per_team;
    int end = begin + points_per_team;

    scalar_t * shared_ptr = (scalar_t *) teamMember.team_shmem().get_shmem(
      sizeof(scalar_t) * value_count * teamMember.team_size());

    // select the array for this thread
    scalar_t * my_ptr = &shared_ptr[teamMember.team_rank() * value_count];

    // and encapsulate it in a struct so it can work with the inner loop
    ArrayType<scalar_t> array(my_ptr);

    // create reducer which handles the ArrayType class
    ArraySumReducer<policy_t, scalar_t> arraySumReducer(
      array, value_count);

    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, begin, end),
      [=] (int i, ArrayType<scalar_t>& threadSum) {
      // add 1.0 for one point such that final sum should be equal to n_points
      int fill_index = i % value_count;
      threadSum.ptr[fill_index] += 1.0;
    }, arraySumReducer);
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

int simple_model(int n_points, int n_teams) {

  typedef Kokkos::Cuda device_t;
  typedef Kokkos::TeamPolicy<device_t> policy_t;
  typedef double scalar_t;

  int array_size = 22;

  ReduceWeightsFunctor<policy_t, scalar_t, device_t> teamFunctor(array_size, n_points);

  scalar_t * pArray = new scalar_t[array_size]; // global data for final reduce
  auto policy = policy_t(n_teams, Kokkos::AUTO);
  Kokkos::parallel_reduce(policy, teamFunctor, pArray);
  scalar_t sum = 0;
  for(int i = 0; i < array_size; ++i) {
    sum += pArray[i];
  }
  delete [] pArray;

  // confirm if array sums to the proper total n_points
  if((int) sum != n_points) {
    return 1;
  }
  else {
    return 0;
  }
}

int main( int argc, char* argv[] )
{
  Kokkos::ScopeGuard kokkosScope(argc, argv);
  
  for(int n_points = 2; n_points < pow(2,16); n_points *= 2) {
    for(int n_teams = 1; n_teams <= n_points; n_teams *= 2) {
      printf("Checking n_points: %-8d  n_teams: %-8d points/team: %-8d     ", n_points, n_teams, n_points/n_teams);
      int err = 0;
      const int num_loops = 1000;
      for(int loops = 0; loops < num_loops; ++loops) {
        err += simple_model(n_points, n_teams);
      }
      if(err == 0) {
        printf("Passed\n");
      } else {
        printf("FAILED %d out of %d times!\n", err, num_loops);
      }
    }
  }

  return 0;
}



/* Example output for the test

Checking n_points: 2         n_teams: 1        points/team: 2            Passed
Checking n_points: 2         n_teams: 2        points/team: 1            Passed
Checking n_points: 4         n_teams: 1        points/team: 4            Passed
Checking n_points: 4         n_teams: 2        points/team: 2            Passed
Checking n_points: 4         n_teams: 4        points/team: 1            Passed
Checking n_points: 8         n_teams: 1        points/team: 8            Passed
Checking n_points: 8         n_teams: 2        points/team: 4            Passed
Checking n_points: 8         n_teams: 4        points/team: 2            Passed
Checking n_points: 8         n_teams: 8        points/team: 1            Passed
Checking n_points: 16        n_teams: 1        points/team: 16           Passed
Checking n_points: 16        n_teams: 2        points/team: 8            Passed
Checking n_points: 16        n_teams: 4        points/team: 4            Passed
Checking n_points: 16        n_teams: 8        points/team: 2            Passed
Checking n_points: 16        n_teams: 16       points/team: 1            Passed
Checking n_points: 32        n_teams: 1        points/team: 32           Passed
Checking n_points: 32        n_teams: 2        points/team: 16           Passed
Checking n_points: 32        n_teams: 4        points/team: 8            Passed
Checking n_points: 32        n_teams: 8        points/team: 4            Passed
Checking n_points: 32        n_teams: 16       points/team: 2            Passed
Checking n_points: 32        n_teams: 32       points/team: 1            Passed
Checking n_points: 64        n_teams: 1        points/team: 64           Passed
Checking n_points: 64        n_teams: 2        points/team: 32           Passed
Checking n_points: 64        n_teams: 4        points/team: 16           Passed
Checking n_points: 64        n_teams: 8        points/team: 8            Passed
Checking n_points: 64        n_teams: 16       points/team: 4            Passed
Checking n_points: 64        n_teams: 32       points/team: 2            Passed
Checking n_points: 64        n_teams: 64       points/team: 1            Passed
Checking n_points: 128       n_teams: 1        points/team: 128          Passed
Checking n_points: 128       n_teams: 2        points/team: 64           Passed
Checking n_points: 128       n_teams: 4        points/team: 32           Passed
Checking n_points: 128       n_teams: 8        points/team: 16           Passed
Checking n_points: 128       n_teams: 16       points/team: 8            Passed
Checking n_points: 128       n_teams: 32       points/team: 4            Passed
Checking n_points: 128       n_teams: 64       points/team: 2            Passed
Checking n_points: 128       n_teams: 128      points/team: 1            Passed
Checking n_points: 256       n_teams: 1        points/team: 256          Passed
Checking n_points: 256       n_teams: 2        points/team: 128          Passed
Checking n_points: 256       n_teams: 4        points/team: 64           Passed
Checking n_points: 256       n_teams: 8        points/team: 32           Passed
Checking n_points: 256       n_teams: 16       points/team: 16           Passed
Checking n_points: 256       n_teams: 32       points/team: 8            Passed
Checking n_points: 256       n_teams: 64       points/team: 4            Passed
Checking n_points: 256       n_teams: 128      points/team: 2            Passed
Checking n_points: 256       n_teams: 256      points/team: 1            Passed
Checking n_points: 512       n_teams: 1        points/team: 512          Passed
Checking n_points: 512       n_teams: 2        points/team: 256          Passed
Checking n_points: 512       n_teams: 4        points/team: 128          Passed
Checking n_points: 512       n_teams: 8        points/team: 64           Passed
Checking n_points: 512       n_teams: 16       points/team: 32           Passed
Checking n_points: 512       n_teams: 32       points/team: 16           Passed
Checking n_points: 512       n_teams: 64       points/team: 8            Passed
Checking n_points: 512       n_teams: 128      points/team: 4            Passed
Checking n_points: 512       n_teams: 256      points/team: 2            Passed
Checking n_points: 512       n_teams: 512      points/team: 1            Passed
Checking n_points: 1024      n_teams: 1        points/team: 1024         Passed
Checking n_points: 1024      n_teams: 2        points/team: 512          Passed
Checking n_points: 1024      n_teams: 4        points/team: 256          Passed
Checking n_points: 1024      n_teams: 8        points/team: 128          Passed
Checking n_points: 1024      n_teams: 16       points/team: 64           FAILED 8 out of 1000 times!
Checking n_points: 1024      n_teams: 32       points/team: 32           Passed
Checking n_points: 1024      n_teams: 64       points/team: 16           Passed
Checking n_points: 1024      n_teams: 128      points/team: 8            Passed
Checking n_points: 1024      n_teams: 256      points/team: 4            Passed
Checking n_points: 1024      n_teams: 512      points/team: 2            Passed
Checking n_points: 1024      n_teams: 1024     points/team: 1            Passed
Checking n_points: 2048      n_teams: 1        points/team: 2048         Passed
Checking n_points: 2048      n_teams: 2        points/team: 1024         Passed
Checking n_points: 2048      n_teams: 4        points/team: 512          Passed
Checking n_points: 2048      n_teams: 8        points/team: 256          Passed
Checking n_points: 2048      n_teams: 16       points/team: 128          FAILED 361 out of 1000 times!
Checking n_points: 2048      n_teams: 32       points/team: 64           FAILED 126 out of 1000 times!
Checking n_points: 2048      n_teams: 64       points/team: 32           Passed
Checking n_points: 2048      n_teams: 128      points/team: 16           Passed
Checking n_points: 2048      n_teams: 256      points/team: 8            Passed
Checking n_points: 2048      n_teams: 512      points/team: 4            Passed
Checking n_points: 2048      n_teams: 1024     points/team: 2            Passed
Checking n_points: 2048      n_teams: 2048     points/team: 1            Passed
Checking n_points: 4096      n_teams: 1        points/team: 4096         Passed
Checking n_points: 4096      n_teams: 2        points/team: 2048         Passed
Checking n_points: 4096      n_teams: 4        points/team: 1024         Passed
Checking n_points: 4096      n_teams: 8        points/team: 512          Passed
Checking n_points: 4096      n_teams: 16       points/team: 256          FAILED 372 out of 1000 times!
Checking n_points: 4096      n_teams: 32       points/team: 128          FAILED 921 out of 1000 times!
Checking n_points: 4096      n_teams: 64       points/team: 64           FAILED 324 out of 1000 times!
Checking n_points: 4096      n_teams: 128      points/team: 32           Passed
Checking n_points: 4096      n_teams: 256      points/team: 16           Passed
Checking n_points: 4096      n_teams: 512      points/team: 8            Passed
Checking n_points: 4096      n_teams: 1024     points/team: 4            Passed
Checking n_points: 4096      n_teams: 2048     points/team: 2            Passed
Checking n_points: 4096      n_teams: 4096     points/team: 1            Passed
Checking n_points: 8192      n_teams: 1        points/team: 8192         Passed
Checking n_points: 8192      n_teams: 2        points/team: 4096         Passed
Checking n_points: 8192      n_teams: 4        points/team: 2048         Passed
Checking n_points: 8192      n_teams: 8        points/team: 1024         Passed
Checking n_points: 8192      n_teams: 16       points/team: 512          FAILED 420 out of 1000 times!
Checking n_points: 8192      n_teams: 32       points/team: 256          FAILED 950 out of 1000 times!
Checking n_points: 8192      n_teams: 64       points/team: 128          FAILED 996 out of 1000 times!
Checking n_points: 8192      n_teams: 128      points/team: 64           FAILED 626 out of 1000 times!
Checking n_points: 8192      n_teams: 256      points/team: 32           Passed
Checking n_points: 8192      n_teams: 512      points/team: 16           Passed
Checking n_points: 8192      n_teams: 1024     points/team: 8            Passed
Checking n_points: 8192      n_teams: 2048     points/team: 4            Passed
Checking n_points: 8192      n_teams: 4096     points/team: 2            Passed
Checking n_points: 8192      n_teams: 8192     points/team: 1            Passed
Checking n_points: 16384     n_teams: 1        points/team: 16384        Passed
Checking n_points: 16384     n_teams: 2        points/team: 8192         Passed
Checking n_points: 16384     n_teams: 4        points/team: 4096         Passed
Checking n_points: 16384     n_teams: 8        points/team: 2048         Passed
Checking n_points: 16384     n_teams: 16       points/team: 1024         FAILED 544 out of 1000 times!
Checking n_points: 16384     n_teams: 32       points/team: 512          FAILED 943 out of 1000 times!
Checking n_points: 16384     n_teams: 64       points/team: 256          FAILED 999 out of 1000 times!
Checking n_points: 16384     n_teams: 128      points/team: 128          FAILED 1000 out of 1000 times!
Checking n_points: 16384     n_teams: 256      points/team: 64           FAILED 970 out of 1000 times!
Checking n_points: 16384     n_teams: 512      points/team: 32           Passed
Checking n_points: 16384     n_teams: 1024     points/team: 16           Passed
Checking n_points: 16384     n_teams: 2048     points/team: 8            Passed
Checking n_points: 16384     n_teams: 4096     points/team: 4            Passed
Checking n_points: 16384     n_teams: 8192     points/team: 2            Passed
Checking n_points: 16384     n_teams: 16384    points/team: 1            Passed
Checking n_points: 32768     n_teams: 1        points/team: 32768        Passed
Checking n_points: 32768     n_teams: 2        points/team: 16384        Passed
Checking n_points: 32768     n_teams: 4        points/team: 8192         Passed
Checking n_points: 32768     n_teams: 8        points/team: 4096         Passed
Checking n_points: 32768     n_teams: 16       points/team: 2048         FAILED 557 out of 1000 times!
Checking n_points: 32768     n_teams: 32       points/team: 1024         FAILED 956 out of 1000 times!
Checking n_points: 32768     n_teams: 64       points/team: 512          FAILED 997 out of 1000 times!
Checking n_points: 32768     n_teams: 128      points/team: 256          FAILED 1000 out of 1000 times!
Checking n_points: 32768     n_teams: 256      points/team: 128          FAILED 1000 out of 1000 times!
Checking n_points: 32768     n_teams: 512      points/team: 64           FAILED 1000 out of 1000 times!
Checking n_points: 32768     n_teams: 1024     points/team: 32           Passed
Checking n_points: 32768     n_teams: 2048     points/team: 16           Passed
Checking n_points: 32768     n_teams: 4096     points/team: 8            Passed
Checking n_points: 32768     n_teams: 8192     points/team: 4            Passed
Checking n_points: 32768     n_teams: 16384    points/team: 2            Passed
Checking n_points: 32768     n_teams: 32768    points/team: 1            Passed

*/
