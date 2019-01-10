
// The simple model creates n points in 1d evenly distributed 0..1,
// then coordinates are squared so half the points lie below 0.25.
// A thread reduce loop inside a team reduce loop checks each coordinate
// and determines if they are left or right of the current cut.
// The total weights to the left of the cut are reduced (summed).
// Then the cut is shifted to estimate a new cut position,
// where the goal is to have the left and right have the same weight.
// The main loop runs the full test for a range of team counts
// and logs the time cost for each.

#include <Kokkos_Core.hpp>
#include <vector>

#define N_POINTS_POW_OF_2 18
#define N_PARTS 20

typedef Kokkos::View<double*> view_t;

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

/////////////////////////////////////////////////////////////
// Struct to initialize the points

struct InitializePointsFunctor {
  typedef double value_type[];
  typedef typename view_t::size_type size_type;

  int n_points;
  view_t coords;

  InitializePointsFunctor(int n_points_, view_t &coords_) :
    n_points(n_points_), coords(coords_)
  {}
  
  KOKKOS_INLINE_FUNCTION
  void operator() (const size_type ii) const {
    // make a simple shift so coords are not ordered
    // after scaling 0-1, do x^2 to make an uneven distribution
    int shift_index = ii + n_points / 2;
    if (shift_index >= n_points) shift_index -= n_points;
    double val = (double) (shift_index) / (double) (n_points - 1);
    coords(ii) = val * val; 
  }
};

  
/////////////////////////////////////////////////////////////
// Struct to surf over points and accrue weights of each part
// Bottom-level:  thread parallelism

struct TeamFunctor {
  typedef TeamFunctor reducer;
  typedef double value_type[];
  typedef typename view_t::size_type size_type;

  const size_type value_count;  // should equal n_parts
  const int n_points_per_team;
  const Kokkos::View<double*> coords;
  const Kokkos::View<double*> cut_line;
    
  KOKKOS_INLINE_FUNCTION
  TeamFunctor(
    const size_type n_parts_, 
    const int n_points_per_team_,
    const Kokkos::View<double*>& coords_,
    const Kokkos::View<double*>& cut_line_) :
    value_count(n_parts_),
    n_points_per_team(n_points_per_team_),
    coords(coords_),
    cut_line(cut_line_)
  {}
   
  KOKKOS_INLINE_FUNCTION
  void operator() (const Kokkos::TeamPolicy<>::member_type teamMember,
                   value_type sum) const 
  {

    int coord_idx_offset = teamMember.league_rank() * n_points_per_team;
    int begin = coord_idx_offset * value_count;
    int end = begin + n_points_per_team * value_count;
    auto range = Kokkos::TeamThreadRange<>(teamMember, begin, end);

    // create the team shared data - each thread gets one of the arrays
    double * shared_ptr = (double *) teamMember.team_shmem().get_shmem(
      sizeof(double) * value_count * teamMember.team_size());

    // select the array for this thread
    ArrayType<double>
      array(&shared_ptr[teamMember.team_rank() * value_count]);

    typedef Kokkos::TeamPolicy<Kokkos::Cuda> policy_t;

    // create reducer which handles the ArrayType class
    ArraySumReducer<policy_t, double, int> arraySumReducer(
      array, value_count);

    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, begin, end),
      [=] (const size_t ii, ArrayType<double>& threadSum) {
        int coord_idx = ii / value_count;
        double coord_tmp = coords(coord_idx);
        int part_idx = ii % value_count;      

        if(coord_tmp >= cut_line(part_idx) &&
          coord_tmp <= cut_line(part_idx+1))
        {
          threadSum.ptr[part_idx] += 1.0;
        }
    }, arraySumReducer);
    teamMember.team_barrier();

    Kokkos::single(Kokkos::PerTeam(teamMember), [=] () {
      for(int n = 0; n < value_count; ++n) {
        sum[n] += array.ptr[n];
      }
    });
  }
    
  KOKKOS_INLINE_FUNCTION 
  void join (volatile value_type dst, const volatile value_type src) const 
  {
    for (size_type j = 0; j < value_count; j++) dst[j] += src[j];
  }
    
  KOKKOS_INLINE_FUNCTION 
  void init(value_type sum) const 
  {
    for (size_type j = 0; j < value_count; j++) sum[j] = 0.0;
  }

  size_t team_shmem_size(int team_size) const
  { return sizeof(double) * value_count * team_size; }
};

//////////////////////////////////////////////////////////////////////////
// store Stats from a run
struct Stats {
  double time_ms;        // time in ms for actual partition
  view_t::HostMirror cut_line;    // the final cut
  int n_teams;        // how many teams we ran with
};

/////////////////////////////////////////////////////////////////////////
// main will call this simple_model test for varies n_teams

Stats simple_model(int n_teams, int n_parts) 
{
  const int n_points = pow(2, N_POINTS_POW_OF_2); 
  const int n_points_per_team = n_points / n_teams;

  view_t coords("coords", n_points);

  InitializePointsFunctor initializePointsFunctor(n_points, coords);

  Kokkos::parallel_for("initialize points", n_points, initializePointsFunctor);

  auto target = (double) coords.size() / n_parts;

  // initial cut lines, evenly spaced
  Kokkos::View<double *> cut_line("cut_line", n_parts+1);
  Kokkos::parallel_for("initialize cut lines", n_parts+1, KOKKOS_LAMBDA (int k) {
    cut_line(k) = k * (1./n_parts);
  });

  // get the time on the main loop
  typedef std::chrono::high_resolution_clock Clock;

  double *overallSum = new double[n_parts];

  TeamFunctor teamFunctor(n_parts, n_points_per_team, coords, cut_line);

  auto policy = Kokkos::TeamPolicy<>(n_teams, Kokkos::AUTO);

  // get the time on the main loop
  typedef std::chrono::high_resolution_clock Clock;
  auto clock_start = Clock::now();

  Kokkos::parallel_reduce(policy, teamFunctor, overallSum);

  double time = static_cast<double>(std::chrono::duration<double, std::milli>(Clock::now() - clock_start).count());

  std::cout << "WEIGHTS  " ;
  double wgtsum = 0.;
  for (int k = 0; k < n_parts; k++) {
    wgtsum += overallSum[k];
    std::cout << overallSum[k] << " ";
  }
  std::cout << "; TARGET = " << target << "; SUM = " << wgtsum << " " 
            << (wgtsum == n_points ? " " : " ERROR: BAD SUM") << std::endl;

  std::cout << "OLD CUTS ";

  Kokkos::View<double *>::HostMirror host_cut_line = Kokkos::create_mirror_view(cut_line);

  Kokkos::deep_copy (host_cut_line, cut_line); // Copy from device to host

  for (int k = 0; k < n_parts+1; k++)
    std::cout << host_cut_line(k) << " ";
  std::cout << std::endl;

  // update cut_lines to better balance
  double sumLeft = 0.;
  double sumTarget = 0.;
  double epsilon = 0.5;

  for (int k = 1; k < n_parts; k++) {
    sumLeft += overallSum[k-1];
    sumTarget = k * target;
    if (sumLeft < sumTarget - epsilon) {
      // cut moves right
      host_cut_line(k) += ((host_cut_line(k+1) - host_cut_line(k)) * 0.5);
    }
    else if (sumLeft > sumTarget + epsilon) {
     // cut moves left
      host_cut_line(k) -= ((host_cut_line(k) - host_cut_line(k-1)) * 0.5);
    }
  }
  std::cout << "NEW CUTS ";
  for (int k = 0; k < n_parts+1; k++)
    std::cout << host_cut_line(k) << " ";
  std::cout << std::endl;

  delete [] overallSum;

  // set the output values
  Stats stat;
  stat.time_ms = time;
  stat.cut_line = host_cut_line;
  stat.n_teams = n_teams;
  return stat;
}

//////////////////////////////////////////////////////////////////////////
int main(int narg, char* arg[] )
{
  Kokkos::ScopeGuard kokkosScope(narg, arg); 

  std::vector<Stats> stats; 
  int n_parts = N_PARTS;
  if (narg > 1) n_parts = std::atoi(arg[1]);

  for(int n_teams = 2; n_teams <= pow(2, N_POINTS_POW_OF_2); n_teams *=2){
    printf("N_TEAMS = %d\n", n_teams);
    stats.push_back(simple_model(n_teams, n_parts));
  }

  // now loop and log each stats - shows how n_teams impacts total time
  for(auto&& stat : stats) {
    printf("teams: %8d   cuts: ", stat.n_teams);
    for (int k = 0; k < int(stat.cut_line.dimension(0)); k++)
      printf("%.2lf ", stat.cut_line(k));
    printf("   time: %.2f ms\n", stat.time_ms);
  }
}


