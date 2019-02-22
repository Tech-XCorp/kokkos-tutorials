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

template<class kernel_t>
int run_test_1(int n_teams, int n_points) {
  auto clock_start = std::chrono::high_resolution_clock::now();
  kernel_t sum = 0;
  typedef Kokkos::TeamPolicy<Kokkos::Cuda>::member_type member_type;
  Kokkos::parallel_reduce(Kokkos::TeamPolicy<Kokkos::Cuda>(n_teams, Kokkos::AUTO()),
    KOKKOS_LAMBDA (const member_type& teamMember, kernel_t & lsum) {
    kernel_t s = 0;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, n_points/n_teams),
      [=] (const int k, kernel_t & inner_lsum) {
      inner_lsum += 1;
    }, s);
    teamMember.team_barrier();
    if(teamMember.team_rank() == 0) {
      lsum += s;
    }
  }, sum);

  int time_us = static_cast<int>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - clock_start).count());

  if((int) sum != n_points) {
    printf("Test Failed!\n");
    std::abort();
  }

  return time_us;
}

int main( int argc, char* argv[] )
{
  Kokkos::ScopeGuard kokkosScope(argc, argv);

  std::vector<int> points_to_test = { (int) pow(2,8), (int) pow(2,18), (int) pow(2,25)};
  for(auto itr = points_to_test.begin(); itr != points_to_test.end(); ++itr) {
    int n_points = *itr;
    printf("\nRun test with n_points: %d\n", n_points);

    printf("   Teams    int  float  double\n"); // header
    for(int n_teams = 1; n_teams <= pow(2,14); n_teams *=2) {
      if(n_teams > n_points) continue;
      int time_us_1 = run_test_1<int>(n_teams, n_points);
      int time_us_2 = run_test_1<float>(n_teams, n_points);
      int time_us_3 = run_test_1<double>(n_teams, n_points);
      printf("%8d %6d %6d %6d\n", n_teams, time_us_1, time_us_2, time_us_3);
    }
  }
}
