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

struct Result {
  int time_us;        // time in ms for actual partition
  int n_teams;        // how many teams we ran with
  int n_points;       // how many points we ran with
};

Result run_test(int n_teams, int n_points) {

  typedef Kokkos::Cuda device_t;
  typedef Kokkos::Cuda execution_t;

  // start clock
  typedef std::chrono::high_resolution_clock Clock;
  auto clock_start = Clock::now();
  double sum = 0;
  typedef Kokkos::TeamPolicy<execution_t>::member_type member_type;
  Kokkos::parallel_reduce(Kokkos::TeamPolicy<execution_t>(n_teams, Kokkos::AUTO()),
    KOKKOS_LAMBDA (const member_type& teamMember, int lsum) {
    int s = 0;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, n_points/n_teams),
      [=] (const int k, int & inner_lsum) {
      int inner_s = 0;
      for(int i = 0; i<10; i++) inner_s++;
      inner_lsum += inner_s;
    },s);
    lsum += s;
  },sum);


  Result result;
  result.time_us = static_cast<int>(std::chrono::duration_cast<
    std::chrono::microseconds>(Clock::now() - clock_start).count());
  result.n_teams = n_teams;
  result.n_points = n_points;
  return result;
}

int main( int argc, char* argv[] )
{
  Kokkos::ScopeGuard kokkosScope(argc, argv); 

  int n_points = 2048 * 4096;
  
  std::vector<Result> results; // store the results for each run
  for(int n_teams = 1; n_teams <= n_points; n_teams *=2) {
    Result result = run_test(n_teams, n_points);
    results.push_back(result); // add to vector for logging at end
  }

  // now loop and log each result - shows how n_teams impacts total time
  for(auto&& result : results) {
    printf("teams: %8d   time: %d us\n", result.n_teams, result.time_us);
  }
}
