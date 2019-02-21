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

typedef Kokkos::Cuda device_t;
typedef Kokkos::Cuda execution_t;
typedef double mj_scalar_t;
typedef int mj_gno_t;
typedef int mj_lno_t;
typedef int mj_part_t;

class TestClass {

public:
  TestClass(int n_points) :
    num_local_coords(n_points),
    max_num_part_along_dim(2),
    max_num_total_part_along_dim(3),
    max_num_cut_along_dim(1),
    max_concurrent_part_calculation(2),
    distribute_points_on_cut_lines(true),
    coord_dim(3),
    num_weights_per_coord(1)
  {
  }

  void prebuild() {

    kokkos_mj_coordinates = Kokkos::View<mj_scalar_t**, device_t>(
      "kokkos_mj_coordinates", num_local_coords, coord_dim);
/*
    auto local_kokkos_mj_coordinates = kokkos_mj_coordinates;
    int local_coord_dim = coord_dim;
    Kokkos::parallel_for(Kokkos::RangePolicy<execution_t, int> (0, num_local_coords), KOKKOS_LAMBDA (const int i) {
      for(int d = 0; d < local_coord_dim; ++d) {
        local_kokkos_mj_coordinates(i,d) = 0;
      } 
    });
 */
    kokkos_mj_weights = Kokkos::View<mj_scalar_t**, device_t>(
      "kokkos_mj_weights", num_local_coords, num_weights_per_coord);
/*
    auto local_kokkos_mj_weights = kokkos_mj_weights;;
    int local_num_weights_per_coord = num_weights_per_coord;
    Kokkos::parallel_for(Kokkos::RangePolicy<execution_t, int> (0, num_local_coords), KOKKOS_LAMBDA (const int i) {
      for(int w = 0; w < local_num_weights_per_coord; ++w) {
        local_kokkos_mj_weights(i, w) = 0;
      }
    });
*/
    kokkos_initial_mj_gnos = Kokkos::View<const mj_gno_t*, device_t>(
      "kokkos_initial_mj_gnos", num_local_coords);
  /* 
    auto local_kokkos_initial_mj_gnos = kokkos_initial_mj_gnos;
    Kokkos::parallel_for(Kokkos::RangePolicy<execution_t, int> (0, num_local_coords), KOKKOS_LAMBDA (const int i) {
      local_kokkos_initial_mj_gnos(i) = 0;
    });
*/
  }
  
  int num_local_coords;
  int max_num_part_along_dim;
  int max_num_total_part_along_dim;
  int max_num_cut_along_dim;
  int max_concurrent_part_calculation;
  bool distribute_points_on_cut_lines;
  int coord_dim;
  int num_weights_per_coord;
  
  int allocate(int n_points);

  int myActualRank = 0;     // initial rank
  
  Kokkos::View<mj_scalar_t **, Kokkos::LayoutLeft, device_t>
    kokkos_mj_coordinates; // two dimension coordinate array
  Kokkos::View<mj_scalar_t **, device_t> kokkos_mj_weights;
  Kokkos::View<const mj_gno_t*, device_t> kokkos_initial_mj_gnos;
  Kokkos::View<mj_gno_t*, device_t> kokkos_current_mj_gnos;
  Kokkos::View<int*, device_t> kokkos_owner_of_coordinate;
  Kokkos::View<mj_lno_t*, device_t> kokkos_coordinate_permutations;
  Kokkos::View<mj_lno_t*, device_t> kokkos_new_coordinate_permutations;
  Kokkos::View<mj_part_t*, device_t> kokkos_assigned_part_ids;
  Kokkos::View<mj_part_t*, device_t> kokkos_info;
  Kokkos::View<mj_lno_t *, device_t> kokkos_part_xadj;
  Kokkos::View<mj_lno_t *, device_t> kokkos_new_part_xadj;
  Kokkos::View<mj_scalar_t *, device_t> kokkos_all_cut_coordinates;
  Kokkos::View<mj_scalar_t *, device_t>
    kokkos_process_cut_line_weight_to_put_left;
  Kokkos::View<mj_scalar_t *, Kokkos::LayoutLeft, device_t>
    kokkos_thread_cut_line_weight_to_put_left;
  Kokkos::View<mj_scalar_t *, device_t> kokkos_cut_coordinates_work_array;
  Kokkos::View<mj_scalar_t *, device_t> kokkos_target_part_weights;
  Kokkos::View<mj_scalar_t *, device_t> kokkos_cut_upper_bound_coordinates;
  Kokkos::View<mj_scalar_t *, device_t> kokkos_cut_lower_bound_coordinates;
  Kokkos::View<mj_scalar_t *, device_t> kokkos_cut_lower_bound_weights;
  Kokkos::View<mj_scalar_t *, device_t> kokkos_cut_upper_bound_weights;
  Kokkos::View<mj_scalar_t *, device_t>
    kokkos_process_local_min_max_coord_total_weight;
  Kokkos::View<mj_scalar_t *, device_t>
    kokkos_global_min_max_coord_total_weight;
  Kokkos::View<bool *, device_t> kokkos_is_cut_line_determined;
  Kokkos::View<mj_part_t *, device_t> kokkos_my_incomplete_cut_count;
  typename decltype (kokkos_my_incomplete_cut_count)::HostMirror
      host_kokkos_my_incomplete_cut_count; 
#ifndef TURN_OFF_MERGE_CHUNKS
  Kokkos::View<mj_part_t *, device_t> kokkos_prefix_sum_num_cuts;
#endif
  Kokkos::View<double *, Kokkos::LayoutLeft, device_t>
    kokkos_thread_part_weights;
  Kokkos::View<mj_scalar_t *, Kokkos::LayoutLeft, device_t>
    kokkos_thread_cut_left_closest_point;
  Kokkos::View<mj_scalar_t *, Kokkos::LayoutLeft, device_t>
    kokkos_thread_cut_right_closest_point;
  Kokkos::View<mj_lno_t *, Kokkos::LayoutLeft, device_t>
    kokkos_thread_point_counts;
  Kokkos::View<mj_scalar_t *, device_t> kokkos_process_rectilinear_cut_weight;
  Kokkos::View<mj_scalar_t *, device_t> kokkos_global_rectilinear_cut_weight;
  Kokkos::View<mj_scalar_t *, device_t>
    kokkos_total_part_weight_left_right_closests;
  Kokkos::View<mj_scalar_t *, device_t>
    kokkos_global_total_part_weight_left_right_closests;
};

int TestClass::allocate(int n_points) {
  auto clock_start = std::chrono::high_resolution_clock::now();

  // points to process that initially owns the coordinate.
  Kokkos::resize(this->kokkos_owner_of_coordinate, 0);

  // Throughout the partitioning execution,
  // instead of the moving the coordinates, hold a permutation array for parts.
  // coordinate_permutations holds the current permutation.
  Kokkos::resize(this->kokkos_coordinate_permutations, this->num_local_coords);

  Kokkos::View<mj_lno_t*, device_t> temp = Kokkos::View<mj_lno_t*, device_t>(
    "kokkos_coordinate_permutations", num_local_coords);
  Kokkos::parallel_for(
    Kokkos::RangePolicy<execution_t, int> (
    0, this->num_local_coords), KOKKOS_LAMBDA (const int i) {
      temp(i) = i;
  });

  // bring the local data back to the class
  this->kokkos_coordinate_permutations = temp;

  // new_coordinate_permutations holds the current permutation.
  this->kokkos_new_coordinate_permutations = Kokkos::View<mj_lno_t*, device_t>(
    "num_local_coords", this->num_local_coords);
  this->kokkos_assigned_part_ids = Kokkos::View<mj_part_t*, device_t>(
    "assigned parts"); // TODO empty is ok for NULL replacement?
  if(this->num_local_coords > 0){
    this->kokkos_assigned_part_ids = Kokkos::View<mj_part_t*, device_t>(
      "assigned part ids", this->num_local_coords);
    this->kokkos_info = Kokkos::View<mj_part_t*, device_t>(
      "info", this->num_local_coords);
  }
  // single partition starts at index-0, and ends at numLocalCoords
  // inTotalCounts array holds the end points in coordinate_permutations array
  // for each partition. Initially sized 1, and single element is set to
  // numLocalCoords.
  this->kokkos_part_xadj = Kokkos::View<mj_lno_t*, device_t>("part xadj", 1);

  // TODO: How do do the above operation on device
  auto local_num_local_coords = this->num_local_coords;
  auto local_kokkos_part_xadj = this->kokkos_part_xadj;
  Kokkos::parallel_for(
    Kokkos::RangePolicy<execution_t, int> (0, 1),
    KOKKOS_LAMBDA (const int i) {
      // the end of the initial partition is the end of coordinates.
      local_kokkos_part_xadj(0) = static_cast<mj_lno_t>(local_num_local_coords);
  });

  // the ends points of the output, this is allocated later.
  this->kokkos_new_part_xadj = Kokkos::View<mj_lno_t*, device_t>("empty");

  // only store this much if cuts are needed to be stored.
  // this->all_cut_coordinates = allocMemory< mj_scalar_t>(this->total_num_cut);
  this->kokkos_all_cut_coordinates = Kokkos::View<mj_scalar_t*, device_t>(
    "all cut coordinates",
    this->max_num_cut_along_dim * this->max_concurrent_part_calculation);
    
  // how much weight percentage should a MPI put left side of the each cutline
  this->kokkos_process_cut_line_weight_to_put_left = Kokkos::View<mj_scalar_t*,
    device_t>("empty");
    
  // how much weight percentage should each thread in MPI put left side of
  // each outline
  this->kokkos_thread_cut_line_weight_to_put_left =
    Kokkos::View<mj_scalar_t*, Kokkos::LayoutLeft, device_t>("empty");
   
  // distribute_points_on_cut_lines = false;
  if(this->distribute_points_on_cut_lines){
    this->kokkos_process_cut_line_weight_to_put_left =
      Kokkos::View<mj_scalar_t *, device_t>(
      "kokkos_process_cut_line_weight_to_put_left",
        this->max_num_cut_along_dim * this->max_concurrent_part_calculation);
    this->kokkos_thread_cut_line_weight_to_put_left =
      Kokkos::View<mj_scalar_t *, Kokkos::LayoutLeft, device_t>(
      "kokkos_thread_cut_line_weight_to_put_left", this->max_num_cut_along_dim);
    this->kokkos_process_rectilinear_cut_weight =
      Kokkos::View<mj_scalar_t *, device_t>(
      "kokkos_process_rectilinear_cut_weight", this->max_num_cut_along_dim);
    this->kokkos_global_rectilinear_cut_weight =
      Kokkos::View<mj_scalar_t *, device_t>(
      "kokkos_global_rectilinear_cut_weight", this->max_num_cut_along_dim);
  }

  // work array to manipulate coordinate of cutlines in different iterations.
  // necessary because previous cut line information is used for determining
  // the next cutline information. therefore, cannot update the cut work array
  // until all cutlines are determined.
  this->kokkos_cut_coordinates_work_array =
    Kokkos::View<mj_scalar_t *, device_t>("kokkos_cut_coordinates_work_array",
    this->max_num_cut_along_dim * this->max_concurrent_part_calculation);

  // cumulative part weight array.
  this->kokkos_target_part_weights = Kokkos::View<mj_scalar_t*, device_t>(
    "kokkos_target_part_weights",
    this->max_num_part_along_dim * this->max_concurrent_part_calculation);
  
  // upper bound coordinate of a cut line
  this->kokkos_cut_upper_bound_coordinates =
    Kokkos::View<mj_scalar_t*, device_t>("kokkos_cut_upper_bound_coordinates",
    this->max_num_cut_along_dim * this->max_concurrent_part_calculation);
    
  // lower bound coordinate of a cut line  
  this->kokkos_cut_lower_bound_coordinates =
    Kokkos::View<mj_scalar_t*, device_t>("kokkos_cut_lower_bound_coordinates",
    this->max_num_cut_along_dim* this->max_concurrent_part_calculation);

  // lower bound weight of a cut line
  this->kokkos_cut_lower_bound_weights =
    Kokkos::View<mj_scalar_t*, device_t>("kokkos_cut_lower_bound_weights",
    this->max_num_cut_along_dim* this->max_concurrent_part_calculation);

  //upper bound weight of a cut line
  this->kokkos_cut_upper_bound_weights =
    Kokkos::View<mj_scalar_t*, device_t>("kokkos_cut_upper_bound_weights",
    this->max_num_cut_along_dim* this->max_concurrent_part_calculation);

  // combined array to exchange the min and max coordinate,
  // and total weight of part.
  this->kokkos_process_local_min_max_coord_total_weight =
    Kokkos::View<mj_scalar_t*, device_t>(
    "kokkos_process_local_min_max_coord_total_weight",
    3 * this->max_concurrent_part_calculation);

  // global combined array with the results for min, max and total weight.
  this->kokkos_global_min_max_coord_total_weight =
    Kokkos::View<mj_scalar_t*, device_t>(
    "kokkos_global_min_max_coord_total_weight",
    3 * this->max_concurrent_part_calculation);

  // is_cut_line_determined is used to determine if a cutline is
  // determined already. If a cut line is already determined, the next
  // iterations will skip this cut line.
  this->kokkos_is_cut_line_determined = Kokkos::View<bool *, device_t>(
    "kokkos_is_cut_line_determined",
    this->max_num_cut_along_dim * this->max_concurrent_part_calculation);

  // my_incomplete_cut_count count holds the number of cutlines that have not
  // been finalized for each part when concurrentPartCount>1, using this
  // information, if my_incomplete_cut_count[x]==0, then no work is done for
  // this part.
  this->kokkos_my_incomplete_cut_count =  Kokkos::View<mj_part_t *, device_t>(
    "kokkos_my_incomplete_cut_count", this->max_concurrent_part_calculation);
    
  // we'll copy to host sometimes so we can access things quickly
  this->host_kokkos_my_incomplete_cut_count =
    Kokkos::create_mirror_view(kokkos_my_incomplete_cut_count);
      
#ifndef TURN_OFF_MERGE_CHUNKS
  this->kokkos_prefix_sum_num_cuts =  Kokkos::View<mj_part_t *, device_t>(
    "kokkos_prefix_sum_num_cuts", this->max_concurrent_part_calculation);
#endif

  // local part weights of each thread.
  this->kokkos_thread_part_weights = Kokkos::View<double *,
    Kokkos::LayoutLeft, device_t>("thread_part_weights",
    this->max_num_total_part_along_dim * this->max_concurrent_part_calculation);

  this->kokkos_thread_cut_left_closest_point = Kokkos::View<mj_scalar_t *,
    Kokkos::LayoutLeft, device_t>("kokkos_thread_cut_left_closest_point",
    this->max_num_cut_along_dim * this->max_concurrent_part_calculation);

  // thread_cut_right_closest_point to hold the closest coordinate to a
  // cutline from right (for each thread)
  this->kokkos_thread_cut_right_closest_point = Kokkos::View<mj_scalar_t *,
    Kokkos::LayoutLeft, device_t>("kokkos_thread_cut_right_closest_point",
    this->max_num_cut_along_dim * this->max_concurrent_part_calculation);

  // to store how many points in each part a thread has.
  this->kokkos_thread_point_counts = Kokkos::View<mj_lno_t *,
    Kokkos::LayoutLeft, device_t>("kokkos_thread_point_counts",
    this->max_num_part_along_dim);

  // for faster communication, concatanation of
  // totalPartWeights sized 2P-1, since there are P parts and P-1 cut lines
  // leftClosest distances sized P-1, since P-1 cut lines
  // rightClosest distances size P-1, since P-1 cut lines.
  this->kokkos_total_part_weight_left_right_closests =
    Kokkos::View<mj_scalar_t*, device_t>(
      "total_part_weight_left_right_closests",
      (this->max_num_total_part_along_dim + this->max_num_cut_along_dim * 2) *
      this->max_concurrent_part_calculation);

  this->kokkos_global_total_part_weight_left_right_closests =
    Kokkos::View<mj_scalar_t*, device_t>(
      "global_total_part_weight_left_right_closests",
      (this->max_num_total_part_along_dim +
      this->max_num_cut_along_dim * 2) * this->max_concurrent_part_calculation);

  Kokkos::View<mj_scalar_t**, Kokkos::LayoutLeft, device_t> coord(
    "coord", this->num_local_coords, this->coord_dim);
 
  auto local_kokkos_mj_coordinates = kokkos_mj_coordinates; 
  auto local_coord_dim = this->coord_dim;
  
  Kokkos::parallel_for(
    Kokkos::RangePolicy<execution_t, int> (
      0, local_num_local_coords),
    KOKKOS_LAMBDA (const int j) {
    for (int i=0; i < local_coord_dim; i++){
      coord(j,i) = local_kokkos_mj_coordinates(j,i);
  }});

  this->kokkos_mj_coordinates = coord;

  Kokkos::View<mj_scalar_t**, device_t> weights(
  "weights", this->num_local_coords, this->num_weights_per_coord);

  auto local_kokkos_mj_weights = kokkos_mj_weights;
  auto local_num_weights_per_coord = this->num_weights_per_coord;
  Kokkos::parallel_for(
    Kokkos::RangePolicy<execution_t, int> (
      0, local_num_local_coords),
    KOKKOS_LAMBDA (const int j) {
    for (int i=0; i < local_num_weights_per_coord; i++){
      weights(j,i) = local_kokkos_mj_weights(j,i);
  }});

  this->kokkos_mj_weights = weights;

  this->kokkos_current_mj_gnos =
  Kokkos::View<mj_gno_t*, device_t>("gids", local_num_local_coords);
  auto local_kokkos_current_mj_gnos = this->kokkos_current_mj_gnos;
  auto local_kokkos_initial_mj_gnos = this->kokkos_initial_mj_gnos;

  this->kokkos_owner_of_coordinate = Kokkos::View<int*, device_t>(
    "kokkos_owner_of_coordinate", this->num_local_coords);

  auto local_kokkos_owner_of_coordinate = this->kokkos_owner_of_coordinate;
  auto local_myActualRank = this->myActualRank;

  Kokkos::parallel_for(
    Kokkos::RangePolicy<execution_t, int> (
      0, local_num_local_coords),
    KOKKOS_LAMBDA (const int j) {
    local_kokkos_current_mj_gnos(j) = local_kokkos_initial_mj_gnos(j);
    local_kokkos_owner_of_coordinate(j) = local_myActualRank;
  });

  int time_us = static_cast<int>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - clock_start).count());

  return time_us;
}

int main( int argc, char* argv[] )
{
  Kokkos::ScopeGuard kokkosScope(argc, argv);
  
  int n_points = 2; // pow(2,25);

  TestClass test(n_points);
  test.prebuild();

  int time = test.allocate(n_points);
  
  printf("time: %d\n", time);
}
