make clean
make -j8 KOKKOS_ARCH="Power8,Kepler37"
bsub -x -Is -q rhel7F -n 1 ./test.cuda
