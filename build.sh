nvcc main.cu -I$EIGEN_INCLUDE_DIR --expt-relaxed-constexpr -O3 -Xcompiler -march=native -Xcompiler -fopenmp 
