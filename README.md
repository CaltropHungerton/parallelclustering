# parallelclustering

I implemented a GPU parallelized version of k-means clustering as a beginner CUDA C++ project.

Using the well-known iris dataset (150 points) I was able to consistently achieve at least a 100x speedup over my regular k-means implementation.

I also have a short python program that uses an actual clustering library that you can use to verify that it's converging correctly.

To run the CUDA C++ implementation, ensure that you have the CUDA toolkit installed on your computer. You may have to reinstall it so that it is able to work with Visual Studio. Using Visual Studio to open it as a CUDA project is much simpler than compiling with NVCC, but you can do that if you want to as well.

Happy clustering!
