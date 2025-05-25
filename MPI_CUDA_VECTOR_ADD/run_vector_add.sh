#!/bin/bash
#SBATCH --job-name=vector_add_mpi       # Job name
#SBATCH --output=vector_add_%j.out      # Output file
#SBATCH --error=vector_add_%j.err       # Error file
#SBATCH --time=01:00:00                 # Wall time (HH:MM:SS)
#SBATCH --partition=gpu                 # Partition name
#SBATCH --gres=gpu:4                    # Number of GPUs
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=4                      # Total number of MPI tasks
#SBATCH --ntasks-per-node=4             # Tasks per node
#SBATCH --cpus-per-task=1               # CPU cores per task
#SBATCH --mem=8G                        # Memory per node

# Load modules
module load GCC/12.2.0 NVHPC/22.7-CUDA-11.7.0 OpenMPI/4.1.4

# Define the output file
OUTPUT_FILE="vector_add_all_output.txt"

# Add header to the output file before running any tests
echo "------------------------- Thread Count: 128 -----------------------------" >> $OUTPUT_FILE
mpirun -np 1 ./mpi_cuda_vector_add 128 >> $OUTPUT_FILE
mpirun -np 2 ./mpi_cuda_vector_add 128 >> $OUTPUT_FILE
mpirun -np 3 ./mpi_cuda_vector_add 128 >> $OUTPUT_FILE
mpirun -np 4 ./mpi_cuda_vector_add 128 >> $OUTPUT_FILE

echo "------------------------- Thread Count: 256 -----------------------------" >> $OUTPUT_FILE
mpirun -np 1 ./mpi_cuda_vector_add 256 >> $OUTPUT_FILE
mpirun -np 2 ./mpi_cuda_vector_add 256 >> $OUTPUT_FILE
mpirun -np 3 ./mpi_cuda_vector_add 256 >> $OUTPUT_FILE
mpirun -np 4 ./mpi_cuda_vector_add 256 >> $OUTPUT_FILE

echo "------------------------- Thread Count: 512 -----------------------------" >> $OUTPUT_FILE
mpirun -np 1 ./mpi_cuda_vector_add 512 >> $OUTPUT_FILE
mpirun -np 2 ./mpi_cuda_vector_add 512 >> $OUTPUT_FILE
mpirun -np 3 ./mpi_cuda_vector_add 512 >> $OUTPUT_FILE
mpirun -np 4 ./mpi_cuda_vector_add 512 >> $OUTPUT_FILE

echo "------------------------- Thread Count: 1024 -----------------------------" >> $OUTPUT_FILE
mpirun -np 1 ./mpi_cuda_vector_add 1024 >> $OUTPUT_FILE
mpirun -np 2 ./mpi_cuda_vector_add 1024 >> $OUTPUT_FILE
mpirun -np 3 ./mpi_cuda_vector_add 1024 >> $OUTPUT_FILE
mpirun -np 4 ./mpi_cuda_vector_add 1024 >> $OUTPUT_FILE
