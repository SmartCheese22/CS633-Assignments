# CS633 - Parallel Computing Assignments

This repository contains assignments and projects completed for the CS633 (Parallel Computing) course, 2024-25. The primary focus is on implementing and optimizing parallel algorithms using MPI (Message Passing Interface) for solving computational problems on distributed systems.

## 📋 Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Dependencies](#dependencies)
- [Compilation](#compilation)
- [Usage](#usage)
- [Performance Analysis](#performance-analysis)
- [Results](#results)
- [License](#license)
- [Author](#author)

## 🎯 Overview

The main project implements a parallel algorithm to find local and global extrema (minima and maxima) in large-scale 3D grid datasets. The implementation uses MPI for distributed memory parallelism, enabling efficient computation across multiple processes with 3D domain decomposition.

### Key Features

- **3D Domain Decomposition**: Distributes data across multiple processes in all three dimensions
- **Ghost Cell Communication**: Implements efficient halo exchange for boundary data
- **Multiple Implementations**: Includes baseline, optimized, and final versions
- **Performance Benchmarking**: Comprehensive scaling analysis and performance metrics
- **Time-series Support**: Handles multiple time steps in a single execution

## 📁 Repository Structure

```
CS633-Assignments/
├── README.md                              # This file
├── Assignment.pdf                         # Assignment specification
├── LICENSE                                # MIT License
│
├── src.c                                  # Baseline implementation
├── optmised_src.c                        # Optimized version with improved performance
├── final_src.c                           # Final optimized implementation
├── src_c.c                               # Alternative implementation
├── src_test.c                            # Test version
│
├── data_64_64_64_3.bin                   # Binary dataset (64×64×64 grid, 3 timesteps)
├── data_64_64_64_3.txt                   # Text dataset
├── data_64_64_96_7.bin                   # Binary dataset (64×64×96 grid, 7 timesteps)
├── data_64_64_96_7.txt                   # Text dataset
│
├── scaling_results.csv                   # Performance scaling data
├── data_64_64_64_3_scaling_boxplot.png   # Scaling visualization
├── data_64_64_96_7_scaling_boxplot.png   # Scaling visualization
├── strong_scaling_boxplot.png            # Strong scaling analysis
└── strong_scaling_plot.png               # Strong scaling plot
```

## 🔧 Dependencies

### Required
- **MPI Implementation**: OpenMPI (3.0+) or MPICH (3.2+)
- **C Compiler**: GCC 7.0+ or any C99-compatible compiler
- **System**: Linux/Unix environment with MPI support

### Optional (for visualization)
- Python 3.x with matplotlib (for generating plots)
- Pandas (for data analysis)

## 🛠 Compilation

### Basic Compilation
```bash
mpicc -o src.x src.c -O3 -lm
```

### Optimized Version
```bash
mpicc -o optmised_src.x optmised_src.c -O3 -lm
```

### Final Version
```bash
mpicc -o final_src.x final_src.c -O3 -lm
```

### Compilation Flags
- `-O3`: Maximum optimization level
- `-lm`: Link math library
- Additional flags for specific architectures:
  ```bash
  mpicc -o src.x src.c -O3 -march=native -mtune=native -lm
  ```

## 🚀 Usage

### Basic Execution
```bash
mpirun -np <num_processes> ./src.x <dataset> <PX> <PY> <PZ> <NX> <NY> <NZ> <NC> <outputfile>
```

### Parameters
- `<num_processes>`: Total number of MPI processes (must equal PX × PY × PZ)
- `<dataset>`: Input data file (e.g., `data_64_64_64_3.txt`)
- `<PX>`: Number of processes in X dimension
- `<PY>`: Number of processes in Y dimension
- `<PZ>`: Number of processes in Z dimension
- `<NX>`: Grid size in X dimension
- `<NY>`: Grid size in Y dimension
- `<NZ>`: Grid size in Z dimension
- `<NC>`: Number of time steps
- `<outputfile>`: Output file for results

### Example Commands

#### 8 Processes (2×2×2 decomposition)
```bash
mpirun -np 8 ./src.x data_64_64_64_3.txt 2 2 2 64 64 64 3 output_8p.txt
```

#### 16 Processes (4×2×2 decomposition)
```bash
mpirun -np 16 ./src.x data_64_64_64_3.txt 4 2 2 64 64 64 3 output_16p.txt
```

#### 64 Processes (4×4×4 decomposition)
```bash
mpirun -np 64 ./src.x data_64_64_96_7.txt 4 4 4 64 64 96 7 output_64p.txt
```

### Output Format

The output file contains three lines:
1. **Local Extrema Counts**: `(min_count_t0, max_count_t0), (min_count_t1, max_count_t1), ...`
2. **Global Extrema Values**: `(global_min_t0, global_max_t0), (global_min_t1, global_max_t1), ...`
3. **Timing Information**: `read_time, computation_time, total_time` (in seconds)

Example output:
```
(125, 143), (98, 156), (110, 132)
(0.125000, 9.875000), (0.250000, 9.750000), (0.312500, 9.625000)
0.125000, 0.875000, 1.000000
```

## 📊 Performance Analysis

### Strong Scaling Results

The implementation demonstrates the following performance characteristics:

#### Dataset: 64×64×64 (3 timesteps)
| Processes | Average Runtime (s) | Speedup | Efficiency |
|-----------|---------------------|---------|------------|
| 8         | 0.931               | 1.00x   | 100%       |
| 16        | 2.429               | 0.38x   | 24%        |
| 32        | 4.178               | 0.22x   | 14%        |
| 64        | 5.168               | 0.18x   | 11%        |

#### Dataset: 64×64×96 (7 timesteps)
| Processes | Average Runtime (s) | Speedup | Efficiency |
|-----------|---------------------|---------|------------|
| 8         | 3.147               | 1.00x   | 100%       |
| 16        | 4.553               | 0.69x   | 43%        |
| 32        | 6.531               | 0.48x   | 30%        |
| 64        | 6.881               | 0.46x   | 29%        |

### Performance Insights

1. **Communication Overhead**: As the number of processes increases, communication overhead dominates for small problem sizes
2. **Load Balancing**: 3D decomposition provides better load balance than 1D or 2D approaches
3. **Scalability**: Larger datasets show better scaling characteristics
4. **Ghost Cell Overhead**: Efficient MPI derived datatypes minimize memory overhead for ghost cell exchanges

### Visualization

Performance plots are available in the repository:
- `strong_scaling_plot.png`: Overall strong scaling behavior
- `strong_scaling_boxplot.png`: Runtime distribution across multiple runs
- `data_64_64_64_3_scaling_boxplot.png`: Scaling for smaller dataset
- `data_64_64_96_7_scaling_boxplot.png`: Scaling for larger dataset

## 📈 Results

### Algorithm Correctness

The implementation correctly identifies:
- **Local Minima**: Points where all 6 neighboring cells have higher values
- **Local Maxima**: Points where all 6 neighboring cells have lower values
- **Global Extrema**: Minimum and maximum values across the entire grid

### Key Implementation Features

1. **3D Domain Decomposition**: Data distributed across PX × PY × PZ process grid
2. **Ghost Cell Layer**: Single layer of ghost cells for neighbor communication
3. **Non-blocking Communication**: Uses `MPI_Isend`/`MPI_Irecv` for overlap
4. **MPI Derived Datatypes**: Efficient slice/plane exchange without packing
5. **Uneven Distribution Handling**: Supports grids not evenly divisible by process counts

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Prathamesh Baviskar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

## 👤 Author

**Prathamesh Baviskar**
- GitHub: [@SmartCheese22](https://github.com/SmartCheese22)
- Course: CS633 - Parallel Computing (2024-25)

## 🎓 Course Information

**Course**: CS633 - Parallel Computing  
**Academic Year**: 2024-25  
**Institution**: IIT Bombay

## 🤝 Contributing

This is an academic assignment repository. While contributions are not expected, suggestions and discussions about the implementation are welcome through issues.

## 📚 References

- MPI Standard Documentation: https://www.mpi-forum.org/
- OpenMPI Documentation: https://www.open-mpi.org/doc/
- Parallel Programming Course Materials (CS633)

## 📝 Notes

- Performance results may vary based on hardware, network topology, and MPI implementation
- For optimal performance on specific architectures, consider tuning compilation flags
- Large-scale runs require sufficient memory per process for local data and ghost cells
- The implementation assumes a 6-connected neighborhood (face neighbors only)

---

**Last Updated**: November 2025  
**Status**: Assignment Complete ✅
