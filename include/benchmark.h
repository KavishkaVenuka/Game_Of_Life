#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <string>
#include <vector>
#include <mutex>
#include "engine.h"

struct BenchmarkResult {
    std::string engine_name;
    int grid_w, grid_h;
    double gen_per_sec;
    double step_ms;
    double speedup; // 0.0 for CPU, > 1.0 for GPU if faster
    double memory_mb;
};

struct BenchmarkSuite {
    std::vector<BenchmarkResult> results;
    bool is_running = false;
    bool is_done    = false;
    int  current_step = 0;
    int  total_steps  = 0;
    std::mutex mtx; // Protects access to results across threads
};

class BenchmarkRunner {
public:
    static void run(
        const std::vector<std::pair<int,int>>& grid_sizes,
        bool has_cuda,
        BenchmarkSuite* suite
    );

private:
    static BenchmarkResult time_engine(
        SimulationEngine* engine,
        int grid_w, int grid_h,
        int warmup_steps, int measure_steps
    );
    static double calc_memory_mb(int grid_w, int grid_h);
};

#endif // BENCHMARK_H
