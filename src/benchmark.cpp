#include "benchmark.h"
#include <chrono>
#include <iostream>
#include <memory>

static const int WARMUP_STEPS  = 10;
static const int MEASURE_STEPS = 100;

double BenchmarkRunner::calc_memory_mb(int w, int h) {
    // Current state, next state, and pixel buffer (ARGB = 4 bytes)
    size_t bytes = (size_t)w * h * (1 + 1 + 4);
    return bytes / (1024.0 * 1024.0);
}

BenchmarkResult BenchmarkRunner::time_engine(
    SimulationEngine* engine,
    int grid_w, int grid_h,
    int warmup_steps, int measure_steps)
{
    engine->init(grid_w, grid_h);
    engine->randomize(20, 12345); // deterministic seed for fairness

    // Warm-up
    for (int i = 0; i < warmup_steps; i++) {
        engine->step();
    }

    // Measure
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < measure_steps; i++) {
        engine->step();
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double step_ms  = total_ms / measure_steps;
    double gen_per_sec = 1000.0 / step_ms;

    BenchmarkResult res;
    res.engine_name = engine->get_name();
    res.grid_w = grid_w;
    res.grid_h = grid_h;
    res.step_ms = step_ms;
    res.gen_per_sec = gen_per_sec;
    res.speedup = 0.0;
    res.memory_mb = calc_memory_mb(grid_w, grid_h);

    return res;
}

void BenchmarkRunner::run(const std::vector<std::pair<int,int>>& grid_sizes, bool has_cuda, BenchmarkSuite* suite) {
    if (!suite) return;

    {
        std::lock_guard<std::mutex> lock(suite->mtx);
        suite->results.clear();
        suite->total_steps = grid_sizes.size() * (has_cuda ? 2 : 1);
        suite->current_step = 0;
        suite->is_running = true;
        suite->is_done = false;
    }

    for (const auto& size : grid_sizes) {
        int w = size.first;
        int h = size.second;

        // CPU Benchmark
        auto cpu_engine = create_simulation_engine(false, w, h);
        BenchmarkResult cpu_res = time_engine(cpu_engine.get(), w, h, WARMUP_STEPS, MEASURE_STEPS);
        cpu_res.engine_name = "CPU"; // Standardize name for display

        {
            std::lock_guard<std::mutex> lock(suite->mtx);
            suite->results.push_back(cpu_res);
            suite->current_step++;
        }

        // GPU Benchmark
        if (has_cuda) {
            auto gpu_engine = create_simulation_engine(true, w, h);
            BenchmarkResult gpu_res = time_engine(gpu_engine.get(), w, h, WARMUP_STEPS, MEASURE_STEPS);
            gpu_res.engine_name = "CUDA GPU";

            // Calculate speedup
            if (cpu_res.gen_per_sec > 0) {
                gpu_res.speedup = gpu_res.gen_per_sec / cpu_res.gen_per_sec;
            }

            {
                std::lock_guard<std::mutex> lock(suite->mtx);
                suite->results.push_back(gpu_res);
                suite->current_step++;
            }
        }
    }

    {
        std::lock_guard<std::mutex> lock(suite->mtx);
        suite->is_running = false;
        suite->is_done = true;
    }
}
