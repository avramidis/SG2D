//---------------------------------------------------------------------------//
// Copyright (c) 2020 Eleftherios Avramidis <el.avramidis@gmail.com>
//
// Distributed under The MIT License (MIT)
// See accompanying file LICENSE
//---------------------------------------------------------------------------//

#ifndef SPATIALGILLESPIE2D_SPATIALGILLEPSIE2D_HPP
#define SPATIALGILLESPIE2D_SPATIALGILLEPSIE2D_HPP

#include "sg2d_export.h"
#include <iostream>
#include <chrono>
#include <cmath>
#include <random>

namespace sg2d {

    class SG2D_EXPORT SpatialGillespie2d {
    private:
        int problem_size;
        int t_steps;

        int** s_p;
        int** s_n;
        int** if_p;
        int** if_n;
        double** r_p;
        double** r_n;

        double K;
        double max_distance = 100;
        double p = 0.3;
        double beta = 0.1;
        double m;
        double a = 0.2;
        double t = 0;

    public:
        SpatialGillespie2d();
        ~SpatialGillespie2d();

        double
        calculate_kernel_normalisation(const double p, const double a, const double max_distance);

        void
        calculate_kernel_value_given_cells_locations(const double m, const double p, const double a,
                const double max_distance,
                const int problem_size, double** kernel_value);

        std::pair<int, int>
        event_cell(std::uniform_real_distribution<double>& distribution, std::default_random_engine& generator,
                const double sum_r_p, const int problem_size, double** r_p);

        void
        run_simulation();
    };
}

#endif
