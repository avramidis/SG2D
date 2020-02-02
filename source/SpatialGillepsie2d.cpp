//---------------------------------------------------------------------------//
// Copyright (c) 2020 Eleftherios Avramidis <el.avramidis@gmail.com>
//
// Distributed under The MIT License (MIT)
// See accompanying file LICENSE
//---------------------------------------------------------------------------//

#include <iostream>
#include <chrono>
#include <cmath>
#include <random>

#include "SpatialGillepsie2d.hpp"

using namespace std;

namespace sg2d {
    SpatialGillespie2d::SpatialGillespie2d()
    {
        problem_size = 40;
        t_steps = 10000;

        std::cout << "Problem size: " << problem_size << std::endl;
        std::cout << "Number of simulation steps: " << t_steps << std::endl;

        s_p = new int* [problem_size];
        s_n = new int* [problem_size];
        if_p = new int* [problem_size];
        if_n = new int* [problem_size];
        r_p = new double* [problem_size];
        r_n = new double* [problem_size];

        for (int i = 0; i<problem_size; i++) {
            s_p[i] = new int[problem_size];
            s_n[i] = new int[problem_size];
            if_p[i] = new int[problem_size];
            if_n[i] = new int[problem_size];
            r_p[i] = new double[problem_size];
            r_n[i] = new double[problem_size];
        }

        max_distance = 100;
        p = 0.3;
        beta = 0.1;
        a = 0.2;
        t = 0;
    }

    SpatialGillespie2d::~SpatialGillespie2d()
    {

    }

    void
    SpatialGillespie2d::run_simulation()
    {
        std::chrono::time_point<std::chrono::steady_clock> start;
        std::chrono::time_point<std::chrono::steady_clock> start_sim_loop;
        std::chrono::time_point<std::chrono::steady_clock> end;
        std::chrono::time_point<std::chrono::steady_clock> timer_event_cell_start, timer_event_cell_finish;
        std::chrono::time_point<std::chrono::steady_clock> timer_update_other_cells_start, timer_update_other_cells_finish;
        double timer_initial_rates;
        double timer_end_sim_loop;
        double timer_event_cell = 0;
        double timer_update_other_cells = 0;

        std::random_device r;

        std::default_random_engine generator(r());
        std::uniform_real_distribution<double> distribution(0.0, 1.0);

        m = calculate_kernel_normalisation(p, a, max_distance);
        std::cout << "m: " << m << std::endl;

        auto** kernel_value = new double* [problem_size];
        for (int i = 0; i<problem_size; i++) {
            kernel_value[i] = new double[problem_size];
        }
        calculate_kernel_value_given_cells_locations(m, p, a, max_distance, problem_size, kernel_value);

        for (int i = 0; i<problem_size; i++) {
            for (int j = 0; j<problem_size; j++) {
                s_p[i][j] = 100;
                s_n[i][j] = 100;
                if_p[i][j] = 0;
                if_n[i][j] = 0;
            }
        }

        for (int i = 0; i<problem_size; i = i+2) {
            for (int j = 0; j<problem_size; j = j+2) {
                s_p[i][j] = 99;
                s_n[i][j] = 99;
                if_p[i][j] = 1;
                if_n[i][j] = 1;
            }
        }

        start = chrono::steady_clock::now();

#pragma omp parallel for default(none) shared(problem_size, if_p, s_p, r_p, max_distance, beta, p, a, m) private(K) collapse(2)
        for (int i = 0; i<problem_size; i++) {
            for (int j = 0; j<problem_size; j++) {
                r_p[i][j] = 0;
                r_p[i][j] = r_p[i][j]+beta*if_p[i][j]*p;
                for (int ii = 0; ii<problem_size; ii++) {
                    for (int jj = 0; jj<problem_size; jj++) {
                        if (if_p[ii][jj]>0) {
                            double distance = sqrt(pow(i-ii, 2)+pow(j-jj, 2));
                            K = 0;
                            if (0==distance) {
                                K = p;
                            }

                            if (0<distance && distance<=max_distance) {
                                K = m*exp(-a*distance);
                            }

                            if (distance>max_distance) {
                                K = 0;
                            }

                            r_p[i][j] = r_p[i][j]+beta*if_p[ii][jj]*K;
                        }
                    }
                }
            }
        }

#pragma omp parallel for default(none) shared(problem_size, r_p, r_n, s_p) collapse(2)
        for (int i = 0; i<problem_size; i++) {
            for (int j = 0; j<problem_size; j++) {
                r_p[i][j] = r_p[i][j]*s_p[i][j];
                r_n[i][j] = r_p[i][j];
            }
        }

        end = chrono::steady_clock::now();
        timer_initial_rates = chrono::duration_cast<chrono::milliseconds>(end-start).count()/1000.0;

        std::cout << r_p[0][0] << std::endl;

        start_sim_loop = chrono::steady_clock::now();

        for (int k = 0; k<t_steps; k++) {

            double sum_r_p = 0;
#pragma omp parallel for reduction(+:sum_r_p) default(none) shared(problem_size, r_p) collapse(2)
            for (int i = 0; i<problem_size; i++) {
                for (int j = 0; j<problem_size; j++) {
                    sum_r_p += r_p[i][j];
                }
            }

            if (sum_r_p==0) {
                std::cout << "All infected!!" << std::endl;
                break;
            }

            double t_jump = -log(distribution(generator))/sum_r_p;
            t = t+t_jump;

            timer_event_cell_start = chrono::steady_clock::now();
            std::pair<int, int> idx = event_cell(distribution, generator, sum_r_p, problem_size, r_p);
            timer_event_cell_finish = chrono::steady_clock::now();
            timer_event_cell = timer_event_cell+chrono::duration_cast<chrono::nanoseconds>(
                    timer_event_cell_finish-timer_event_cell_start).count()*1e-9;

            s_n[idx.first][idx.second] = s_p[idx.first][idx.second]-1;
            if_n[idx.first][idx.second] = if_p[idx.first][idx.second]+1;

            // Update infection rate of event cell
            K = p;
            if (s_n[idx.first][idx.second]>0) {
                r_n[idx.first][idx.second] =
                        r_p[idx.first][idx.second]-(r_p[idx.first][idx.second]/s_p[idx.first][idx.second])+
                                s_n[idx.first][idx.second]*beta*K;
            }
            else {
                r_n[idx.first][idx.second] = 0;
            }

            timer_update_other_cells_start = chrono::steady_clock::now();
            // Update infection rate of other cells
#pragma omp parallel for default(none) shared(idx, problem_size, r_p, r_n, s_p, s_n, beta, m, a, max_distance, kernel_value) private(K) collapse(2)
            for (int i = 0; i<problem_size; i++) {
                for (int j = 0; j<problem_size; j++) {
                    if (idx.first!=i || idx.second!=j) {
                        if (s_n[idx.first][idx.second]>0) {
                            K = kernel_value[abs(i-idx.first)][abs(j-idx.second)];
                            r_n[i][j] = r_p[i][j]+beta*s_p[i][j]*K;
                        }
                        else {
                            r_n[i][j] = 0;
                        }
                    }
                }
            }
            timer_update_other_cells_finish = chrono::steady_clock::now();
            timer_update_other_cells = timer_update_other_cells+chrono::duration_cast<chrono::nanoseconds>(
                    timer_update_other_cells_finish-timer_update_other_cells_start).count()*1e-9;

#pragma omp parallel for default(none) shared(problem_size, r_p, r_n, s_p, s_n, if_p, if_n) collapse(2)
            for (int i = 0; i<problem_size; i++) {
                for (int j = 0; j<problem_size; j++) {
                    r_p[i][j] = r_n[i][j];
                    s_p[i][j] = s_n[i][j];
                    if_p[i][j] = if_n[i][j];
                }
            }
        }

        end = chrono::steady_clock::now();
        timer_end_sim_loop = chrono::duration_cast<chrono::milliseconds>(end-start_sim_loop).count()/1000.0;

        end = chrono::steady_clock::now();
        cout << "Elapsed time in seconds: "
             << chrono::duration_cast<chrono::milliseconds>(end-start).count()/1000.0
             << endl;

        std::cout << "timer_initial_rates: " << timer_initial_rates << std::endl;
        std::cout << "timer_end_sim_loop: " << timer_end_sim_loop << std::endl;
        std::cout << "timer_event_cell: " << timer_event_cell << std::endl;
        std::cout << "timer_update_other_cells: " << timer_update_other_cells << std::endl;

        std::cout << r_p[0][0] << std::endl;
        std::cout << r_n[0][0] << std::endl;
        std::cout << if_p[0][0] << std::endl;
        std::cout << if_n[0][0] << std::endl;
        std::cout << s_p[0][0] << std::endl;
        std::cout << s_n[0][0] << std::endl;

        int n_infected = 0;
        int n_susceptible = 0;
        for (int i = 0; i<problem_size; i++) {
            for (int j = 0; j<problem_size; j++) {
                n_infected += if_n[i][j];
                n_susceptible += s_n[i][j];
            }
        }
        std::cout << "Number of infected: " << n_infected << std::endl;
        std::cout << "Number of susceptible: " << n_susceptible << std::endl;
    }

    double
    SpatialGillespie2d::calculate_kernel_normalisation(double const p, double const a, double const max_distance)
    {

        auto** area = new double* [int(max_distance*2)];
        for (int i = 0; i<int(max_distance*2); i++) {
            area[i] = new double[int(max_distance*2)];
        }

        for (int i = 0; i<max_distance*2; i++) {
            for (int j = 0; j<max_distance*2; j++) {
                if (i==j) {
                    area[i][j] = p;
                }
                else {
                    double distance = sqrt(pow(i-max_distance, 2)+pow(j-max_distance, 2));
                    if (0<distance && distance<=max_distance) {
                        area[i][j] = exp(-a*distance);
                    }

                    if (distance>max_distance) {
                        area[i][j] = 0;
                    }
                }
            }
        }

        double area_sum = 0;
        for (int i = 0; i<max_distance*2; i++) {
            for (int j = 0; j<max_distance*2; j++) {
                area_sum += area[i][j];
            }
        }

        delete[] area;

        return 1.00/area_sum;
    }

    void
    SpatialGillespie2d::calculate_kernel_value_given_cells_locations(double const m, double const p, double const a,
            double const max_distance,
            int const problem_size,
            double** kernel_value)
    {

        for (int i = 0; i<problem_size; i++) {
            for (int j = 0; j<problem_size; j++) {
                double distance = sqrt(pow(i, 2)+pow(j, 2));
                kernel_value[i][j] = 0;
                if (distance==0) {
                    kernel_value[i][j] = p;
                }

                if (0<distance && distance<=max_distance) {
                    kernel_value[i][j] = m*exp(-a*distance);
                }

                if (distance>max_distance) {
                    kernel_value[i][j] = 0;
                }
            }
        }
    }

    std::pair<int, int>
    SpatialGillespie2d::event_cell(std::uniform_real_distribution<double>& distribution,
            std::default_random_engine& generator,
            double const sum_r_p,
            int const problem_size, double** r_p)
    {

        double r = sum_r_p*distribution(generator);
        std::pair<int, int> idx;
        for (int i = 0; i<problem_size; i++) {
            for (int j = 0; j<problem_size; j++) {
                if (r_p[i][j]>0) {
                    r = r-r_p[i][j];
                    if (r<=0) {
                        idx.first = i;
                        idx.second = j;
                        return idx;
                    }
                }
            }
        }

        return idx;
    }
    
}