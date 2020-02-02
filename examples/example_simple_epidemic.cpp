//---------------------------------------------------------------------------//
// Copyright (c) 2020 Eleftherios Avramidis <el.avramidis@gmail.com>
//
// Distributed under The MIT License (MIT)
// See accompanying file LICENSE
//---------------------------------------------------------------------------//

#include <iostream>
#include "SpatialGillepsie2d.hpp"

int
main(int argc, char* argv[])
{

    sg2d::SpatialGillespie2d spatial_gillespie_2_d;

    spatial_gillespie_2_d.run_simulation();

    return 0;
}