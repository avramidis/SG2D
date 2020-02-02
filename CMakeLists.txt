#---------------------------------------------------------------------------//
# Copyright (c) 2020 Eleftherios Avramidis <el.avramidis@gmail.com>
#
# Distributed under The MIT License (MIT)
# See accompanying file LICENSE
#---------------------------------------------------------------------------//

cmake_minimum_required(VERSION 3.15.3)

project(SG2D
        VERSION 1.0.0
        DESCRIPTION "Spatial epidemiological simulator."
        HOMEPAGE_URL "https://github.com/avramidis/SpatialGillespie2d")

option(SG2D_ENABLE_EXAMPLES "Enable examples" ON)
option(SG2D_ENABLE_TESTS "Enable tests" OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN YES)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    message(STATUS "USING OPTIONS FOR GNU COMPILER")
    set(MY_DEBUG_OPTIONS "-O0;-g;-Wall;-Wextra;-Wpedantic")
    set(MY_RELEASE_OPTIONS "-O3;-Wall;-Wextra;-Wpedantic")
endif ()

if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    message(STATUS "USING OPTIONS FOR MSVC COMPILER")
    set(MY_DEBUG_OPTIONS "/Wall /wd4710")
    set(MY_RELEASE_OPTIONS "/Wall;/wd4710;/wd4711;/wd4514")
endif ()

#add_subdirectory(dependencies)

add_subdirectory(source)
include(GenerateExportHeader)
generate_export_header(sg2d)

if (SG2D_ENABLE_EXAMPLES)
    message(STATUS "Adding examples")
    add_subdirectory(examples)
endif ()