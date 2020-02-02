#---------------------------------------------------------------------------//
# Copyright (c) 2020 Eleftherios Avramidis <el.avramidis@gmail.com>
#
# Distributed under The MIT License (MIT)
# See accompanying file LICENSE
#---------------------------------------------------------------------------//

include(GNUInstallDirs)
add_library(sg2d SHARED)

target_sources(sg2d PRIVATE
        SpatialGillepsie2d.cpp)

target_include_directories(sg2d PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>/..
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

set_target_properties(sg2d PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

#target_link_libraries(sg2d PRIVATE cxxplot)
#target_link_libraries(sg2d PRIVATE Chronos)

if (WIN32)
    message(STATUS "WINDOWS SYSTEM!!!")
    target_compile_definitions(sg2d PUBLIC sg2d_EXPORT)

    if (SG2D_ENABLE_EXAMPLES)
        message(STATUS "WINDOWS SYSTEM!!!")
        add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/examples/sg2d.dll
                COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:sg2d>" "$<TARGET_FILE_DIR:example_simple_epidemic>"
                VERBATIM)
        add_custom_target(copyddlexamples ALL
                DEPENDS ${CMAKE_BINARY_DIR}/examples/sg2d.dll sg2d
                VERBATIM)
    endif ()

    if (ENABLE_TESTS)
        #    add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/tests/cxxplot.dll
        #            COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:cxxplot>" "$<TARGET_FILE_DIR:test_timer>"
        #            VERBATIM)
        #    add_custom_target(copyddltests ALL
        #            DEPENDS ${CMAKE_BINARY_DIR}/tests/cxxplot.dll cxxplot
        #            VERBATIM)
    endif ()
endif ()

install(TARGETS sg2d EXPORT sg2dTargets
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT sg2d_RunTime
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT sg2d_RunTime NAMELINK_COMPONENT sg2d_Development
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT sg2d_Development
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
        TYPE INCLUDE
        FILES_MATCHING
        PATTERN *.hpp
        PERMISSIONS OWNER_READ OWNER_WRITE
        GROUP_READ
        WORLD_READ)

install(FILES ${CMAKE_BINARY_DIR}/sg2d_export.h
        TYPE INCLUDE
        PERMISSIONS OWNER_READ OWNER_WRITE
        GROUP_READ
        WORLD_READ)

install(EXPORT sg2dTargets
        NAMESPACE sg2d::
        FILE sg2dConfig.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sg2d)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
        cxxplotConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion)

install(FILES ${PROJECT_BINARY_DIR}/source/sg2dConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sg2d)