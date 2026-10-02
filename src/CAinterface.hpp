// Copyright Lawrence Livermore National Security, LLC and other ExaCA Project Developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#ifndef EXACA_INTERFACE_HPP
#define EXACA_INTERFACE_HPP

#include "CAcelldata.hpp"
#include "CAconfig.hpp"
#include "CAorientation.hpp"
#include "CAparsefiles.hpp"
#include "CAtemperature.hpp"
#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// Data representing the active cells at the solid-liquid interface, including MPI buffers
template <typename MemorySpace>
struct Interface {

    using memory_space = MemorySpace;
    using view_type_buffer = Kokkos::View<float **, memory_space>;
    using view_type_float = Kokkos::View<float *, memory_space>;
    using view_type_int = Kokkos::View<int *, memory_space>;
    using view_type_int_host = typename view_type_int::host_mirror_type;
    using neighbor_list_type = Kokkos::Array<int, 26>;
    using periodic_bc_type = Kokkos::Array<int, 2>;
    using periodic_bc_corner_type = Kokkos::Array<int, 4>;

    // Using the default exec space for this memory space.
    using execution_space = typename memory_space::execution_space;

    // Size of send/recv buffers
    int buf_size, buf_components;
    view_type_float diagonal_length, octahedron_center, crit_diagonal_length;
    view_type_buffer buffer_south_send, buffer_north_send, buffer_south_recv, buffer_north_recv;
    view_type_int send_size_south, send_size_north, steering_vector, num_steer;
    view_type_int_host send_size_south_host, send_size_north_host, num_steer_host;
    // Initial size of new octahedra
    float _init_oct_size;

    // Neighbor lists
    neighbor_list_type neighbor_x, neighbor_y, neighbor_z;

    // Periodic buffer data
    bool is_periodic;
    periodic_bc_type coord_x_interior, coord_x_exterior, coord_y_interior, coord_y_exterior, octahedron_center_offset_x,
        octahedron_center_offset_y;
    periodic_bc_corner_type corner_x_interior, corner_x_exterior, corner_y_interior, corner_y_exterior;
    // Parallel dispatch tags.
    struct RefillBuffersTag {};

    // Constructor for views and view bounds for current layer
    // Use default initialization to 0 for num_steer_host and num_steer and buffer counts
    Interface(const int id, const int domain_size, const float init_oct_size, const int nx, const int ny,
              const bool _is_periodic = false, const int buf_size_initial_estimate = 25,
              const int buf_components_temp = 9)
        : diagonal_length(view_type_float(Kokkos::ViewAllocateWithoutInitializing("diagonal_length"), domain_size))
        , octahedron_center(
              view_type_float(Kokkos::ViewAllocateWithoutInitializing("octahedron_center"), 3 * domain_size))
        , crit_diagonal_length(
              view_type_float(Kokkos::ViewAllocateWithoutInitializing("crit_diagonal_length"), 26 * domain_size))
        , buffer_south_send(view_type_buffer(Kokkos::ViewAllocateWithoutInitializing("buffer_south_send"),
                                             buf_size_initial_estimate, buf_components_temp))
        , buffer_north_send(view_type_buffer(Kokkos::ViewAllocateWithoutInitializing("buffer_north_send"),
                                             buf_size_initial_estimate, buf_components_temp))
        , buffer_south_recv(view_type_buffer(Kokkos::ViewAllocateWithoutInitializing("buffer_south_recv"),
                                             buf_size_initial_estimate, buf_components_temp))
        , buffer_north_recv(view_type_buffer(Kokkos::ViewAllocateWithoutInitializing("buffer_north_recv"),
                                             buf_size_initial_estimate, buf_components_temp))
        , send_size_south(view_type_int("send_size_south", 1))
        , send_size_north(view_type_int("send_size_north", 1))
        , steering_vector(view_type_int(Kokkos::ViewAllocateWithoutInitializing("steering_vector"), domain_size))
        , num_steer(view_type_int("steering_vector_size", 1))
        , send_size_south_host(view_type_int_host("send_size_south_host", 1))
        , send_size_north_host(view_type_int_host("send_size_north_host", 1))
        , num_steer_host(view_type_int_host("steering_vector_size_host", 1))
        , _init_oct_size(init_oct_size) {

        // Set initial buffer size to the estimate
        buf_size = buf_size_initial_estimate;
        // Set number of components in the buffer
        buf_components = buf_components_temp;
        // Send/recv buffers for ghost node data should be initialized with -1s in the first index as placeholders for
        // empty positions in the buffer, and with send size counts of 0
        resetBuffers();
        // Initialize neighbor lists for iterating over active cells
        neighborListInit();
        // If boundary conditions are periodic, initialize offset and coordinate swap views
        is_periodic = _is_periodic;
        if (is_periodic) {
            // X locations of cell data to be copied from
            coord_x_interior[0] = 1;
            coord_x_interior[1] = nx - 2;
            // X locations of cell data to be copied to
            coord_x_exterior[0] = nx - 1;
            coord_x_exterior[1] = 0;
            // Y locations of cell data to be copied from
            coord_y_interior[0] = 1;
            coord_y_interior[1] = ny - 2;
            // Y locations of cell data to be copied to
            coord_y_exterior[0] = ny - 1;
            coord_y_exterior[1] = 0;
            // XY corners to be copied form
            corner_x_interior[0] = coord_x_interior[0];
            corner_x_interior[1] = coord_x_interior[0];
            corner_x_interior[2] = coord_x_interior[1];
            corner_x_interior[3] = coord_x_interior[1];
            corner_y_interior[0] = coord_y_interior[0];
            corner_y_interior[1] = coord_y_interior[1];
            corner_y_interior[2] = coord_y_interior[0];
            corner_y_interior[3] = coord_y_interior[1];
            // XY corners to be copied to
            corner_x_exterior[0] = coord_x_exterior[0];
            corner_x_exterior[1] = coord_x_exterior[0];
            corner_x_exterior[2] = coord_x_exterior[1];
            corner_x_exterior[3] = coord_x_exterior[1];
            corner_y_exterior[0] = coord_y_exterior[0];
            corner_y_exterior[1] = coord_y_exterior[1];
            corner_y_exterior[2] = coord_y_exterior[0];
            corner_y_exterior[3] = coord_y_exterior[1];
            // Offsets for X locations of octahedra centers
            octahedron_center_offset_x[0] = nx - 2;
            octahedron_center_offset_x[1] = -(nx - 2);
            // Offsets for Y locations of octahedra centers
            octahedron_center_offset_y[0] = ny - 2;
            octahedron_center_offset_y[1] = -(ny - 2);
        }
        if (id == 0)
            std::cout << "Done with interface initialization" << std::endl;
    }

    // Set first index in send buffers to -1 (placeholder) for all cells in the buffer, and reset the counts of number
    // of cells contained in buffers to 0s
    void resetBuffers() {

        auto buffer_north_send_local = buffer_north_send;
        auto buffer_south_send_local = buffer_south_send;
        auto send_size_north_local = send_size_north;
        auto send_size_south_local = send_size_south;
        Kokkos::parallel_for(
            "BufferReset", buf_size, KOKKOS_LAMBDA(const int &i) {
                buffer_north_send_local(i, 0) = -1.0;
                buffer_south_send_local(i, 0) = -1.0;
            });
        Kokkos::parallel_for(
            "HaloCountReset", 1, KOKKOS_LAMBDA(const int) {
                send_size_north_local(0) = 0;
                send_size_south_local(0) = 0;
            });
    }

    // Initialize neighbor list structures (neighbor_x, neighbor_y, neighbor_z)
    void neighborListInit() {

        // Neighbors 0 through 5 are nearest neighbors, 6 through 17 are second nearest neighbors, and 18 through 25 are
        // third nearest neighbors
        neighbor_x = {1, 0, 0, -1, 0, 0, 1, 1, 0, -1, -1, 0, -1, -1, 0, 1, 1, 0, 1, -1, 1, 1, -1, -1, 1, -1};
        neighbor_y = {0, 1, 0, 0, -1, 0, 1, 0, 1, -1, 0, -1, 1, 0, -1, -1, 0, 1, 1, 1, -1, 1, -1, 1, -1, -1};
        neighbor_z = {0, 0, 1, 0, 0, -1, 0, 1, 1, 0, -1, -1, 0, 1, 1, 0, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1};
    }

    // Increase size of buffers if necessary, returning the new buffer size. Return true if the buffers were resized
    int resizeBuffers(const int id, const int cycle, const int num_cells_buffer_padding = 25) {

        bool resize_performed = false;
        int old_buf_size = buf_size;
        send_size_north_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), send_size_north);
        send_size_south_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), send_size_south);
        int max_count_local = Kokkos::max(send_size_north_host(0), send_size_south_host(0));
        int max_count_global;
        MPI_Allreduce(&max_count_local, &max_count_global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (max_count_global > old_buf_size) {
            // Increase buffer size to fit all data
            // Add num_cells_buffer_padding (defaults to 25) cells as additional padding
            int new_buf_size = max_count_global + num_cells_buffer_padding;
            Kokkos::resize(buffer_north_send, new_buf_size, buf_components);
            Kokkos::resize(buffer_south_send, new_buf_size, buf_components);
            Kokkos::resize(buffer_north_recv, new_buf_size, buf_components);
            Kokkos::resize(buffer_south_recv, new_buf_size, buf_components);
            // Reset count variables on device to the old buffer size
            auto send_size_north_local = send_size_north;
            auto send_size_south_local = send_size_south;
            auto old_buf_size_local = old_buf_size;
            auto buffer_north_send_local = buffer_north_send;
            auto buffer_south_send_local = buffer_south_send;
            auto buf_components_local = buf_components;
            Kokkos::parallel_for(
                "ResetCounts", 1, KOKKOS_LAMBDA(const int &) {
                    send_size_north_local(0) = old_buf_size_local;
                    send_size_south_local(0) = old_buf_size_local;
                });
            // Set -1 values for the new (currently empty) positions in the resized buffer
            Kokkos::parallel_for(
                "InitNewBufCapacity", Kokkos::RangePolicy<>(old_buf_size_local, new_buf_size),
                KOKKOS_LAMBDA(const int &buf_position) {
                    for (int buf_comp = 0; buf_comp < buf_components_local; buf_comp++) {
                        buffer_north_send_local(buf_position, buf_comp) = -1.0;
                        buffer_south_send_local(buf_position, buf_comp) = -1.0;
                    }
                });
            buf_size = new_buf_size;
            resize_performed = true;
            if (id == 0)
                std::cout << "On time step " << cycle << ", resized the send/recv buffers to " << buf_size << std::endl;
        }
        return resize_performed;
    }

    // Resize and reinitialize structs governing the active cells before the next layer of a multilayer problem. Realloc
    // is used as the old values from the structs are not needed
    void initNextLayer(const int domain_size) {

        // Realloc steering vector as domain_size for the next layer may be different
        Kokkos::realloc(steering_vector, domain_size);

        // Realloc active cell data structure and halo regions
        Kokkos::realloc(diagonal_length, domain_size);
        Kokkos::realloc(octahedron_center, 3 * domain_size);
        Kokkos::realloc(crit_diagonal_length, 26 * domain_size);

        // Reset active cell data structures to zeros
        Kokkos::deep_copy(diagonal_length, 0);
        Kokkos::deep_copy(octahedron_center, 0);
        Kokkos::deep_copy(crit_diagonal_length, 0);
    }

    // Assign octahedron a small initial size, and a center location
    // Note that the Y coordinate is relative to the domain origin to keep the coordinate system continuous across ranks
    KOKKOS_INLINE_FUNCTION
    void createNewOctahedron(const int index, const int (&cell_location)[3], const int y_offset) const {
        diagonal_length(index) = _init_oct_size;
        octahedron_center(3 * index) = cell_location[0] + 0.5;
        octahedron_center(3 * index + 1) = cell_location[1] + y_offset + 0.5;
        octahedron_center(3 * index + 2) = cell_location[2] + 0.5;
    }

    // Create a new octahedron based on capture of cell centered at (xp,yp,zp) from cell at index, in capture direction
    // l, updating the octahedra_data struct
    template <typename ViewType>
    KOKKOS_INLINE_FUNCTION void createNewOctahedron(float (&octahedron_data)[4], const int l, const int index,
                                                    const float xp, const float yp, const float zp,
                                                    const ViewType grain_unit_vector, const int my_orientation,
                                                    const int my_phase_id, const int neighbor_index) const {
        // (cxold, cyold, czold) are the coordinates of this decentered octahedron
        const float cxold = octahedron_center(3 * index);
        const float cyold = octahedron_center(3 * index + 1);
        const float czold = octahedron_center(3 * index + 2);

        // (x0,y0,z0) is a vector pointing from this decentered octahedron center to the image
        // of the center of the new cell
        const float x0 = xp - cxold;
        const float y0 = yp - cyold;
        const float z0 = zp - czold;

        // Calculate unit vectors for the octahedron that intersect the new cell center
        const int angle_1_pos = ((grain_unit_vector(9 * my_orientation, my_phase_id) * x0 +
                                  grain_unit_vector(9 * my_orientation + 1, my_phase_id) * y0 +
                                  grain_unit_vector(9 * my_orientation + 2, my_phase_id) * z0) > 0);
        const int angle_2_pos = ((grain_unit_vector(9 * my_orientation + 3, my_phase_id) * x0 +
                                  grain_unit_vector(9 * my_orientation + 4, my_phase_id) * y0 +
                                  grain_unit_vector(9 * my_orientation + 5, my_phase_id) * z0) > 0);
        const int angle_3_pos = ((grain_unit_vector(9 * my_orientation + 6, my_phase_id) * x0 +
                                  grain_unit_vector(9 * my_orientation + 7, my_phase_id) * y0 +
                                  grain_unit_vector(9 * my_orientation + 8, my_phase_id) * z0) > 0);
        const float diag_1x = grain_unit_vector(9 * my_orientation, my_phase_id) * (2 * angle_1_pos - 1);
        const float diag_1y = grain_unit_vector(9 * my_orientation + 1, my_phase_id) * (2 * angle_1_pos - 1);
        const float diag_1z = grain_unit_vector(9 * my_orientation + 2, my_phase_id) * (2 * angle_1_pos - 1);

        const float diag_2x = grain_unit_vector(9 * my_orientation + 3, my_phase_id) * (2 * angle_2_pos - 1);
        const float diag_2y = grain_unit_vector(9 * my_orientation + 4, my_phase_id) * (2 * angle_2_pos - 1);
        const float diag_2z = grain_unit_vector(9 * my_orientation + 5, my_phase_id) * (2 * angle_2_pos - 1);

        const float diag_3x = grain_unit_vector(9 * my_orientation + 6, my_phase_id) * (2 * angle_3_pos - 1);
        const float diag_3y = grain_unit_vector(9 * my_orientation + 7, my_phase_id) * (2 * angle_3_pos - 1);
        const float diag_3z = grain_unit_vector(9 * my_orientation + 8, my_phase_id) * (2 * angle_3_pos - 1);

        // The capturing face of the octahedron is a triangle, with 3 (x,y,z) coordinates
        // representing the vertices. These vertices are located a distance equivalent to the
        // critical diagonal length for cell capture from the old octahedron center along the
        // unit vector directions
        float triangle_x[3], triangle_y[3], triangle_z[3];
        const float crit_diagonal_length_capture = crit_diagonal_length(26 * index + l);

        triangle_x[0] = cxold + crit_diagonal_length_capture * diag_1x;
        triangle_y[0] = cyold + crit_diagonal_length_capture * diag_1y;
        triangle_z[0] = czold + crit_diagonal_length_capture * diag_1z;

        triangle_x[1] = cxold + crit_diagonal_length_capture * diag_2x;
        triangle_y[1] = cyold + crit_diagonal_length_capture * diag_2y;
        triangle_z[1] = czold + crit_diagonal_length_capture * diag_2z;

        triangle_x[2] = cxold + crit_diagonal_length_capture * diag_3x;
        triangle_y[2] = cyold + crit_diagonal_length_capture * diag_3y;
        triangle_z[2] = czold + crit_diagonal_length_capture * diag_3z;
        // Determine which of the 3 corners of the capturing face is closest to the captured
        // cell center
        float dist_to_corner[3];
        dist_to_corner[0] = Kokkos::hypot(triangle_x[0] - xp, triangle_y[0] - yp, triangle_z[0] - zp);
        dist_to_corner[1] = Kokkos::hypot(triangle_x[1] - xp, triangle_y[1] - yp, triangle_z[1] - zp);
        dist_to_corner[2] = Kokkos::hypot(triangle_x[2] - xp, triangle_y[2] - yp, triangle_z[2] - zp);

        const int corner_0_closer_1 = (dist_to_corner[0] < dist_to_corner[1]);
        const int corner_1_closer_2 = (dist_to_corner[1] < dist_to_corner[2]);
        const int corner_2_closer_0 = (dist_to_corner[2] < dist_to_corner[0]);

        const int triangle_index = 2 * (corner_2_closer_0 - corner_1_closer_2) * corner_2_closer_0 +
                                   (corner_1_closer_2 - corner_0_closer_1) * corner_1_closer_2;
        const float mindist_to_corner = dist_to_corner[triangle_index];
        const float xc = triangle_x[triangle_index];
        const float yc = triangle_y[triangle_index];
        const float zc = triangle_z[triangle_index];

        const float x1 = triangle_x[(triangle_index + 1) % 3];
        const float y1 = triangle_y[(triangle_index + 1) % 3];
        const float z1 = triangle_z[(triangle_index + 1) % 3];
        const float x2 = triangle_x[(triangle_index + 2) % 3];
        const float y2 = triangle_y[(triangle_index + 2) % 3];
        const float z2 = triangle_z[(triangle_index + 2) % 3];

        // Distance between the nearest corner of the capturing face (xc,yc,zc) and the other
        // two corners (should theoretically be the same, but may be slightly different due to
        // floating point errors) Previously d4
        const float dist_first_corner = Kokkos::hypot(xc - x1, yc - y1, zc - z1);
        // Previously d2
        const float dist_second_corner = Kokkos::hypot(xc - x2, yc - y2, zc - z2);

        // Projecting the captured cell center (xp,yp,zp) onto the nearest two edges of the
        // triangular octahedron face (connects the closest corner xc,yc,zc to the corners
        // x1,y1,z1 and x2,y2,z2), what are the distances from this projected point to the two
        // nearest corners for each edge? Previously j_1
        float proj_nearest_corner_edge_1 = 0;
        // Previously j_2
        float proj_next_nearest_corner_edge_1 = dist_first_corner;
        // Previously i_1
        float proj_next_nearest_corner_edge_2 = 0;
        // Previously i_2
        float proj_nearest_corner_edge_2 = dist_second_corner;

        // If minimum distance to corner = 0, the octahedron corner captured the new cell
        // center
        if (mindist_to_corner != 0) {
            proj_nearest_corner_edge_1 =
                ((xp - x1) * (xc - x1) + (yp - y1) * (yc - y1) + (zp - z1) * (zc - z1)) / dist_first_corner;
            proj_next_nearest_corner_edge_1 = dist_first_corner - proj_nearest_corner_edge_1;
            proj_nearest_corner_edge_2 =
                ((xp - x2) * (xc - x2) + (yp - y2) * (yc - y2) + (zp - z2) * (zc - z2)) / dist_second_corner;
            proj_next_nearest_corner_edge_2 = dist_second_corner - proj_nearest_corner_edge_2;
        }

        // Truncate the lengths at sqrt(3) for a max initial size of an octahedron
        const float l_12 = 0.5 * (Kokkos::fmin(proj_nearest_corner_edge_1, Kokkos::sqrt(3.0f)) +
                                  Kokkos::fmin(proj_next_nearest_corner_edge_1, Kokkos::sqrt(3.0f)));
        const float l_13 = 0.5 * (Kokkos::fmin(proj_nearest_corner_edge_2, Kokkos::sqrt(3.0f)) +
                                  Kokkos::fmin(proj_next_nearest_corner_edge_2, Kokkos::sqrt(3.0f)));
        // half diagonal length of new octahedron
        const float new_octahedron_diag_length = Kokkos::sqrt(2.0f) * Kokkos::fmax(l_12, l_13);

        diagonal_length(neighbor_index) = new_octahedron_diag_length;
        // Calculate coordinates of new decentered octahedron center
        const float capt_diag_x = xc - cxold;
        const float capt_diag_y = yc - cyold;
        const float capt_diag_z = zc - czold;
        const float capt_diag_magnitude = Kokkos::hypot(capt_diag_x, capt_diag_y, capt_diag_z);
        const float capt_diag_unit_vec_x = capt_diag_x / capt_diag_magnitude;
        const float capt_diag_unit_vec_y = capt_diag_y / capt_diag_magnitude;
        const float capt_diag_unit_vec_z = capt_diag_z / capt_diag_magnitude;
        // (cx, cy, cz) are the coordinates of the new active cell's decentered octahedron
        const float cx = xc - new_octahedron_diag_length * capt_diag_unit_vec_x;
        const float cy = yc - new_octahedron_diag_length * capt_diag_unit_vec_y;
        const float cz = zc - new_octahedron_diag_length * capt_diag_unit_vec_z;

        octahedron_center(3 * neighbor_index) = cx;
        octahedron_center(3 * neighbor_index + 1) = cy;
        octahedron_center(3 * neighbor_index + 2) = cz;

        octahedron_data[0] = cx;
        octahedron_data[1] = cy;
        octahedron_data[2] = cz;
        octahedron_data[3] = new_octahedron_diag_length;
    }

    // For the newly active cell located at 1D array position index (3D center coordinate of xp, yp, zp),
    // update crit_diagonal_length values for cell capture of neighboring cells. The octahedron has a center located at
    // (cx, cy, cz) Note that yp and cy are relative to the domain origin to keep the coordinate system continuous
    // across ranks
    template <typename ViewType>
    KOKKOS_INLINE_FUNCTION void calcCritDiagonalLength(const int index, const float xp, const float yp, const float zp,
                                                       const float cx, const float cy, const float cz,
                                                       const int my_orientation, const ViewType grain_unit_vector,
                                                       const int my_phase_id = 0) const {
        // Calculate critical octahedron diagonal length to activate nearest neighbor.
        // First, calculate the unique planes (4) associated with all octahedron faces (8)
        // Then just look at distance between face and the point of interest (cell center of
        // neighbor). The critical diagonal length will be the maximum of these (since all other
        // planes will have passed over the point by then
        // ... meaning it must be in the octahedron)
        float fx[4], fy[4], fz[4];

        fx[0] = grain_unit_vector(9 * my_orientation, my_phase_id) +
                grain_unit_vector(9 * my_orientation + 3, my_phase_id) +
                grain_unit_vector(9 * my_orientation + 6, my_phase_id);
        fx[1] = grain_unit_vector(9 * my_orientation, my_phase_id) -
                grain_unit_vector(9 * my_orientation + 3, my_phase_id) +
                grain_unit_vector(9 * my_orientation + 6, my_phase_id);
        fx[2] = grain_unit_vector(9 * my_orientation, my_phase_id) +
                grain_unit_vector(9 * my_orientation + 3, my_phase_id) -
                grain_unit_vector(9 * my_orientation + 6, my_phase_id);
        fx[3] = grain_unit_vector(9 * my_orientation, my_phase_id) -
                grain_unit_vector(9 * my_orientation + 3, my_phase_id) -
                grain_unit_vector(9 * my_orientation + 6, my_phase_id);

        fy[0] = grain_unit_vector(9 * my_orientation + 1, my_phase_id) +
                grain_unit_vector(9 * my_orientation + 4, my_phase_id) +
                grain_unit_vector(9 * my_orientation + 7, my_phase_id);
        fy[1] = grain_unit_vector(9 * my_orientation + 1, my_phase_id) -
                grain_unit_vector(9 * my_orientation + 4, my_phase_id) +
                grain_unit_vector(9 * my_orientation + 7, my_phase_id);
        fy[2] = grain_unit_vector(9 * my_orientation + 1, my_phase_id) +
                grain_unit_vector(9 * my_orientation + 4, my_phase_id) -
                grain_unit_vector(9 * my_orientation + 7, my_phase_id);
        fy[3] = grain_unit_vector(9 * my_orientation + 1, my_phase_id) -
                grain_unit_vector(9 * my_orientation + 4, my_phase_id) -
                grain_unit_vector(9 * my_orientation + 7, my_phase_id);

        fz[0] = grain_unit_vector(9 * my_orientation + 2, my_phase_id) +
                grain_unit_vector(9 * my_orientation + 5, my_phase_id) +
                grain_unit_vector(9 * my_orientation + 8, my_phase_id);
        fz[1] = grain_unit_vector(9 * my_orientation + 2, my_phase_id) -
                grain_unit_vector(9 * my_orientation + 5, my_phase_id) +
                grain_unit_vector(9 * my_orientation + 8, my_phase_id);
        fz[2] = grain_unit_vector(9 * my_orientation + 2, my_phase_id) +
                grain_unit_vector(9 * my_orientation + 5, my_phase_id) -
                grain_unit_vector(9 * my_orientation + 8, my_phase_id);
        fz[3] = grain_unit_vector(9 * my_orientation + 2, my_phase_id) -
                grain_unit_vector(9 * my_orientation + 5, my_phase_id) -
                grain_unit_vector(9 * my_orientation + 8, my_phase_id);

        for (int n = 0; n < 26; n++) {
            float x0 = xp + neighbor_x[n] - cx;
            float y0 = yp + neighbor_y[n] - cy;
            float z0 = zp + neighbor_z[n] - cz;
            float d0 = x0 * fx[0] + y0 * fy[0] + z0 * fz[0];
            float d1 = x0 * fx[1] + y0 * fy[1] + z0 * fz[1];
            float d2 = x0 * fx[2] + y0 * fy[2] + z0 * fz[2];
            float d3 = x0 * fx[3] + y0 * fy[3] + z0 * fz[3];
            float dfabs = fmax(fmax(fabs(d0), fabs(d1)), fmax(fabs(d2), fabs(d3)));
            crit_diagonal_length(26 * index + n) = dfabs;
        }
    }

    // Load data (grain_id, octahedron_center, diagonal_length) into ghost nodes if the given coord_y is associated with
    // a 1D halo region Uses check to ensure that the buffer position does not reach the buffer size - if it does, keep
    // incrementing the send size counters for use resizing the buffers in the future
    KOKKOS_INLINE_FUNCTION
    int loadGhostNodesSuccess(const bool mpi_parallel, const int load_success_type, const int load_failure_type,
                              const int ghost_grain_id, const float (&octahedron_data)[4], const int ghost_phase_id,
                              const int ny_local, const int coord_x, const int coord_y, const int coord_z,
                              const bool at_north_boundary, const bool at_south_boundary,
                              const int n_grain_orientations) const {
        // No halo load if not running on multiple MPI ranks
        if (!mpi_parallel)
            return load_success_type;
        // Otherwise, attempt to load the octahedron data into the appropriate buffer
        if ((coord_y == 1) && (!(at_south_boundary))) {
            int ghost_position_south = Kokkos::atomic_fetch_add(&send_size_south(0), 1);
            if (ghost_position_south >= buf_size)
                return load_failure_type;
            else {
                buffer_south_send(ghost_position_south, 0) = static_cast<float>(coord_x);
                buffer_south_send(ghost_position_south, 1) = static_cast<float>(coord_z);
                buffer_south_send(ghost_position_south, 2) =
                    static_cast<float>(getGrainOrientation(ghost_grain_id, n_grain_orientations, false));
                buffer_south_send(ghost_position_south, 3) =
                    static_cast<float>(getGrainNumber(ghost_grain_id, n_grain_orientations));
                buffer_south_send(ghost_position_south, 4) = octahedron_data[0];
                buffer_south_send(ghost_position_south, 5) = octahedron_data[1];
                buffer_south_send(ghost_position_south, 6) = octahedron_data[2];
                buffer_south_send(ghost_position_south, 7) = octahedron_data[3];
                buffer_south_send(ghost_position_south, 8) = ghost_phase_id;
            }
        }
        else if ((coord_y == ny_local - 2) && (!(at_north_boundary))) {
            int ghost_position_north = Kokkos::atomic_fetch_add(&send_size_north(0), 1);
            if (ghost_position_north >= buf_size)
                return load_failure_type;
            else {
                buffer_north_send(ghost_position_north, 0) = static_cast<float>(coord_x);
                buffer_north_send(ghost_position_north, 1) = static_cast<float>(coord_z);
                buffer_north_send(ghost_position_north, 2) =
                    static_cast<float>(getGrainOrientation(ghost_grain_id, n_grain_orientations, false));
                buffer_north_send(ghost_position_north, 3) =
                    static_cast<float>(getGrainNumber(ghost_grain_id, n_grain_orientations));
                buffer_north_send(ghost_position_north, 4) = octahedron_data[0];
                buffer_north_send(ghost_position_north, 5) = octahedron_data[1];
                buffer_north_send(ghost_position_north, 6) = octahedron_data[2];
                buffer_north_send(ghost_position_north, 7) = octahedron_data[3];
                buffer_north_send(ghost_position_north, 8) = ghost_phase_id;
            }
        }
        return load_success_type;
    }

    // If periodic boundary conditions are used and the recieve buffer is being unpacked, ensure the Y coordinate of the
    // unpacked cell accounts for the periodicity if at global domain edges
    KOKKOS_FUNCTION
    float getAdjustedOctahedronCenterBufferY(const int id, const int np, const int coord_y, const int ny_local,
                                             const float original_octahedron_center_y) const {
        float new_octahedron_center_y;
        if (!is_periodic)
            new_octahedron_center_y = original_octahedron_center_y;
        else {
            if ((coord_y == 0) && (id == 0))
                new_octahedron_center_y = original_octahedron_center_y + octahedron_center_offset_y[1];
            else if ((coord_y == ny_local - 1) && (id == np - 1))
                new_octahedron_center_y = original_octahedron_center_y + octahedron_center_offset_y[0];
            else
                new_octahedron_center_y = original_octahedron_center_y;
        }
        return new_octahedron_center_y;
    }

    // If data doesn't fit in the buffer after the resize, warn that buffer data may have been lost
    KOKKOS_INLINE_FUNCTION
    void checkBufferSize() const {
#if KOKKOS_VERSION >= 40200
        Kokkos::printf("Error: Send/recv buffer resize failed to include all necessary data, predicted "
                       "results at MPI processor boundaries may be inaccurate\n");
#endif
    }
};

#endif
