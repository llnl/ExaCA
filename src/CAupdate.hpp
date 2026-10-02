// Copyright Lawrence Livermore National Security, LLC and other ExaCA Project Developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#ifndef EXACA_UPDATE_HPP
#define EXACA_UPDATE_HPP

#include "CAcelldata.hpp"
#include "CAgrid.hpp"
#include "CAinputs.hpp"
#include "CAinterface.hpp"
#include "CAinterfacialresponse.hpp"
#include "CAorientation.hpp"
#include "CAprint.hpp"
#include "CAtemperature.hpp"

#include <Kokkos_Core.hpp>

#include <string>

// For problems where the entire domain starts as liquid/active cells and then solidifies, initialize octahedra for
// initial active cells
template <typename MemorySpace>
void createOctahedra_NoRemelt(const Grid &grid, CellData<MemorySpace> &celldata, Temperature<MemorySpace> &temperature,
                              Orientation<MemorySpace> &orientation, Interface<MemorySpace> &interface) {

    auto grain_id = celldata.getGrainIDSubview(grid);

    Kokkos::parallel_for(
        "InitSV", grid.domain_size, KOKKOS_LAMBDA(const int &index) {
            if (celldata.cell_type(index) == Active) {
                // Create octahedra for cells initially designated as active (
                int cell_location[3];
                grid.getCoordXYZ(cell_location, index);

                const int my_grain_id = grain_id(index);
                // The orientation for the new grain will depend on its Grain ID
                const int my_orientation = getGrainOrientation(my_grain_id, orientation.n_grain_orientations);

                temperature.setStartingUndercooling(0, index);

                // Initialize new octahedron
                interface.createNewOctahedron(index, cell_location, grid.y_offset);

                // Octahedron center is at (cx, cy, cz) - note that the Y coordinate is relative to the domain
                // origin to keep the coordinate system continuous across ranks
                const float cx = cell_location[0] + 0.5;
                const float cy = cell_location[1] + grid.y_offset + 0.5;
                const float cz = cell_location[2] + 0.5;

                interface.calcCritDiagonalLength(index, cx, cy, cz, cx, cy, cz, my_orientation,
                                                 orientation.grain_unit_vector);
            }
        });
}

// For the case where cells may melt and solidify multiple times, determine which cells are associated with the
// "steering vector" of cells that are either active, or becoming active this time step, or undergoing melting
template <typename MemorySpace>
void remeltActivateCells(const int cycle, const Grid &grid, const InterfacialResponseFunction &irf,
                         CellData<MemorySpace> &celldata, Temperature<MemorySpace> &temperature,
                         Interface<MemorySpace> &interface) {

    auto grain_id = celldata.getGrainIDSubview(grid);
    auto phase_id = celldata.getPhaseIDSubview(grid);
    // Do any cells go above/below the liquidus time on this time step?
    if (temperature.liquidus_time_counter < temperature.num_liquidus_times_this_layer) {
        if (cycle == temperature.liquidus_time_list_host(temperature.liquidus_time_counter)) {
            bool melt_sol_check = true;
            // At least one cell goes above/below the liquidus this time step
            const int first_event = temperature.liquidus_time_counter;
            // Are there any other events this time step to check?
            while (melt_sol_check) {
                temperature.liquidus_time_counter++;
                // If the previous nucleation event was the last one for this layer of the simulation, exit loop
                if (temperature.liquidus_time_counter == temperature.num_liquidus_times_this_layer)
                    break;
                // If the next nucleation event corresponds to a future time step, finish check
                if (cycle != temperature.liquidus_time_list_host(temperature.liquidus_time_counter))
                    melt_sol_check = false;
            }
            const int last_event = temperature.liquidus_time_counter;
            // Loop over each event, perform associated cell type transitions and steering vector additions
            auto policy = Kokkos::RangePolicy<>(first_event, last_event);
            Kokkos::parallel_for(
                "LiquidusTimeCheck", policy, KOKKOS_LAMBDA(const int liquidus_time_counter) {
                    const int index = temperature.liquidus_cell_list(liquidus_time_counter);
                    const int celltype = celldata.cell_type(index);
                    const bool cooling_yn = (temperature.cooling_rate_list(liquidus_time_counter) >= 0);
                    if (cooling_yn) {
                        temperature.last_time_below_liquidus(index) = cycle;
                        temperature.current_cooling_rate(index) = temperature.cooling_rate_list(liquidus_time_counter);
                        if ((celltype == Liquid) && (grain_id(index) != 0)) {
                            // Get the x, y, z coordinates of the cell on this MPI rank
                            int cell_location[3];
                            grid.getCoordXYZ(cell_location, index);
                            // If this cell has cooled to the liquidus temperature, borders at least one solid
                            // cell, and is part of a grain, it should become active. This only needs to be checked on
                            // the time step where the cell reaches the liquidus, not every time step beyond this
                            for (int l = 0; l < 26; l++) {
                                // "l" correpsponds to the specific neighboring cell
                                // Local coordinates of adjacent cell center
                                int neighbor_coord_x = cell_location[0] + interface.neighbor_x[l];
                                int neighbor_coord_y = cell_location[1] + interface.neighbor_y[l];
                                int neighbor_coord_z = cell_location[2] + interface.neighbor_z[l];
                                const int neighbor_index =
                                    grid.getNeighbor1DIndex(neighbor_coord_x, neighbor_coord_y, neighbor_coord_z);
                                if (neighbor_index != -1) {
                                    // TODO: Should check if at global domain edge in X or Y too, since this would also
                                    // make the cell a candidate for activation. This check could also be done outside
                                    // of the loop over neighbors l=0:25 since coord_z does not vary inside this loop
                                    if ((celldata.cell_type(neighbor_index) == Solid) || (cell_location[2] == 0)) {
                                        // Cell activation to be performed as part of steering vector
                                        l = 26;
                                        interface.steering_vector(
                                            Kokkos::atomic_fetch_add(&interface.num_steer(0), 1)) = index;
                                        celldata.cell_type(index) = FutureActive;
                                        // Single phase alloy or second phase transformation after the first
                                        // solidification event, set phase to primary. Otherwise, set phase to secondary
                                        const int phase_id_old = phase_id(index);
                                        phase_id(index) = irf.getPreferredPhaseActivation(phase_id_old);
                                        // This cell was at the edge of the temperature field - set indicator to true if
                                        // this is being tracked
                                        celldata.setMeltEdge(index, true);
                                    }
                                }
                            }
                        }
                    }
                    else {
                        // Cell melts, undercooling is reset to 0 from the previous value, if any
                        celldata.cell_type(index) = Liquid;
                        // Reset this to max value as this cell is no longer below the liquidus
                        temperature.last_time_below_liquidus(index) = std::numeric_limits<int>::max();
                        // Any adjacent active cells should also be remelted, as these cells are more likely heating up
                        // than cooling down These are converted to the temporary FutureLiquid state, to be later
                        // iterated over and loaded into the steering vector as necessary Get the x, y, z coordinates of
                        // the cell on this MPI rank
                        int cell_location[3];
                        grid.getCoordXYZ(cell_location, index);
                        for (int l = 0; l < 26; l++) {
                            // "l" correpsponds to the specific neighboring cell
                            // Local coordinates of adjacent cell center
                            int neighbor_coord_x = cell_location[0] + interface.neighbor_x[l];
                            int neighbor_coord_y = cell_location[1] + interface.neighbor_y[l];
                            int neighbor_coord_z = cell_location[2] + interface.neighbor_z[l];
                            const int neighbor_index =
                                grid.getNeighbor1DIndex(neighbor_coord_x, neighbor_coord_y, neighbor_coord_z);
                            if (neighbor_index != -1) {
                                if (celldata.cell_type(neighbor_index) == Active) {
                                    // Mark adjacent active cells to this as cells that should be converted into liquid,
                                    // as they are more likely heating than cooling
                                    celldata.cell_type(neighbor_index) = FutureLiquid;
                                    interface.steering_vector(Kokkos::atomic_fetch_add(&interface.num_steer(0), 1)) =
                                        neighbor_index;
                                }
                            }
                        }
                    }
                });
        }
    }
    Kokkos::fence();
}

// Determine which cells are associated with the "steering vector" of cells that are active and below the liquidus
// temperature on this time step
template <typename MemorySpace>
void fillSteeringVector(const int cycle, const Grid &grid, CellData<MemorySpace> &celldata,
                        Temperature<MemorySpace> &temperature, Interface<MemorySpace> &interface) {
    Kokkos::parallel_for(
        "FillSV", grid.domain_size, KOKKOS_LAMBDA(const int &index) {
            if ((celldata.cell_type(index) == Active) && (cycle > temperature.last_time_below_liquidus(index))) {
                // Add active cells below liquidus to steering vector
                interface.steering_vector(Kokkos::atomic_fetch_add(&interface.num_steer(0), 1)) = index;
            }
        });
    Kokkos::fence();

    // Copy size of steering vector (containing positions of undercooled active cells added in this routine, as well as
    // FutureLiquid and FutureActive cells added in remeltActivateCells) to the host
    Kokkos::deep_copy(interface.num_steer_host, interface.num_steer);
}

// Decentered octahedron algorithm for the capture of new interface cells by grains
template <typename MemorySpace>
void cellCapture(const int cycle, const bool mpi_parallel, const Grid &grid, const InterfacialResponseFunction &irf,
                 CellData<MemorySpace> &celldata, Temperature<MemorySpace> &temperature,
                 Interface<MemorySpace> &interface, Orientation<MemorySpace> &orientation) {

    // Get grain_id subview for this layer
    auto grain_id = celldata.getGrainIDSubview(grid);
    auto phase_id = celldata.getPhaseIDSubview(grid);
    // Loop over list of active and soon-to-be active cells, potentially performing cell capture events and updating
    // cell types
    Kokkos::parallel_for(
        "CellCapture", interface.num_steer_host(0), KOKKOS_LAMBDA(const int &num) {
            // Reset steering vector size on device to 0, to be rebuilt next time step
            interface.num_steer(0) = 0;
            // Get the 1D index of cell from the steering vector
            const int index = interface.steering_vector(num);
            // Using the 1D index, get the x, y, z coordinates of the cell on this MPI rank
            int cell_location[3];
            grid.getCoordXYZ(cell_location, index);
            const int cell_type_old = celldata.cell_type(index);
            // Cells of interest for the CA - active cells and future active/liquid cells
            if (cell_type_old == Active) {
                // Get undercooling of active cell
                const float local_undercooling = temperature.getUndercooling(cycle, index);
                // Update diagonal length of octahedron based on local undercooling and interfacial response
                // function
                interface.diagonal_length(index) += irf.compute(local_undercooling, phase_id(index));
                const float diagonal_length_cell = interface.diagonal_length(index);
                // Switch that becomes false if the cell has at least 1 liquid type neighbor
                bool deactivate_cell = true;
                // Cycle through all neighboring cells on this processor to see if they have been captured
                for (int l = 0; l < 26; l++) {
                    // Local coordinates of adjacent cell center
                    const int neighbor_coord_x = cell_location[0] + interface.neighbor_x[l];
                    const int neighbor_coord_y = cell_location[1] + interface.neighbor_y[l];
                    const int neighbor_coord_z = cell_location[2] + interface.neighbor_z[l];
                    // Check if neighbor is in bounds
                    const int neighbor_index =
                        grid.getNeighbor1DIndex(neighbor_coord_x, neighbor_coord_y, neighbor_coord_z);
                    if (neighbor_index != -1) {
                        const int neighbor_cell_type = celldata.cell_type(neighbor_index);
                        if (neighbor_cell_type == Liquid)
                            deactivate_cell = false;
                        // Capture of cell located at "neighbor_index" if this condition is satisfied
                        if ((diagonal_length_cell >= interface.crit_diagonal_length(26 * index + l)) &&
                            (neighbor_cell_type == Liquid)) {
                            // Use of atomic_compare_exchange
                            // (https://github.com/kokkos/kokkos/wiki/Kokkos%3A%3Aatomic_compare_exchange) old_val =
                            // atomic_compare_exchange(ptr_to_value,comparison_value, new_value); Atomically sets the
                            // value at the address given by ptr_to_value to new_value if the current value at
                            // ptr_to_value is equal to comparison_value Returns the previously stored value at the
                            // address independent on whether the exchange has happened. If this cell's is a liquid
                            // cell, change it to "TemporaryUpdate" type and return a value of "liquid" If this cell has
                            // already been changed to "TemporaryUpdate" type, return a value of "0"
                            int old_cell_type_value = Kokkos::atomic_compare_exchange(
                                &celldata.cell_type(neighbor_index), Liquid, TemporaryUpdate);
                            // Only proceed if cell_type was previously liquid (this current thread changed the value to
                            // TemporaryUpdate)
                            if (old_cell_type_value == Liquid) {
                                // Cell capture event
                                const int my_grain_id = grain_id(index);
                                const int my_orientation =
                                    getGrainOrientation(my_grain_id, orientation.n_grain_orientations);

                                // This cell was not at the edge of the temperature field - set indicator to false if
                                // this is being tracked
                                celldata.setMeltEdge(neighbor_index, false);

                                // The new cell is captured by this cell's growing octahedron
                                grain_id(neighbor_index) = my_grain_id;
                                const int my_phase_id = phase_id(index);
                                phase_id(neighbor_index) = my_phase_id;
                                // Store the initial undercooling for the newly captured cell, if this output was
                                // toggled
                                temperature.setStartingUndercooling(cycle, neighbor_index);

                                // (xp,yp,zp) are the global coordinates of the new cell's center
                                // Note that the Y coordinate is relative to the domain origin to keep the coordinate
                                // system continuous across ranks
                                const float xp = neighbor_coord_x + 0.5;
                                const float yp = neighbor_coord_y + grid.y_offset + 0.5;
                                const float zp = neighbor_coord_z + 0.5;

                                // Get new octahedron position and size
                                float octahedron_data[4];
                                interface.createNewOctahedron(octahedron_data, l, index, xp, yp, zp,
                                                              orientation.grain_unit_vector, my_orientation,
                                                              my_phase_id, neighbor_index);
                                // Get new critical diagonal length values for the newly activated cell (at array
                                // position "neighbor_index")
                                interface.calcCritDiagonalLength(neighbor_index, xp, yp, zp, octahedron_data[0],
                                                                 octahedron_data[1], octahedron_data[2], my_orientation,
                                                                 orientation.grain_unit_vector, my_phase_id);

                                // Collect data for the ghost nodes, if necessary
                                // Data loaded into the ghost nodes is for the cell that was just captured
                                // Note: this only updates the cell type once all octahedron attributes have been
                                // calculated to avoid any potential race condition with operating on the active cell
                                // before it has been fully initialized
                                celldata.cell_type(neighbor_index) = interface.loadGhostNodesSuccess(
                                    mpi_parallel, Active, ActiveFailedBufferLoad, my_grain_id, octahedron_data,
                                    my_phase_id, grid.ny_local, neighbor_coord_x, neighbor_coord_y, neighbor_coord_z,
                                    grid.at_north_boundary, grid.at_south_boundary, orientation.n_grain_orientations);
                            } // End if statement within locked capture loop
                        } // End if statement for outer capture loop
                    } // End if statement over neighbors on the active grid
                } // End loop over all neighbors of this active cell
                if (deactivate_cell) {
                    // This active cell has no more neighboring cells to be captured, is now solid
                    celldata.cell_type(index) = Solid;
                    // Set undercooling of cell at solidification completion/update counter if optional inputs were
                    // toggled
                    temperature.setEndingUndercooling(cycle, index);
                    temperature.updateSolidificationCounter(index);
                }
            }
            else if (cell_type_old == FutureActive) {
                // Successful nucleation event - this cell is becoming a new active cell
                celldata.cell_type(index) = TemporaryUpdate; // avoid operating on the new active cell before its
                                                             // associated octahedron data is initialized
                const int my_grain_id = grain_id(index); // grain_id was assigned as part of Nucleation
                const int my_phase_id = phase_id(index); // phase_id was assigned as part of Nucleation

                temperature.setStartingUndercooling(cycle, index);

                // Initialize new octahedron
                interface.createNewOctahedron(index, cell_location, grid.y_offset);
                // The orientation for the new grain will depend on its Grain ID (nucleated grains have negative
                // grain_id values)
                const int my_orientation = getGrainOrientation(my_grain_id, orientation.n_grain_orientations);
                // Octahedron center is at (cx, cy, cz) - note that the Y coordinate is relative to the domain
                // origin to keep the coordinate system continuous across ranks
                const float cx = cell_location[0] + 0.5;
                const float cy = cell_location[1] + grid.y_offset + 0.5;
                const float cz = cell_location[2] + 0.5;
                float octahedron_data[4] = {cx, cy, cz, interface._init_oct_size};
                // Calculate critical values at which this active cell leads to the activation of a neighboring
                // liquid cell. Octahedron center and cell center overlap for octahedra created as part of a new
                // grain
                interface.calcCritDiagonalLength(index, cx, cy, cz, cx, cy, cz, my_orientation,
                                                 orientation.grain_unit_vector, my_phase_id);
                // Collect data for the ghost nodes, if necessary
                // Data loaded into the ghost nodes is for the cell that was just activated or underwent nucleation
                // Note: this only updates the cell type once all octahedron attributes have been calculated to avoid
                // any potential race condition with operating on the active cell before it has been fully initialized
                celldata.cell_type(index) = interface.loadGhostNodesSuccess(
                    mpi_parallel, Active, ActiveFailedBufferLoad, my_grain_id, octahedron_data, my_phase_id,
                    grid.ny_local, cell_location[0], cell_location[1], cell_location[2], grid.at_north_boundary,
                    grid.at_south_boundary, orientation.n_grain_orientations);
            }
            else if (cell_type_old == FutureLiquid) {
                // This type was assigned to a cell that was recently transformed from active to liquid, due to its
                // bordering of a cell above the liquidus. This information may need to be sent to other MPI ranks
                // Dummy values for Grain ID, Phase ID, and octahedron data
                float octahedron_data[4] = {0.0};
                celldata.cell_type(index) = interface.loadGhostNodesSuccess(
                    mpi_parallel, Liquid, LiquidFailedBufferLoad, 0, octahedron_data, 0, grid.ny_local,
                    cell_location[0], cell_location[1], cell_location[2], grid.at_north_boundary,
                    grid.at_south_boundary, orientation.n_grain_orientations);
            }
        });
    Kokkos::fence();
}

// Check buffers for overflow and resize/refill as necessary
template <typename MemorySpace>
void checkBuffers(const int id, const int cycle, const Grid &grid, CellData<MemorySpace> &celldata,
                  Interface<MemorySpace> &interface, const int n_grain_orientations) {
    // Count the number of cells' in halo regions where the data did not fit into the send buffers
    // Reduce across all ranks, as the same buf_size should be maintained across all ranks
    // If any rank overflowed its buffer size, resize all buffers to the new size plus a padding (default val of 25
    // cells)
    bool resize_performed = interface.resizeBuffers(id, cycle);
    if (resize_performed)
        refillBuffers(grid, celldata, interface, n_grain_orientations);
}

// Refill the buffers as necessary starting from the old count size, using the data from cells marked with type
// ActiveFailedBufferLoad
template <typename MemorySpace>
void refillBuffers(const Grid &grid, CellData<MemorySpace> &celldata, Interface<MemorySpace> &interface,
                   const int n_grain_orientations) {

    auto grain_id = celldata.getGrainIDSubview(grid);
    auto phase_id = celldata.getPhaseIDSubview(grid);
    Kokkos::parallel_for(
        "FillSendBuffersOverflow", grid.nx, KOKKOS_LAMBDA(const int &coord_x) {
            for (int coord_z = 0; coord_z < grid.nz_layer; coord_z++) {
                int index_south_buffer = grid.get1DIndex(coord_x, 1, coord_z);
                int index_north_buffer = grid.get1DIndex(coord_x, grid.ny_local - 2, coord_z);
                if (celldata.cell_type(index_south_buffer) == ActiveFailedBufferLoad) {
                    int ghost_grain_id = grain_id(index_south_buffer);
                    float octahedron_data[4] = {interface.octahedron_center(3 * index_south_buffer),
                                                interface.octahedron_center(3 * index_south_buffer + 1),
                                                interface.octahedron_center(3 * index_south_buffer + 2),
                                                interface.diagonal_length(index_south_buffer)};
                    float ghost_phase_id = phase_id(index_south_buffer);
                    // Collect data for the ghost nodes, if necessary
                    // Data loaded into the ghost nodes is for the cell that was just captured
                    celldata.cell_type(index_south_buffer) = interface.loadGhostNodesSuccess(
                        true, Active, ActiveFailedBufferLoad, ghost_grain_id, octahedron_data, ghost_phase_id,
                        grid.ny_local, coord_x, 1, coord_z, grid.at_north_boundary, grid.at_south_boundary,
                        n_grain_orientations);
                    // If data doesn't fit in the buffer after the resize, warn that buffer data may have been lost
                    if (celldata.cell_type(index_south_buffer) == ActiveFailedBufferLoad)
                        interface.checkBufferSize();
                }
                else if (celldata.cell_type(index_south_buffer) == LiquidFailedBufferLoad) {
                    // Dummy values for first 4 arguments (Grain ID and octahedron center coordinates), 0 for
                    // diagonal length
                    float octahedron_data[4] = {0.0};
                    celldata.cell_type(index_south_buffer) = interface.loadGhostNodesSuccess(
                        true, Liquid, LiquidFailedBufferLoad, 0, octahedron_data, 0, grid.ny_local, coord_x, 1, coord_z,
                        grid.at_north_boundary, grid.at_south_boundary, n_grain_orientations);
                    // If data doesn't fit in the buffer after the resize, warn that buffer data may have been lost
                    if (celldata.cell_type(index_south_buffer) == LiquidFailedBufferLoad)
                        interface.checkBufferSize();
                }
                if (celldata.cell_type(index_north_buffer) == ActiveFailedBufferLoad) {
                    int ghost_grain_id = grain_id(index_north_buffer);
                    float octahedron_data[4] = {interface.octahedron_center(3 * index_north_buffer),
                                                interface.octahedron_center(3 * index_north_buffer + 1),
                                                interface.octahedron_center(3 * index_north_buffer + 2),
                                                interface.diagonal_length(index_north_buffer)};
                    float ghost_phase_id = phase_id(index_north_buffer);
                    // Collect data for the ghost nodes, if necessary
                    // Data loaded into the ghost nodes is for the cell that was just captured
                    celldata.cell_type(index_north_buffer) = interface.loadGhostNodesSuccess(
                        true, Active, ActiveFailedBufferLoad, ghost_grain_id, octahedron_data, ghost_phase_id,
                        grid.ny_local, coord_x, grid.ny_local - 2, coord_z, grid.at_north_boundary,
                        grid.at_south_boundary, n_grain_orientations);
                    // If data doesn't fit in the buffer after the resize, warn that buffer data may have been lost
                    if (celldata.cell_type(index_north_buffer) == ActiveFailedBufferLoad)
                        interface.checkBufferSize();
                }
                else if (celldata.cell_type(index_north_buffer) == LiquidFailedBufferLoad) {
                    // Dummy values for first 4 arguments (Grain ID and octahedron center coordinates), 0 for
                    // diagonal length
                    float octahedron_data[4] = {0.0};
                    celldata.cell_type(index_north_buffer) = interface.loadGhostNodesSuccess(
                        true, Liquid, LiquidFailedBufferLoad, 0, octahedron_data, 0, grid.ny_local, coord_x, 1, coord_z,
                        grid.at_north_boundary, grid.at_south_boundary, n_grain_orientations);
                    // If data doesn't fit in the buffer after the resize, warn that buffer data may have been lost
                    if (celldata.cell_type(index_north_buffer) == LiquidFailedBufferLoad)
                        interface.checkBufferSize();
                }
            }
        });
    Kokkos::fence();
}

// 1D domain decomposition: update ghost nodes with new cell data from nucleation.nucleateGrain and cellCapture routines
template <typename MemorySpace>
void haloUpdate(const int, const int id, const int np, const Grid &grid, CellData<MemorySpace> &celldata,
                Interface<MemorySpace> &interface, Orientation<MemorySpace> &orientation) {

    std::vector<MPI_Request> send_requests(2, MPI_REQUEST_NULL);
    std::vector<MPI_Request> recv_requests(2, MPI_REQUEST_NULL);

    // Send data to each other rank (MPI_Isend)
    MPI_Isend(interface.buffer_south_send.data(), interface.buf_components * interface.buf_size, MPI_FLOAT,
              grid.neighbor_rank_south, 0, MPI_COMM_WORLD, &send_requests[0]);
    MPI_Isend(interface.buffer_north_send.data(), interface.buf_components * interface.buf_size, MPI_FLOAT,
              grid.neighbor_rank_north, 1, MPI_COMM_WORLD, &send_requests[1]);

    // Receive buffers for all neighbors (MPI_Irecv)
    MPI_Irecv(interface.buffer_south_recv.data(), interface.buf_components * interface.buf_size, MPI_FLOAT,
              grid.neighbor_rank_south, 1, MPI_COMM_WORLD, &recv_requests[0]);
    MPI_Irecv(interface.buffer_north_recv.data(), interface.buf_components * interface.buf_size, MPI_FLOAT,
              grid.neighbor_rank_north, 0, MPI_COMM_WORLD, &recv_requests[1]);

    // unpack in any order
    bool unpack_complete = false;
    auto grain_id = celldata.getGrainIDSubview(grid);
    auto phase_id = celldata.getPhaseIDSubview(grid);
    while (!unpack_complete) {
        // Get the next buffer to unpack from rank "unpack_index"
        int unpack_index = MPI_UNDEFINED;
        MPI_Waitany(2, recv_requests.data(), &unpack_index, MPI_STATUS_IGNORE);
        // If there are no more buffers to unpack, leave the while loop
        if (MPI_UNDEFINED == unpack_index) {
            unpack_complete = true;
        }
        // Otherwise unpack the next buffer.
        else {
            Kokkos::parallel_for(
                "BufferUnpack", interface.buf_size, KOKKOS_LAMBDA(const int &buf_position) {
                    int coord_x, coord_y, coord_z, index, new_grain_id, new_phase_id;
                    float new_octahedron_center_x, new_octahedron_center_y, new_octahedron_center_z,
                        new_diagonal_length;
                    bool place = false;
                    // Which rank was the data received from? Is there valid data at this position in the buffer
                    // (i.e., not set to -1.0)?
                    if ((unpack_index == 0) && (interface.buffer_south_recv(buf_position, 0) != -1.0) &&
                        (grid.neighbor_rank_south != MPI_PROC_NULL)) {
                        // Data received from South
                        coord_x = static_cast<int>(interface.buffer_south_recv(buf_position, 0));
                        coord_y = 0;
                        coord_z = static_cast<int>(interface.buffer_south_recv(buf_position, 1));
                        index = grid.get1DIndex(coord_x, coord_y, coord_z);
                        // Two possibilities: buffer data with non-zero diagonal length was loaded, and a liquid
                        // cell may have to be updated to active - or zero diagonal length data was loaded, and an
                        // active cell may have to be updated to liquid
                        if ((celldata.cell_type(index) == Liquid) &&
                            (interface.buffer_south_recv(buf_position, 7) > 0.0)) {
                            place = true;
                            int my_grain_orientation = static_cast<int>(interface.buffer_south_recv(buf_position, 2));
                            int my_grain_number = static_cast<int>(interface.buffer_south_recv(buf_position, 3));
                            new_grain_id =
                                getGrainID(my_grain_orientation, my_grain_number, orientation.n_grain_orientations);
                            new_octahedron_center_x = interface.buffer_south_recv(buf_position, 4);
                            // Adjust center in Y for periodic boundary if needed
                            new_octahedron_center_y = interface.getAdjustedOctahedronCenterBufferY(
                                id, np, coord_y, grid.ny_local, interface.buffer_south_recv(buf_position, 5));
                            new_octahedron_center_z = interface.buffer_south_recv(buf_position, 6);
                            new_diagonal_length = interface.buffer_south_recv(buf_position, 7);
                            new_phase_id = interface.buffer_south_recv(buf_position, 8);
                        }
                        else if ((celldata.cell_type(index) == Active) &&
                                 (interface.buffer_south_recv(buf_position, 7) == 0.0)) {
                            celldata.cell_type(index) = Liquid;
                        }
                    }
                    else if ((unpack_index == 1) && (interface.buffer_north_recv(buf_position, 0) != -1.0) &&
                             (grid.neighbor_rank_north != MPI_PROC_NULL)) {
                        // Data received from North
                        coord_x = static_cast<int>(interface.buffer_north_recv(buf_position, 0));
                        coord_y = grid.ny_local - 1;
                        coord_z = static_cast<int>(interface.buffer_north_recv(buf_position, 1));
                        index = grid.get1DIndex(coord_x, coord_y, coord_z);
                        // Two possibilities: buffer data with non-zero diagonal length was loaded, and a liquid
                        // cell may have to be updated to active - or zero diagonal length data was loaded, and an
                        // active cell may have to be updated to liquid
                        if ((celldata.cell_type(index) == Liquid) &&
                            (interface.buffer_north_recv(buf_position, 7) > 0.0)) {
                            place = true;
                            int my_grain_orientation = static_cast<int>(interface.buffer_north_recv(buf_position, 2));
                            int my_grain_number = static_cast<int>(interface.buffer_north_recv(buf_position, 3));
                            new_grain_id =
                                getGrainID(my_grain_orientation, my_grain_number, orientation.n_grain_orientations);
                            new_octahedron_center_x = interface.buffer_north_recv(buf_position, 4);
                            // Adjust center in Y for periodic boundary if needed
                            new_octahedron_center_y = interface.getAdjustedOctahedronCenterBufferY(
                                id, np, coord_y, grid.ny_local, interface.buffer_north_recv(buf_position, 5));
                            new_octahedron_center_z = interface.buffer_north_recv(buf_position, 6);
                            new_diagonal_length = interface.buffer_north_recv(buf_position, 7);
                            new_phase_id = interface.buffer_north_recv(buf_position, 8);
                        }
                        else if ((celldata.cell_type(index) == Active) &&
                                 (interface.buffer_north_recv(buf_position, 7) == 0.0)) {
                            celldata.cell_type(index) = Liquid;
                        }
                    }
                    if (place) {
                        // Update this ghost node cell's information with data from other rank
                        grain_id(index) = new_grain_id;
                        interface.octahedron_center(3 * index) = new_octahedron_center_x;
                        interface.octahedron_center(3 * index + 1) = new_octahedron_center_y;
                        interface.octahedron_center(3 * index + 2) = new_octahedron_center_z;
                        int my_orientation = getGrainOrientation(grain_id(index), orientation.n_grain_orientations);
                        interface.diagonal_length(index) = static_cast<float>(new_diagonal_length);
                        phase_id(index) = new_phase_id;
                        // Cell center - note that the Y coordinate is relative to the domain origin to keep the
                        // coordinate system continuous across ranks
                        float xp = coord_x + 0.5;
                        float yp = coord_y + grid.y_offset + 0.5;
                        float zp = coord_z + 0.5;
                        // Calculate critical values at which this active cell leads to the activation of a
                        // neighboring liquid cell
                        interface.calcCritDiagonalLength(index, xp, yp, zp, new_octahedron_center_x,
                                                         new_octahedron_center_y, new_octahedron_center_z,
                                                         my_orientation, orientation.grain_unit_vector, new_phase_id);
                        celldata.cell_type(index) = Active;
                    }
                });
        }
    }

    // Reset send buffer data to -1 (used as placeholder) and reset the number of cells stored in the buffers to 0
    interface.resetBuffers();
    // Wait on send requests
    MPI_Waitall(2, send_requests.data(), MPI_STATUSES_IGNORE);
    Kokkos::fence();
}

// Update the periodic boundaries at the +/-X edges of the domain and, if not already handled via the domain
// decomposition in Y, update the periodic boundaries at the +/-Y edges of the domain. Only need to update where cell at
// boundary was liquid and is now active (a steering vector as used with the halo regions might make this faster in the
// future)
template <typename MemorySpace>
void updatePeriodicBoundaries(const bool mpi_parallel, const Grid &grid, CellData<MemorySpace> &celldata,
                              Interface<MemorySpace> &interface, Orientation<MemorySpace> &orientation) {
    auto grain_id = celldata.getGrainIDSubview(grid);
    auto phase_id = celldata.getPhaseIDSubview(grid);

    // Update for +/-X excludes corners
    Kokkos::parallel_for(
        "UpdateBoundariesX", (grid.ny_local - 2) * grid.nz_layer, KOKKOS_LAMBDA(const int &boundary_cell_idx) {
            const int coord_y = boundary_cell_idx / grid.nz_layer + 1;
            const int coord_z = boundary_cell_idx % grid.nz_layer;
            for (int bound_num = 0; bound_num < 2; bound_num++) {
                const int index_interior = grid.get1DIndex(interface.coord_x_interior[bound_num], coord_y, coord_z);
                const int index_exterior = grid.get1DIndex(interface.coord_x_exterior[bound_num], coord_y, coord_z);
                if ((celldata.cell_type(index_interior) == Active) && (celldata.cell_type(index_exterior) == Liquid)) {
                    // Copy cell attributes to maintain periodic boundary
                    grain_id(index_exterior) = grain_id(index_interior);
                    phase_id(index_exterior) = phase_id(index_interior);
                    // Adjust octahedron center in X, Y, Z
                    interface.octahedron_center(3 * index_exterior) = interface.octahedron_center(3 * index_interior) +
                                                                      interface.octahedron_center_offset_x[bound_num];
                    interface.octahedron_center(3 * index_exterior + 1) =
                        interface.octahedron_center(3 * index_interior + 1);
                    interface.octahedron_center(3 * index_exterior + 2) =
                        interface.octahedron_center(3 * index_interior + 2);
                    interface.diagonal_length(index_exterior) = interface.diagonal_length(index_interior);
                    const int my_orientation =
                        getGrainOrientation(grain_id(index_exterior), orientation.n_grain_orientations);
                    // (xp,yp,zp) are the global coordinates of the new cell's center
                    // Note that the Y coordinate is relative to the domain origin to keep the coordinate
                    // system continuous across ranks
                    const float xp = interface.coord_x_exterior[bound_num] + 0.5;
                    const float yp = coord_y + grid.y_offset + 0.5;
                    const float zp = coord_z + 0.5;
                    interface.calcCritDiagonalLength(
                        index_exterior, xp, yp, zp, interface.octahedron_center(3 * index_exterior),
                        interface.octahedron_center(3 * index_exterior + 1),
                        interface.octahedron_center(3 * index_exterior + 2), my_orientation,
                        orientation.grain_unit_vector, phase_id(index_exterior));
                    celldata.cell_type(index_exterior) = Active;
                    float octahedron_data[4] = {interface.octahedron_center(3 * index_exterior),
                                                interface.octahedron_center(3 * index_exterior + 1),
                                                interface.octahedron_center(3 * index_exterior + 2),
                                                interface.diagonal_length(index_exterior)};
                    // Load into halo regions if necessary
                    celldata.cell_type(index_exterior) = interface.loadGhostNodesSuccess(
                        mpi_parallel, Active, ActiveFailedBufferLoad, grain_id(index_exterior), octahedron_data,
                        phase_id(index_exterior), grid.ny_local, interface.coord_x_exterior[bound_num], coord_y,
                        coord_z, grid.at_north_boundary, grid.at_south_boundary, orientation.n_grain_orientations);
                }
            }
        });
    // Update for +/-Y and corners - only performed if no MPI (i.e., rank 0 domain wraps around in Y)
    if (!mpi_parallel) {
        Kokkos::parallel_for(
            "UpdateBoundariesY", grid.nx * grid.nz_layer, KOKKOS_LAMBDA(const int &boundary_cell_idx) {
                const int coord_x = boundary_cell_idx / grid.nz_layer;
                const int coord_z = boundary_cell_idx % grid.nz_layer;
                for (int bound_num = 0; bound_num < 2; bound_num++) {
                    const int index_interior = grid.get1DIndex(coord_x, interface.coord_y_interior[bound_num], coord_z);
                    const int index_exterior = grid.get1DIndex(coord_x, interface.coord_y_exterior[bound_num], coord_z);
                    if ((celldata.cell_type(index_interior) == Active) &&
                        (celldata.cell_type(index_exterior) == Liquid)) {
                        // Copy cell attributes to maintain periodic boundary
                        grain_id(index_exterior) = grain_id(index_interior);
                        phase_id(index_exterior) = phase_id(index_interior);
                        interface.octahedron_center(3 * index_exterior) =
                            interface.octahedron_center(3 * index_interior);
                        // Adjust octahedron center in Y
                        interface.octahedron_center(3 * index_exterior + 1) =
                            interface.octahedron_center(3 * index_interior + 1) +
                            interface.octahedron_center_offset_y[bound_num];
                        interface.octahedron_center(3 * index_exterior + 2) =
                            interface.octahedron_center(3 * index_interior + 2);
                        interface.diagonal_length(index_exterior) = interface.diagonal_length(index_interior);
                        const int my_orientation =
                            getGrainOrientation(grain_id(index_exterior), orientation.n_grain_orientations);
                        // (xp,yp,zp) are the global coordinates of the new cell's center
                        // Note that the Y coordinate is relative to the domain origin to keep the coordinate
                        // system continuous across ranks
                        const float xp = coord_x + 0.5;
                        const float yp = interface.coord_y_exterior[bound_num] + grid.y_offset + 0.5;
                        const float zp = coord_z + 0.5;
                        interface.calcCritDiagonalLength(
                            index_exterior, xp, yp, zp, interface.octahedron_center(3 * index_exterior),
                            interface.octahedron_center(3 * index_exterior + 1),
                            interface.octahedron_center(3 * index_exterior + 2), my_orientation,
                            orientation.grain_unit_vector, phase_id(index_exterior));
                        celldata.cell_type(index_exterior) = Active;
                    }
                }
            });
    }
    Kokkos::fence();
}
//*****************************************************************************/
// Jump to the next time step with work to be done, if no melting or solidification events occur in the next 5000 time
// steps
template <typename MemorySpace>
void jumpTimeStep(int &cycle, Nucleation<MemorySpace> &nucleation, Temperature<MemorySpace> &temperature,
                  const Grid &grid, CellData<MemorySpace> &celldata, const int id, const int layernumber, const int np,
                  Orientation<MemorySpace> &orientation, Print &print, const double deltat,
                  Interface<MemorySpace> &interface) {

    int local_next_melt_time_step;
    if (temperature.liquidus_time_counter == temperature.num_liquidus_times_this_layer)
        local_next_melt_time_step = std::numeric_limits<int>::max();
    else
        local_next_melt_time_step = temperature.liquidus_time_list_host(temperature.liquidus_time_counter);
    int next_melt_time_step;
    MPI_Allreduce(&local_next_melt_time_step, &next_melt_time_step, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    if ((next_melt_time_step - cycle) > 5000) {
        // Print current state of the system for desired output fields for any of the time
        // steps between now and when melting/solidification occurs again, if the print option for idle frame
        // printing was toggled

        print.printIdleIntralayer(id, np, layernumber, deltat, cycle, grid, celldata, temperature, interface,
                                  orientation, next_melt_time_step);
        // Jump to next time step when melting occurs
        cycle = next_melt_time_step - 1;
        // If nucleation events were possible in any of the skipped time steps, update the counter accordingly as
        // these nucleation events could not have occured without any available liquid cells
        nucleation.advanceCounterSkippedTimeSteps(cycle);
        if (id == 0)
            std::cout << "Jumping to cycle " << cycle + 1 << std::endl;
    }
}

//*****************************************************************************/
// Prints intermediate code output to stdout and checks to see if solidification is complete
template <typename MemorySpace>
void intermediateOutputAndCheck(const int id, const int np, int &cycle, const Grid &grid,
                                int successful_nuc_events_this_rank, int &x_switch, Nucleation<MemorySpace> &nucleation,
                                CellData<MemorySpace> &celldata, Temperature<MemorySpace> &temperature,
                                std::string simulation_type, const int layernumber,
                                Orientation<MemorySpace> &orientation, Print &print, const double deltat,
                                Interface<MemorySpace> &interface) {

    auto grain_id = celldata.getGrainIDSubview(grid);
    int local_superheated_cells, local_undercooled_cells, local_active_cells, local_solid_cells;
    Kokkos::parallel_reduce(
        "IntermediateOutput", grid.domain_size,
        KOKKOS_LAMBDA(const int &index, int &sum_superheated, int &sum_undercooled, int &sum_active, int &sum_solid) {
            int cell_type_this_cell = celldata.cell_type(index);
            if (cell_type_this_cell == Liquid) {
                if (cycle < temperature.last_time_below_liquidus(index))
                    sum_superheated += 1;
                else
                    sum_undercooled += 1;
            }
            else if (cell_type_this_cell == Active)
                sum_active += 1;
            else if (cell_type_this_cell == Solid)
                sum_solid += 1;
        },
        local_superheated_cells, local_undercooled_cells, local_active_cells, local_solid_cells);

    int global_successful_nuc_events_this_rank = 0;
    int global_superheated_cells, global_undercooled_cells, global_active_cells, global_solid_cells;
    MPI_Reduce(&local_superheated_cells, &global_superheated_cells, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_undercooled_cells, &global_undercooled_cells, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_active_cells, &global_active_cells, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_solid_cells, &global_solid_cells, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&successful_nuc_events_this_rank, &global_successful_nuc_events_this_rank, 1, MPI_INT, MPI_SUM, 0,
               MPI_COMM_WORLD);
    // Cells of interest are those currently undergoing a melting-solidification cycle
    int remaining_cells_of_interest;

    if (id == 0) {
        std::cout << "Current time step " << cycle << " on layer number " << layernumber << std::endl;
        std::cout << "Number of liquid cells in simulation (superheated/undercooled): " << global_superheated_cells
                  << "/" << global_undercooled_cells << std::endl;
        std::cout << "Number of active (solid-liquid interface) cells in simulation: " << global_active_cells
                  << std::endl;
        std::cout << "Number of solid cells in simulation (finished/to be remelted): " << global_solid_cells
                  << std::endl;
        std::cout << "Number of nucleation events during simulation of this layer: "
                  << global_successful_nuc_events_this_rank << std::endl;
        std::cout << "======================================================================================"
                  << std::endl;
        remaining_cells_of_interest = global_active_cells + global_undercooled_cells + global_superheated_cells;
    }
    int local_liquidus_events_remaining = temperature.num_liquidus_times_this_layer - temperature.liquidus_time_counter;
    int liquidus_events_remaining;
    MPI_Allreduce(&local_liquidus_events_remaining, &liquidus_events_remaining, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_liquidus_events_remaining, &liquidus_events_remaining, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Bcast(&remaining_cells_of_interest, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Every cell has crossed the liquidus for the final time and all cells are solid
    if ((liquidus_events_remaining == 0) && (remaining_cells_of_interest == 0))
        x_switch = 1;
    if ((x_switch == 0) && (simulation_type != "Directional") && (remaining_cells_of_interest == 0))
        jumpTimeStep(cycle, nucleation, temperature, grid, celldata, id, layernumber, np, orientation, print, deltat,
                     interface);
}

//*****************************************************************************/
// Prints intermediate code output to stdout and checks to see the single grain simulation end condition (the grain has
// reached a domain edge) has been satisfied
template <typename ViewTypeInt>
void intermediateOutputAndCheck(const int id, int cycle, const Grid &grid, int &x_switch, ViewTypeInt cell_type) {

    int local_liquid_cells, local_active_cells, local_solid_cells;
    using memory_space = typename ViewTypeInt::memory_space;
    Kokkos::View<bool **, memory_space> edges_reached(Kokkos::ViewAllocateWithoutInitializing("edges_reached"), 3,
                                                      2); // init to false
    Kokkos::deep_copy(edges_reached, false);

    Kokkos::parallel_reduce(
        "IntermediateOutput", grid.domain_size,
        KOKKOS_LAMBDA(const int &index, int &sum_liquid, int &sum_active, int &sum_solid) {
            if (cell_type(index) == Liquid)
                sum_liquid += 1;
            else if (cell_type(index) == Active) {
                sum_active += 1;
                // Did this cell reach a domain edge?
                int coord_x = grid.getCoordX(index);
                int coord_y = grid.getCoordY(index);
                int coord_z = grid.getCoordZ(index);
                int coord_y_global = coord_y + grid.y_offset;
                if (coord_x == 0)
                    edges_reached(0, 0) = true;
                if (coord_x == grid.nx - 1)
                    edges_reached(0, 1) = true;
                if (coord_y_global == 0)
                    edges_reached(1, 0) = true;
                if (coord_y_global == grid.ny - 1)
                    edges_reached(1, 1) = true;
                if (coord_z == 0)
                    edges_reached(2, 0) = true;
                if (coord_z == grid.nz - 1)
                    edges_reached(2, 1) = true;
            }
            else if (cell_type(index) == Solid)
                sum_solid += 1;
        },
        local_liquid_cells, local_active_cells, local_solid_cells);

    int global_liquid_cells, global_active_cells, global_solid_cells;
    MPI_Reduce(&local_liquid_cells, &global_liquid_cells, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_active_cells, &global_active_cells, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_solid_cells, &global_solid_cells, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (id == 0)
        std::cout << "cycle = " << cycle << " : Liquid cells = " << global_liquid_cells
                  << " Active cells = " << global_active_cells << " Solid cells = " << global_solid_cells << std::endl;

    // Each rank checks to see if a global domain boundary was reached
    auto edges_reached_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), edges_reached);
    int x_switch_local = 0;
    std::vector<std::string> edge_dims = {"X", "Y", "Z"};
    std::vector<std::string> edge_names = {"Lower", "Upper"};
    for (int edgedim = 0; edgedim < 3; edgedim++) {
        for (int edgename = 0; edgename < 2; edgename++) {
            if (edges_reached_host(edgedim, edgename)) {
                std::cout << edge_names[edgename] << " edge of domain in the " << edge_dims[edgedim]
                          << " direction was reached on rank " << id << " and cycle " << cycle
                          << "; simulation is complete" << std::endl;
                x_switch_local = 1;
            }
        }
    }
    // Simulation ends if a global domain boundary was reached on any rank
    MPI_Allreduce(&x_switch_local, &x_switch, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
}

#endif
