// Copyright Lawrence Livermore National Security, LLC and other ExaCA Project Developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#ifndef EXACA_TEMPS_HPP
#define EXACA_TEMPS_HPP

#include "CAgrid.hpp"
#include "CAinputdata.hpp"
#include "CAparsefiles.hpp"
#include "CAtypes.hpp"
#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// Reduced form of the time-temperature history and temperature field variables used by ExaCA
template <typename MemorySpace>
struct Temperature {

    using memory_space = MemorySpace;
    using view_type_short = Kokkos::View<short *, memory_space>;
    using view_type_int = Kokkos::View<int *, memory_space>;
    using view_type_int_2d = Kokkos::View<int **, memory_space>;
    using view_type_int_3d = Kokkos::View<int ***, memory_space>;
    using view_type_double = Kokkos::View<double *, memory_space>;
    using view_type_double_2d = Kokkos::View<double **, memory_space>;
    using view_type_float = Kokkos::View<float *, memory_space>;
    using view_type_float_2d = Kokkos::View<float **, memory_space>;
    using view_type_short_host = typename view_type_short::host_mirror_type;
    using view_type_int_host = typename view_type_int::host_mirror_type;
    using view_type_int_2d_host = typename view_type_int_2d::host_mirror_type;
    using view_type_int_3d_host = typename view_type_int_3d::host_mirror_type;
    using view_type_double_host = typename view_type_double::host_mirror_type;
    using view_type_double_2d_host = typename view_type_double_2d::host_mirror_type;
    using view_type_float_host = typename view_type_float::host_mirror_type;
    using view_type_float_2d_host = typename view_type_float_2d::host_mirror_type;
    using view_type_coupled = Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace>;

    // Using the default exec space for this memory space.
    using execution_space = typename memory_space::execution_space;

    int liquidus_time_counter = 0;
    int num_liquidus_times_this_layer = 0;
    // Only nonzero in the special case of 0 cooling rate/uniform undercooling
    float init_undercooling = 0.0;
    // Times at which each cell crosses the liquidus temperature
    view_type_int_host liquidus_time_list_host;
    // Cells associated with each value in liquidus_time_list_host
    view_type_int liquidus_cell_list;
    // Cooling rate associated with each value in liquidus_time_list_host (INT_MAX placeholder for cells that are
    // heating)
    view_type_float cooling_rate_list;
    // The last time a given cell cooled below the liquidus (0.0 placeholder if it hasn't cooled below the liquidus)
    view_type_int last_time_below_liquidus;
    // Cooling rate currently in use by each cell
    view_type_float current_cooling_rate;
    // The number of times that each CA cell will traverse the liquidus during this layer
    view_type_int_host number_of_solidification_events;
    int max_num_solidification_events = 1;
    // Data structure for storing raw temperature data from file(s)
    // Store data as double - needed for small time steps to resolve local differences in solidification conditions
    // Each data point has 6 values (X, Y, Z coordinates, melting time, liquidus time, and cooling rate)
    const int num_temperature_components = 6;
    view_type_double_2d_host raw_temperature_data;
    // These contain "number_of_layers" values corresponding to the location within "raw_temperature_data" of the first
    // data element in each temperature file, if used
    view_type_int_host first_value, last_value;
    // Layer associated with last time a cell undergoes melting/soldification
    view_type_short layer_id_all_layers, layer_id;
    // Number of times in the layer a cell has undergone solidification so far (optional to store based on selected
    // inputs)
    bool store_solidification_event_counter = false;
    view_type_int solidification_event_counter;
    // Undercooling when a solidification first began in a cell (optional to store based on selected inputs)
    bool store_undercooling_current = false;
    view_type_float undercooling_current_all_layers, undercooling_current;
    // Undercooling when a solidification ended in a cell (optional to store based on selected inputs)
    bool store_undercooling_solidification_start = false;
    view_type_float undercooling_solidification_start_all_layers, undercooling_solidification_start;
    // Temperature field inputs from file
    TemperatureInputs _inputs;

    // Constructor creates views with size based on the grid inputs. If any of the output options
    // solidification_event_counter, undercooling_solidification_start, undercooling_current, and layer_id are toggled,
    // the associated views are allocated and filled during the simulation
    Temperature(const Grid &grid, TemperatureInputs inputs, PrintInputs print_inputs,
                const int est_num_temperature_data_points = 100000)
        : liquidus_time_list_host(
              view_type_int_host(Kokkos::ViewAllocateWithoutInitializing("liquidus_time_list"), grid.domain_size))
        , liquidus_cell_list(
              view_type_int(Kokkos::ViewAllocateWithoutInitializing("liquidus_cell_list"), grid.domain_size))
        , cooling_rate_list(
              view_type_float(Kokkos::ViewAllocateWithoutInitializing("cooling_rate_list"), grid.domain_size))
        , last_time_below_liquidus(view_type_int("last_time_below_liquidus", grid.domain_size))
        , current_cooling_rate(view_type_float("current_cooling_rate", grid.domain_size))
        , number_of_solidification_events(view_type_int_host(
              Kokkos::ViewAllocateWithoutInitializing("number_of_solidification_events"), grid.domain_size))
        , raw_temperature_data(view_type_double_2d_host(Kokkos::ViewAllocateWithoutInitializing("raw_temperature_data"),
                                                        est_num_temperature_data_points, num_temperature_components))
        , first_value(view_type_int_host(Kokkos::ViewAllocateWithoutInitializing("first_value"), grid.number_of_layers))
        , last_value(view_type_int_host(Kokkos::ViewAllocateWithoutInitializing("last_value"), grid.number_of_layers))
        , layer_id_all_layers(view_type_short(Kokkos::ViewAllocateWithoutInitializing("layer_id_all_layers"),
                                              grid.domain_size_all_layers))
        , solidification_event_counter(
              view_type_int(Kokkos::ViewAllocateWithoutInitializing("solidification_event_counter"), 1))
        , undercooling_current_all_layers(view_type_float(
              Kokkos::ViewAllocateWithoutInitializing("undercooling_current_all_layers"), grid.domain_size_all_layers))
        , undercooling_solidification_start_all_layers(
              view_type_float(Kokkos::ViewAllocateWithoutInitializing("undercooling_current_all_layers"), 1))

        , _inputs(inputs) {

        Kokkos::deep_copy(layer_id_all_layers, -1);
        layer_id = getLayerSubview(layer_id_all_layers, grid.layer_range);

        // Allocate views for optional stored data based on print inputs
        initOptionalViews(grid, print_inputs);
    }

    // Constructor using in-memory temperature data from external source (input_temperature_data)
    Temperature(const int id, const int np, const Grid &grid, TemperatureInputs inputs, PrintInputs print_inputs,
                view_type_coupled input_temperature_data, std::vector<int> first_value_finch,
                std::vector<int> last_value_finch)
        : liquidus_time_list_host(
              view_type_int_host(Kokkos::ViewAllocateWithoutInitializing("liquidus_time_list"), 2 * grid.domain_size))
        , liquidus_cell_list(
              view_type_int(Kokkos::ViewAllocateWithoutInitializing("liquidus_cell_list"), 2 * grid.domain_size))
        , cooling_rate_list(
              view_type_float(Kokkos::ViewAllocateWithoutInitializing("cooling_rate_list"), 2 * grid.domain_size))
        , last_time_below_liquidus(view_type_int("last_time_below_liquidus", grid.domain_size))
        , current_cooling_rate(view_type_float("current_cooling_rate", grid.domain_size))
        , number_of_solidification_events(view_type_int_host(
              Kokkos::ViewAllocateWithoutInitializing("number_of_solidification_events"), grid.domain_size))
        , raw_temperature_data(view_type_double_2d_host(Kokkos::ViewAllocateWithoutInitializing("raw_temperature_data"),
                                                        input_temperature_data.extent(0) * (inputs.number_of_copies),
                                                        num_temperature_components))
        , first_value(view_type_int_host(Kokkos::ViewAllocateWithoutInitializing("first_value"), grid.number_of_layers))
        , last_value(view_type_int_host(Kokkos::ViewAllocateWithoutInitializing("last_value"), grid.number_of_layers))
        , layer_id_all_layers(view_type_short(Kokkos::ViewAllocateWithoutInitializing("layer_id_all_layers"),
                                              grid.domain_size_all_layers))
        , solidification_event_counter(
              view_type_int(Kokkos::ViewAllocateWithoutInitializing("solidification_event_counter"), 1))
        , undercooling_current_all_layers(view_type_float(
              Kokkos::ViewAllocateWithoutInitializing("undercooling_current_all_layers"), grid.domain_size_all_layers))
        , undercooling_solidification_start_all_layers(
              view_type_float(Kokkos::ViewAllocateWithoutInitializing("undercooling_current_all_layers"), 1))
        , _inputs(inputs) {

        copyTemperatureData(id, np, 0, grid, input_temperature_data, first_value_finch, last_value_finch);
        initOptionalViews(grid, print_inputs);
        Kokkos::deep_copy(layer_id_all_layers, -1);
        layer_id = getLayerSubview(layer_id_all_layers, grid.layer_range);
    }

    void initOptionalViews(const Grid &grid, PrintInputs print_inputs) {
        if ((print_inputs.intralayer_solidification_event_counter) ||
            (print_inputs.interlayer_solidification_event_counter)) {
            store_solidification_event_counter = true;
            // Solidification event counter is only stored for the current layer, init to zeros
            Kokkos::realloc(solidification_event_counter, grid.domain_size);
            Kokkos::deep_copy(solidification_event_counter, 0);
        }
        if ((print_inputs.intralayer_undercooling_current) || (print_inputs.interlayer_undercooling_current)) {
            store_undercooling_current = true;
            // Undercooling is stored for all layers, as is the subview for the current layer's undercooling, init both
            // to zeros
            Kokkos::realloc(undercooling_current_all_layers, grid.domain_size_all_layers);
            Kokkos::deep_copy(undercooling_current_all_layers, 0.0);
            // Subview for current layer
            undercooling_current = getLayerSubview(undercooling_current_all_layers, grid.layer_range);
        }
        if ((print_inputs.intralayer_undercooling_solidification_start) ||
            (print_inputs.interlayer_undercooling_solidification_start)) {
            store_undercooling_solidification_start = true;
            Kokkos::realloc(undercooling_solidification_start_all_layers, grid.domain_size_all_layers);
            Kokkos::deep_copy(undercooling_solidification_start_all_layers, 0.0);
            undercooling_solidification_start =
                getLayerSubview(undercooling_solidification_start_all_layers, grid.layer_range);
        }
    }

    // If using a problem type that involves translating or mirroring temperature data, output a new X or Y value based
    // on value from the file/from finch
    double getTranslatedXY(const double input_xy_value, const double xy_min, const double xy_max,
                           const double xy_offset, const int line_number, const bool mirror) {
        double xy_translated = input_xy_value + line_number * xy_offset;
        if ((mirror) && (line_number % 2 == 1))
            xy_translated = xy_max - xy_translated + xy_min;
        return xy_translated;
    }

    // Based on the X coordinate compared to the global domain bounds and the Y coordinate compared to the MPI rank Y
    // bounds (integers), return whether or not this translated coordinate is in bounds on this rank
    bool translatedXYInBounds(const double x_translated, const double y_translated, const Grid grid) {
        const int coord_x = Kokkos::round((x_translated - grid.x_min) / grid.deltax);
        const int coord_y_global = Kokkos::round((y_translated - grid.y_min) / grid.deltax);
        bool xy_in_bounds;
        if ((coord_x >= 0) && (coord_x < grid.nx) && (coord_y_global >= grid.y_offset) &&
            (coord_y_global < grid.y_offset + grid.ny_local))
            xy_in_bounds = true;
        else
            xy_in_bounds = false;
        return xy_in_bounds;
    }

    // Copy data from external source onto the appropriate MPI ranks
    void copyTemperatureData(const int id, const int np, const int first_layer_init, const Grid &grid,
                             view_type_coupled input_temperature_data, std::vector<int> first_value_finch,
                             std::vector<int> last_value_finch, const int resize_padding = 100) {

        // Take first num_temperature_components columns of input_temperature_data
        int finch_temp_components = input_temperature_data.extent(1);
        // Number of layers of Finch data to copy into temperature struct at once
        int num_layers_init;
        if (_inputs.layerwise_temp_read)
            num_layers_init = 1;
        else
            num_layers_init = _inputs.temp_files_in_series;
        int last_layer_init = first_layer_init + num_layers_init;
        int temperature_point_counter = 0;
        for (int layer = first_layer_init; layer < last_layer_init; layer++) {

            // input_temperature_data contains data for one or multiple Finch simulations of various layers - which are
            // tracked via the first_value_finch and last_value_finch vectors. The first_value and last_value vectors
            // track, on each ExaCA MPI rank, which portions of the data are associated with which layer
            const int layer_start = first_value_finch[layer];
            const int layer_end = last_value_finch[layer];
            std::pair<int, int> finch_layer_range = std::make_pair(layer_start, layer_end);
            auto input_temperature_data_layer = Kokkos::subview(input_temperature_data, finch_layer_range, Kokkos::ALL);

            // First, store data with Y coordinates in bounds for this rank in raw_temperature_data
            const int temperature_point_counter_start = temperature_point_counter;
            extractTemperatureData(input_temperature_data_layer, raw_temperature_data, temperature_point_counter, grid);

            // Communication pattern - sending to right, receiving from left
            int left, right;
            if (id == 0)
                left = np - 1;
            else
                left = id - 1;
            if (id == np - 1)
                right = 0;
            else
                right = id + 1;

            int finch_data_size = input_temperature_data_layer.extent(0);
            std::cout << "Rank " << id << " has " << finch_data_size << " events from Finch simulation number " << layer
                      << ", corresponding to events " << layer_start << " through " << layer_end - 1 << std::endl;
            // Send and recieve data so each rank parses all finch data points
            view_type_coupled finch_data_send(Kokkos::ViewAllocateWithoutInitializing("finch_data_send"),
                                              layer_end - layer_start, finch_temp_components);
            Kokkos::deep_copy(finch_data_send, input_temperature_data_layer);
            int send_data_size = finch_data_size;
            for (int i = 0; i < np - 1; i++) {
                // Get size for sending/receiving
                int recv_data_size;
                MPI_Request send_request_size, recv_request_size;
                MPI_Isend(&send_data_size, 1, MPI_INT, right, 0, MPI_COMM_WORLD, &send_request_size);
                MPI_Irecv(&recv_data_size, 1, MPI_INT, left, 0, MPI_COMM_WORLD, &recv_request_size);
                MPI_Wait(&send_request_size, MPI_STATUS_IGNORE);
                MPI_Wait(&recv_request_size, MPI_STATUS_IGNORE);
                // Allocate view for received data
                view_type_coupled finch_data_recv(Kokkos::ViewAllocateWithoutInitializing("finch_data_recv"),
                                                  recv_data_size, finch_temp_components);

                // Send data to the right, recieve data from the left - if needed, increase size of data stored on this
                // rank to accomodate received data
                MPI_Request send_request_data, recv_request_data;
                MPI_Isend(finch_data_send.data(), send_data_size * finch_temp_components, MPI_DOUBLE, right, 1,
                          MPI_COMM_WORLD, &send_request_data);
                MPI_Irecv(finch_data_recv.data(), recv_data_size * finch_temp_components, MPI_DOUBLE, left, 1,
                          MPI_COMM_WORLD, &recv_request_data);
                int current_size = raw_temperature_data.extent(0);
                if (temperature_point_counter + _inputs.number_of_copies * recv_data_size >= current_size)
                    Kokkos::resize(raw_temperature_data,
                                   temperature_point_counter + _inputs.number_of_copies * recv_data_size +
                                       resize_padding,
                                   num_temperature_components);
                MPI_Wait(&send_request_data, MPI_STATUS_IGNORE);
                MPI_Wait(&recv_request_data, MPI_STATUS_IGNORE);

                // Unpack the appropriate received data into raw_temperature_data
                extractTemperatureData(finch_data_recv, raw_temperature_data, temperature_point_counter, grid);

                // Replace send buffer with the received data
                Kokkos::realloc(finch_data_send, recv_data_size, finch_temp_components);
                Kokkos::deep_copy(finch_data_send, finch_data_recv);
                send_data_size = recv_data_size;
            }
            first_value(layer) = temperature_point_counter_start;
            last_value(layer) = temperature_point_counter;
            std::cout << "Rank " << id << " has " << temperature_point_counter - temperature_point_counter_start
                      << " events to simulate with ExaCA from Finch simulation number " << layer << std::endl;
        }
        // Resize with the number of temperature data points on this rank now known
        Kokkos::resize(raw_temperature_data, temperature_point_counter, num_temperature_components);

        // If reading multiple files during initialization, associated each layer with the appropriate portion of the
        // Finch data
        if (!(_inputs.layerwise_temp_read))
            fillFirstLastValue(grid.number_of_layers, _inputs.temp_files_in_series);
    }

    // Get temperature data from Finch and store as raw temperatures.
    template <typename SrcViewType, typename DstViewType>
    void extractTemperatureData(const SrcViewType &temp_src, DstViewType &temp_dst, int &temp_count, const Grid grid) {
        int data_size = temp_src.extent(0);
        // Ensure that dst view has sufficient capacity to store values from temp_src
        int max_num_pts_extracted = data_size * _inputs.number_of_copies;
        int current_dst_view_size = temp_dst.extent(0);
        int free_space_dst_view = temp_dst.extent(0) - temp_count;
        if (max_num_pts_extracted > free_space_dst_view)
            Kokkos::resize(temp_dst, current_dst_view_size + max_num_pts_extracted - free_space_dst_view,
                           num_temperature_components);
        for (int n = 0; n < data_size; n++) {
            for (int l = 0; l < _inputs.number_of_copies; l++) {
                // Consider each translated point in either X or Y, optionally mirroring the point in the non-translated
                // direction
                // If using a problem type that involves translating or mirroring temperature data, output a new X value
                // based on value from the file/from finch
                const double x_translated =
                    getTranslatedXY(temp_src(n, 0), grid.x_min, grid.x_max, _inputs.x_offset, l, _inputs.mirror_x);
                const double y_translated =
                    getTranslatedXY(temp_src(n, 1), grid.y_min, grid.y_max, _inputs.y_offset, l, _inputs.mirror_y);
                const bool xy_in_bounds = translatedXYInBounds(x_translated, y_translated, grid);
                // Only store if the point is in-bounds in X and Y
                if (xy_in_bounds) {
                    temp_dst(temp_count, 0) = x_translated;
                    temp_dst(temp_count, 1) = y_translated;
                    temp_dst(temp_count, 2) = temp_src(n, 2);
                    // Offset for melting and liquidus times
                    double time_offset = l * _inputs.temporal_offset;
                    temp_dst(temp_count, 3) = temp_src(n, 3) + time_offset;
                    temp_dst(temp_count, 4) = temp_src(n, 4) + time_offset;
                    temp_dst(temp_count, 5) = temp_src(n, 5);
                    // Increment counter for each point stored on this rank
                    temp_count++;
                }
            }
        }
    }

    // Read and parse the temperature file (double precision values in a comma-separated, ASCII format with a header
    // line - or a binary string of double precision values), storing the x, y, z, tm, tl, cr values in the
    // raw_temperature_data view. Each rank only contains the points corresponding to cells within the associated Y
    // bounds. number_of_temperature_data_points is incremented on each rank as data is added to raw_temperature_data
    void parseTemperatureData(const std::string tempfile_thislayer, const Grid grid,
                              int &number_of_temperature_data_points, const bool binary_input_data,
                              const int temperature_buffer_increment = 100000) {

        std::ifstream temperature_filestream;
        temperature_filestream.open(tempfile_thislayer);
        if (binary_input_data) {
            while (!temperature_filestream.eof()) {
                std::vector<double> parsed_data(num_temperature_components);
                // Get x, y, z, tm, tl, cr - additional temperature components (Gx, Gy, Gz) have so far have been unused
                // in ExaCA - read and ignore these
                for (int component = 0; component < num_temperature_components; component++)
                    parsed_data[component] = readBinaryData<double>(temperature_filestream);
                // If no data was extracted from the stream, the end of the file was reached
                if (!(temperature_filestream))
                    break;
                // If translating/rotating line scan data, do this in a loop
                for (int l = 0; l < _inputs.number_of_copies; l++) {
                    // Consider each translated point in either X or Y, optionally mirroring the point in the
                    // non-translated direction If using a problem type that involves translating or mirroring
                    // temperature data, output a new X value based on value from the file/from finch
                    const double x_translated =
                        getTranslatedXY(parsed_data[0], grid.x_min, grid.x_max, _inputs.x_offset, l, _inputs.mirror_x);
                    const double y_translated =
                        getTranslatedXY(parsed_data[1], grid.y_min, grid.y_max, _inputs.y_offset, l, _inputs.mirror_y);
                    const bool xy_in_bounds = translatedXYInBounds(x_translated, y_translated, grid);
                    if (xy_in_bounds) {
                        // This data point is inside the bounds of interest for this MPI rank
                        // Store the x and y values in RawData
                        raw_temperature_data(number_of_temperature_data_points, 0) = x_translated;
                        raw_temperature_data(number_of_temperature_data_points, 1) = y_translated;
                        // z coordinate is stored as-is
                        raw_temperature_data(number_of_temperature_data_points, 2) = parsed_data[2];
                        // tm and tl may require an offset in time if l > 0
                        const double time_offset = l * _inputs.temporal_offset;
                        raw_temperature_data(number_of_temperature_data_points, 3) = parsed_data[3] + time_offset;
                        raw_temperature_data(number_of_temperature_data_points, 4) = parsed_data[4] + time_offset;
                        // cr is stored as-is
                        raw_temperature_data(number_of_temperature_data_points, 5) = parsed_data[5];
                        // Increment number of temperature data points stored
                        number_of_temperature_data_points++;
                        int raw_temperature_data_extent = raw_temperature_data.extent(0);
                        // Adjust size of raw_temperature_data if it is near full
                        if (number_of_temperature_data_points >= raw_temperature_data_extent)
                            Kokkos::resize(raw_temperature_data,
                                           raw_temperature_data_extent + temperature_buffer_increment,
                                           num_temperature_components);
                    }
                }
            }
        }
        else {
            // Get number of columns in this temperature file
            std::string header_line;
            getline(temperature_filestream, header_line);
            int vals_per_line = checkForHeaderValues(header_line);
            while (!temperature_filestream.eof()) {
                std::vector<std::string> parsed_line(
                    num_temperature_components); // Each line has an x, y, z, tm, tl, cr
                std::string read_line;
                if (!getline(temperature_filestream, read_line))
                    break;
                // Only parse the first 6 columns of the temperature data
                splitString(read_line, parsed_line, vals_per_line);
                // If translating/rotating line scan data, do this in a loop
                for (int l = 0; l < _inputs.number_of_copies; l++) {
                    // Consider each translated point in either X or Y, optionally mirroring the point in the
                    // non-translated direction If using a problem type that involves translating or mirroring
                    // temperature data, output a new X value based on value from the file/from finch
                    const double x_temperature_point = getInputDouble(parsed_line[0]);
                    const double y_temperature_point = getInputDouble(parsed_line[1]);
                    const double x_translated = getTranslatedXY(x_temperature_point, grid.x_min, grid.x_max,
                                                                _inputs.x_offset, l, _inputs.mirror_x);
                    const double y_translated = getTranslatedXY(y_temperature_point, grid.y_min, grid.y_max,
                                                                _inputs.y_offset, l, _inputs.mirror_y);
                    const bool xy_in_bounds = translatedXYInBounds(x_translated, y_translated, grid);
                    // Only store if the point is in-bounds in X and Y
                    if (xy_in_bounds) {
                        // Store the translated/mirrored x, y, and offset tm and tl values in raw_temperature_data along
                        // with z and cr value (unchanged during any translation/mirroring of data)
                        raw_temperature_data(number_of_temperature_data_points, 0) = x_translated;
                        raw_temperature_data(number_of_temperature_data_points, 1) = y_translated;
                        raw_temperature_data(number_of_temperature_data_points, 2) = getInputDouble(parsed_line[2]);
                        const double time_offset = l * _inputs.temporal_offset;
                        raw_temperature_data(number_of_temperature_data_points, 3) =
                            getInputDouble(parsed_line[3]) + time_offset;
                        raw_temperature_data(number_of_temperature_data_points, 4) =
                            getInputDouble(parsed_line[4]) + time_offset;
                        raw_temperature_data(number_of_temperature_data_points, 5) = getInputDouble(parsed_line[5]);
                        number_of_temperature_data_points++;
                        // Adjust size of raw_temperature_data if it is near full
                        int raw_temperature_data_extent = raw_temperature_data.extent(0);
                        // Adjust size of raw_temperature_data if it is near full
                        if (number_of_temperature_data_points >= raw_temperature_data_extent) {
                            Kokkos::resize(raw_temperature_data,
                                           raw_temperature_data_extent + temperature_buffer_increment,
                                           num_temperature_components);
                        }
                    }
                }
            }
        }
    }

    // For simulations where temperature data is read during ExaCA initialization and reused for multiple layers, fill
    // out first_value and last_value to associate each layer with the correct temperature data
    void fillFirstLastValue(const int number_of_layers, const int temp_files_in_series) {
        for (int layer_read_count = temp_files_in_series; layer_read_count < number_of_layers; layer_read_count++) {
            if (temp_files_in_series == 1) {
                // All layers have the same temperature data
                first_value(layer_read_count) = first_value(layer_read_count - 1);
                last_value(layer_read_count) = last_value(layer_read_count - 1);
            }
            else {
                // All layers have different temperature data but in a repeating pattern
                int repeated_file = layer_read_count % temp_files_in_series;
                first_value(layer_read_count) = first_value(repeated_file);
                last_value(layer_read_count) = last_value(repeated_file);
            }
        }
    }

    // Read in temperature data from files, stored in the host view raw_temperature_data, with the appropriate MPI ranks
    // storing the appropriate data
    void readTemperatureData(int id, const Grid &grid, int layernumber) {

        // Y coordinates of this rank's data, inclusive and including ghost nodes
        int lower_y_bound = grid.y_offset;
        int upper_y_bound = grid.y_offset + grid.ny_local - 1;

        std::cout << "On MPI rank " << id << ", the Y bounds (in cells) are [" << lower_y_bound << "," << upper_y_bound
                  << "]" << std::endl;
        int number_of_temperature_data_points = 0;
        int first_layer_to_read, last_layer_to_read;
        if (_inputs.layerwise_temp_read) {
            first_layer_to_read = layernumber;
            last_layer_to_read = layernumber;
        }
        else {
            first_layer_to_read = 0;
            last_layer_to_read = std::min(grid.number_of_layers, _inputs.temp_files_in_series) - 1;
        }
        // Which temperature files should be read? Just the one file for layer "layernumber", or all of them?
        for (int layer_read_count = first_layer_to_read; layer_read_count <= last_layer_to_read; layer_read_count++) {

            std::string tempfile_thislayer;
            if (_inputs.layerwise_temp_read) {
                int LayerInSeries = layernumber % _inputs.temp_files_in_series;
                tempfile_thislayer = _inputs.temp_paths[LayerInSeries];
            }
            else
                tempfile_thislayer = _inputs.temp_paths[layer_read_count];

            first_value(layer_read_count) = number_of_temperature_data_points;
            // Read and parse temperature file for either binary or ASCII, storing the appropriate values on each MPI
            // rank within raw_temperature_data and incrementing number_of_temperature_data_points appropriately
            bool binary_input_data = checkTemperatureFileFormat(tempfile_thislayer);
            parseTemperatureData(tempfile_thislayer, grid, number_of_temperature_data_points, binary_input_data);
            last_value(layer_read_count) = number_of_temperature_data_points;
        } // End loop over all files read for all layers
        Kokkos::resize(raw_temperature_data, number_of_temperature_data_points, num_temperature_components);
        // Determine start values for each layer's data within raw_temperature_data, if all layers were read
        if (!(_inputs.layerwise_temp_read))
            fillFirstLastValue(grid.number_of_layers, _inputs.temp_files_in_series);
    }

    // Initialize temperature data with a fixed thermal gradient in Z (can also be zero) for Directional/SingleGrain
    // problem types
    void initialize(const int id, const std::string simulation_type, const Grid &grid, const double deltat) {

        // Initialize temperature field in Z direction with thermal gradient G set in input file
        // Liquidus front (InitUndercooling = 0) is at domain bottom for directional solidification, is at domain center
        // (with custom InitUndercooling value) for single grain solidification
        int location_init_undercooling, location_liquidus_isotherm;
        if (simulation_type == "Directional")
            location_init_undercooling = 0;
        else
            location_init_undercooling = Kokkos::floorf(static_cast<float>(grid.nz) / 2.0);

        // If thermal gradient is 0, liquidus isotherm does not exist - initialize to nz to avoid divide by zero error
        // and ensure all cells are initialized as undercooled (i.e., at Z coordinates less than nz)
        if (_inputs.G == 0)
            location_liquidus_isotherm = grid.nz;
        else
            location_liquidus_isotherm =
                location_init_undercooling + Kokkos::round(_inputs.init_undercooling / (_inputs.G * grid.deltax));
        // If cooling rate is 0, need to set an initial undercooling
        if (_inputs.R == 0)
            init_undercooling = _inputs.init_undercooling;

        // List of liquidus time events - Since iteration over liqudius time events does not occur for the SingleGrain
        // and Directional problem types (as they have a single last_time_below_liquidus and current_cooling_rate that
        // apply to all cells, and the Active cells present are undercooled at the start of the simulation), the
        // liquidus_time_list_host, liquidus_cell_list, and cooling_rate_list views are filled with data only for
        // reference in printing temperature field info
        std::vector<int> liquidus_time_step_read(2 * grid.domain_size), cell_read(2 * grid.domain_size);
        std::vector<float> cooling_rate_read(2 * grid.domain_size);
        auto current_cooling_rate_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), current_cooling_rate);
        auto last_time_below_liquidus_host =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), last_time_below_liquidus);
        for (int coord_z = 0; coord_z < grid.nz; coord_z++) {
            const int dist_from_liquidus = coord_z - location_liquidus_isotherm;
            for (int coord_x = 0; coord_x < grid.nx; coord_x++) {
                for (int coord_y = 0; coord_y < grid.ny_local; coord_y++) {
                    // 1D cell index
                    const int index = grid.get1DIndex(coord_x, coord_y, coord_z);
                    // Initially melted (time step 0)
                    liquidus_time_step_read[num_liquidus_times_this_layer] = 0;
                    cell_read[num_liquidus_times_this_layer] = index;
                    cooling_rate_read[num_liquidus_times_this_layer] = -1.0;
                    num_liquidus_times_this_layer++;
                    // Cells reach liquidus at a time dependent on their Z coordinate
                    // Cells with negative last_time_below_liquidus are already undercooled, times may be negative
                    liquidus_time_step_read[num_liquidus_times_this_layer] =
                        Kokkos::round(dist_from_liquidus * _inputs.G * grid.deltax / (_inputs.R * deltat));
                    cell_read[num_liquidus_times_this_layer] = index;
                    // Cells cool at a constant rate
                    cooling_rate_read[num_liquidus_times_this_layer] = _inputs.R * deltat;
                    // Cell will only have one cooling rate/liquidus time
                    last_time_below_liquidus_host(index) = liquidus_time_step_read[num_liquidus_times_this_layer];
                    current_cooling_rate_host(index) = cooling_rate_read[num_liquidus_times_this_layer];
                    num_liquidus_times_this_layer++;
                }
            }
        }

        // From the 3 vectors liquidus_time_step_read, cell_read, cooling_rate_read initialize the views
        // liquidus_time_list_host, liquidus_cell_list (stored on device), and cooling_rate_list (stored on device)
        initOrderedTimeTempHistory(liquidus_time_step_read, cell_read, cooling_rate_read,
                                   num_liquidus_times_this_layer);

        // Counter goes unused since all melting events have occurred and the time at which the cells go below the
        // liquidus/their cooling rates are also known. Set counter to max value
        liquidus_time_counter = num_liquidus_times_this_layer;

        // Copy host view data back to device
        current_cooling_rate = Kokkos::create_mirror_view_and_copy(memory_space(), current_cooling_rate_host);
        last_time_below_liquidus = Kokkos::create_mirror_view_and_copy(memory_space(), last_time_below_liquidus_host);

        // Each cell solidifies once
        Kokkos::deep_copy(number_of_solidification_events, 1);
        Kokkos::deep_copy(layer_id_all_layers, 0);
        MPI_Barrier(MPI_COMM_WORLD);
        if (id == 0) {
            std::cout << "Temperature field initialized for unidirectional solidification with G = " << _inputs.G
                      << " K/m, initial undercooling at Z = " << location_init_undercooling << " of "
                      << _inputs.init_undercooling << " K below the liquidus" << std::endl;
            std::cout << "Done with temperature field initialization" << std::endl;
        }
    }

    // From the 3 vectors liquidus_time_step_vec, cell_vec, cooling_rate_vec, initialize the views
    // liquidus_time_list_host, liquidus_cell_list (stored on device), and cooling_rate_list (stored on device)
    void initOrderedTimeTempHistory(std::vector<int> liquidus_time_step_vec, std::vector<int> cell_vec,
                                    std::vector<float> cooling_rate_vec, const int number_vals_stored) {

        // Reorder liquidus time events based on the times at which they occur
        std::vector<std::tuple<int, int, float>> time_temp_history;
        time_temp_history.reserve(number_vals_stored);
        for (int n = 0; n < number_vals_stored; n++) {
            time_temp_history.push_back(std::make_tuple(liquidus_time_step_vec[n], cell_vec[n], cooling_rate_vec[n]));
        }
        std::sort(time_temp_history.begin(), time_temp_history.end());

        // Resize views for number of values in the vectors
        Kokkos::realloc(liquidus_time_list_host, number_vals_stored);
        Kokkos::realloc(liquidus_cell_list, number_vals_stored);
        Kokkos::realloc(cooling_rate_list, number_vals_stored);

        // Copy to host view for liquidus_time_list, temporary host views for list of cells and cooling rates
        auto liquidus_cell_list_host = Kokkos::create_mirror_view(Kokkos::HostSpace(), liquidus_cell_list);
        auto cooling_rate_list_host = Kokkos::create_mirror_view(Kokkos::HostSpace(), cooling_rate_list);
        for (int n = 0; n < num_liquidus_times_this_layer; n++) {
            liquidus_time_list_host(n) = std::get<0>(time_temp_history[n]);
            liquidus_cell_list_host(n) = std::get<1>(time_temp_history[n]);
            cooling_rate_list_host(n) = std::get<2>(time_temp_history[n]);
        }

        // Copy host view data back to device
        liquidus_cell_list = Kokkos::create_mirror_view_and_copy(memory_space(), liquidus_cell_list_host);
        cooling_rate_list = Kokkos::create_mirror_view_and_copy(memory_space(), cooling_rate_list_host);
    }

    // Store temperature data for a hemispherical spot melt (single layer simulation)
    void storeSpotData(const int id, const Grid &grid, const double freezing_range, double deltat, double spot_radius) {

        // Outer edges of spots are initialized at the liquidus temperature
        // Spots cool at constant rate R, spot thermal gradient = G
        float isotherm_velocity = (_inputs.R / _inputs.G) * deltat / grid.deltax; // in cells per time step
        int spot_time_est = spot_radius / isotherm_velocity + (freezing_range / _inputs.R) / deltat; // in time steps
        // Spot center location - center of domain in X and Y, top of domain in Z
        float spot_center_x = spot_radius + 1;
        float spot_center_y = spot_radius + 1;
        float spot_center_z = grid.nz - 0.5;

        if (id == 0)
            std::cout << "Initializing temperature field for the hemispherical spot, which will take approximately "
                      << spot_time_est << " time steps to fully solidify" << std::endl;

        // Add spot data to raw_temperature_data as if read from a file
        // Ensure raw_temperature_data can store all data
        Kokkos::realloc(raw_temperature_data, grid.domain_size, 6);
        // Iterate over cells on this rank, determine if inside of spot
        for (int coord_z = 0; coord_z < grid.nz; coord_z++) {
            for (int coord_x = 0; coord_x < grid.nx; coord_x++) {
                for (int coord_y = 0; coord_y < grid.ny_local; coord_y++) {
                    // Distance of this cell from the spot center
                    float dist_z = spot_center_z - coord_z;
                    float dist_x = spot_center_x - coord_x;
                    int coord_y_global = coord_y + grid.y_offset;
                    float dist_y = spot_center_y - coord_y_global;
                    float tot_dist = Kokkos::hypot(dist_x, dist_y, dist_z);
                    if (tot_dist <= spot_radius) {
                        // x, y, z of location in global domain
                        raw_temperature_data(num_liquidus_times_this_layer, 0) =
                            grid.x_min + static_cast<double>(coord_x) * grid.deltax;
                        raw_temperature_data(num_liquidus_times_this_layer, 1) =
                            grid.y_min + static_cast<double>(coord_y_global) * grid.deltax;
                        raw_temperature_data(num_liquidus_times_this_layer, 2) =
                            grid.z_min + static_cast<double>(coord_z) * grid.deltax;
                        // Melting time in seconds - all cells melt at the start of the simulation
                        raw_temperature_data(num_liquidus_times_this_layer, 3) = 0.0;
                        // Liquidus time in seconds (related to distance to spot edge), must be >= deltat
                        raw_temperature_data(num_liquidus_times_this_layer, 4) =
                            deltat * (((spot_radius - tot_dist) / isotherm_velocity) + 1.0);
                        // Cooling rate per time step (K/s)
                        raw_temperature_data(num_liquidus_times_this_layer, 5) = _inputs.R;
                        // Increment counter for points on this rank that melt/resolidify
                        num_liquidus_times_this_layer++;
                    }
                }
            }
        }
        // Remove empty space in raw_temperature_data
        Kokkos::resize(raw_temperature_data, num_liquidus_times_this_layer, 6);
        // Start/end indices of temperature data in raw_temperature_data
        first_value(0) = 0;
        last_value(0) = num_liquidus_times_this_layer;
        MPI_Barrier(MPI_COMM_WORLD);
        if (id == 0)
            std::cout
                << "Spot melt temperature field initialized, number of cells to undergo melting/solidification is "
                << num_liquidus_times_this_layer << std::endl;
    }

    // Calculate the number of times that a cell in layer "layernumber" undergoes melting/solidification, and store in
    // max_solidification_events_host
    view_type_int_host calcNumberOfSolidificationEvents(const int id, const int layernumber, const int start_range,
                                                        const int end_range, const Grid &grid) {

        view_type_int_host _number_of_solidification_events("number_of_solidification_events_temp", grid.domain_size);
        for (int i = start_range; i < end_range; i++) {
            // Get the integer X, Y, Z coordinates associated with this data point, with the Y coordinate based on
            // local MPI rank's grid
            int coord_x = getTempCoordX(i, grid.x_min, grid.deltax);
            int coord_y = getTempCoordY(i, grid.y_min, grid.deltax, grid.y_offset);
            int coord_z = getTempCoordZ(i, grid.deltax, grid.layer_height, layernumber, grid.z_min_layer);
            // Convert to 1D coordinate in the current layer's domain
            int index = grid.get1DIndex(coord_x, coord_y, coord_z);
            _number_of_solidification_events(index)++;
        }
        // Get the max of _number_of_solidification_events across all ranks
        int max_count = 0;
        for (int i = 0; i < grid.domain_size; i++) {
            if (_number_of_solidification_events(i) > max_count)
                max_count = _number_of_solidification_events(i);
        }
        MPI_Allreduce(&max_count, &max_num_solidification_events, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (id == 0)
            std::cout << "The maximum number of melting/solidification events during layer " << layernumber << " is "
                      << max_num_solidification_events << std::endl;
        return _number_of_solidification_events;
    }

    // Read data from storage, and calculate the normalized x value of the data point
    int getTempCoordX(const int i, const double x_min, const double deltax) const {
        int x_coord = Kokkos::round((raw_temperature_data(i, 0) - x_min) / deltax);
        return x_coord;
    }
    // Read data from storage, and calculate the normalized y value of the data point. If the optional offset argument
    // is given, the return value is calculated relative to the edge of the MPI rank's local simulation domain (which is
    // offset by y_offset cells from the global domain edge)
    int getTempCoordY(const int i, const double y_min, const double deltax, const int y_offset = 0) const {
        int y_coord = Kokkos::round((raw_temperature_data(i, 1) - y_min) / deltax) - y_offset;
        return y_coord;
    }
    // Read data from storage, and calculate the normalized z value of the data point
    int getTempCoordZ(const int i, const double deltax, const int layer_height, const int layer_counter,
                      const view_type_double_host z_min_layer) const {
        int z_coord = Kokkos::round(
            (raw_temperature_data(i, 2) + deltax * layer_height * layer_counter - z_min_layer[layer_counter]) / deltax);
        return z_coord;
    }
    // Read data from storage, obtain melting time
    double getTempCoordTM(const int i) const {
        double t_melting = raw_temperature_data(i, 3);
        return t_melting;
    }
    // Read data from storage, obtain liquidus time
    double getTempCoordTL(const int i) const {
        double t_liquidus = raw_temperature_data(i, 4);
        return t_liquidus;
    }
    // Read data from storage, obtain cooling rate
    double getTempCoordCR(const int i) const {
        double cooling_rate = raw_temperature_data(i, 5);
        return cooling_rate;
    }

    // Initialize temperature fields for layer "layernumber" in case where temperature data comes from file(s), could be
    // done on device but currently relies on vector/host functions as is the case for the placeNuclei function
    void initialize(const int layernumber, const int id, const Grid &grid, const double freezing_range,
                    const double deltat) {

        // Counter starts at 0 for each layer
        liquidus_time_counter = 0;
        // Resize for new layer's domain size
        Kokkos::realloc(last_time_below_liquidus, grid.domain_size);
        Kokkos::realloc(current_cooling_rate, grid.domain_size);
        // Init values for views updated as heating above/cooling below the liquidus occur
        Kokkos::deep_copy(last_time_below_liquidus, std::numeric_limits<int>::max());
        Kokkos::deep_copy(current_cooling_rate, 0.0);
        // LayerID for the next layer
        layer_id = getLayerSubview(layer_id_all_layers, grid.layer_range);
        // Data was already read into the "raw_temperature_data" data structure
        // Determine which section of "raw_temperature_data" is relevant for this layer of the overall domain
        int start_range = first_value[layernumber];
        int end_range = last_value[layernumber];
        num_liquidus_times_this_layer = 2 * (end_range - start_range);

        // Get the number of times each cell goes below the liquidus, and the max number of solidification events in a
        // cell
        Kokkos::realloc(number_of_solidification_events, grid.domain_size);
        number_of_solidification_events =
            calcNumberOfSolidificationEvents(id, layernumber, start_range, end_range, grid);
        std::vector<int> liquidus_time_step_read(num_liquidus_times_this_layer),
            cell_read(num_liquidus_times_this_layer);
        std::vector<float> cooling_rate_read(num_liquidus_times_this_layer);

        double largest_time = 0;
        double largest_time_global = 0;
        if (id == 0)
            std::cout << "Range of raw data for layer " << layernumber << " on rank 0 is " << start_range << " to "
                      << end_range << std::endl;
        MPI_Barrier(MPI_COMM_WORLD);
        for (int i = start_range; i < end_range; i++) {
            // Get the integer X, Y, Z coordinates associated with this data point, along with the associated TM,
            // TL, CR values coord_y is relative to this MPI rank's grid, while coord_y_global is relative to the
            // overall simulation domain
            int coord_x = getTempCoordX(i, grid.x_min, grid.deltax);
            int coord_y = getTempCoordY(i, grid.y_min, grid.deltax, grid.y_offset);
            int coord_z = getTempCoordZ(i, grid.deltax, grid.layer_height, layernumber, grid.z_min_layer);
            double t_melting = getTempCoordTM(i);
            double t_liquidus = getTempCoordTL(i);
            double cooling_rate_cell = getTempCoordCR(i);

            // 1D cell coordinate on this MPI rank's domain
            int index = grid.get1DIndex(coord_x, coord_y, coord_z);

            // Two events for this cell - going above and then going below liquidus temperature
            int idx_this_layer = 2 * (i - start_range);
            liquidus_time_step_read[idx_this_layer] = Kokkos::round(t_melting / deltat) + 1;
            cell_read[idx_this_layer] = index;
            cooling_rate_read[idx_this_layer] = -1.0; // heating

            liquidus_time_step_read[idx_this_layer + 1] = Kokkos::round(t_liquidus / deltat) + 1;
            // Ensure that cell doesn't go above and below liquidus on the same time step
            if (liquidus_time_step_read[idx_this_layer + 1] == liquidus_time_step_read[idx_this_layer])
                liquidus_time_step_read[idx_this_layer + 1] = liquidus_time_step_read[idx_this_layer + 1] + 1;
            cell_read[idx_this_layer + 1] = index;
            cooling_rate_read[idx_this_layer + 1] = std::abs(cooling_rate_cell) * deltat; // cooling

            // Estimate of the time step where the last possible solidification is expected to occur
            double solidus_time = t_liquidus + freezing_range / cooling_rate_cell;
            if (solidus_time > largest_time)
                largest_time = solidus_time;
        }
        MPI_Allreduce(&largest_time, &largest_time_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (id == 0)
            std::cout << " Layer " << layernumber
                      << " time step where all cells are cooled below solidus for the final time is "
                      << Kokkos::round((largest_time_global) / deltat) << " (or " << largest_time_global << " seconds)"
                      << std::endl;
        if (id == 0)
            std::cout << "Layer " << layernumber << " temperatures read" << std::endl;

        // Reorder liquidus time events based on the times at which they occur
        initOrderedTimeTempHistory(liquidus_time_step_read, cell_read, cooling_rate_read,
                                   num_liquidus_times_this_layer);

        // LayerID data on device for this layer - assign the current LayerID if this cell appears at least once on the
        // list of melt/solidification events
        auto _layer_id = layer_id;
        auto _liquidus_cell_list = liquidus_cell_list;
        auto policy = Kokkos::RangePolicy<execution_space>(0, num_liquidus_times_this_layer);
        Kokkos::parallel_for(
            "LayerIDInit", policy, KOKKOS_LAMBDA(const int &event_num) {
                const int index = _liquidus_cell_list(event_num);
                _layer_id(index) = layernumber;
            });
        Kokkos::fence();
        MPI_Barrier(MPI_COMM_WORLD);

        // For the optionally stored views, reset solidification event counter and get the subview associated with the
        // undercooling field for the next layer
        if (store_undercooling_current)
            undercooling_current = getLayerSubview(undercooling_current_all_layers, grid.layer_range);
        if (store_undercooling_solidification_start)
            undercooling_solidification_start =
                getLayerSubview(undercooling_solidification_start_all_layers, grid.layer_range);
        if (store_solidification_event_counter) {
            Kokkos::realloc(solidification_event_counter, grid.domain_size);
            Kokkos::deep_copy(solidification_event_counter, 0);
        }
        if (id == 0) {
            std::cout << "Layer " << layernumber << " temperature field is from Z = " << grid.z_layer_bottom
                      << " through " << grid.nz_layer + grid.z_layer_bottom - 1 << " of the global domain" << std::endl;
            std::cout << "Done with temperature field initialization" << std::endl;
        }
    }

    // Get the subview associated with data from one layer of a multilayer simulation
    template <typename return_view_type>
    return_view_type getLayerSubview(return_view_type input_view, std::pair<int, int> layer_range) {
        return Kokkos::subview(input_view, layer_range);
    }

    // For each Z coordinate, find the smallest undercooling at which solidification started and finished, writing this
    // data to an output file
    view_type_float_2d_host getFrontUndercoolingStartFinish(const int id, const Grid &grid) {
        view_type_float start_solidification_z(Kokkos::ViewAllocateWithoutInitializing("start_solidification_z"),
                                               grid.nz);
        view_type_float end_solidification_z(Kokkos::ViewAllocateWithoutInitializing("end_solidification_z"), grid.nz);
        auto _undercooling_solidification_start = undercooling_solidification_start;
        auto _undercooling_current = undercooling_current;
        auto policy = Kokkos::RangePolicy<execution_space>(0, grid.nz);
        Kokkos::parallel_for(
            "GetMinUndercooling", policy, KOKKOS_LAMBDA(const int &coord_z) {
                float min_start_undercooling = Kokkos::Experimental::finite_max_v<float>;
                float min_end_undercooling = Kokkos::Experimental::finite_max_v<float>;
                for (int coord_x = 0; coord_x < grid.nx; coord_x++) {
                    for (int coord_y = 1; coord_y < grid.ny_local - 1; coord_y++) {
                        int index = grid.get1DIndex(coord_x, coord_y, coord_z);
                        if (_undercooling_solidification_start(index) < min_start_undercooling)
                            min_start_undercooling = _undercooling_solidification_start(index);
                        if (_undercooling_current(index) < min_end_undercooling)
                            min_end_undercooling = _undercooling_current(index);
                    }
                }
                start_solidification_z(coord_z) = min_start_undercooling;
                end_solidification_z(coord_z) = min_end_undercooling;
            });

        // Rank 0 - collect min values from all ranks to get the global start/end solidification undercoolings
        view_type_float start_solidification_z_reduced(
            Kokkos::ViewAllocateWithoutInitializing("start_solidification_z_red"), grid.nz);
        view_type_float end_solidification_z_reduced(
            Kokkos::ViewAllocateWithoutInitializing("end_solidification_z_red"), grid.nz);
        // Could potentially perform these reductions on the 2D view storing both start and end values, but depends on
        // 2D view data layout
        MPI_Reduce(start_solidification_z.data(), start_solidification_z_reduced.data(), grid.nz, MPI_FLOAT, MPI_MIN, 0,
                   MPI_COMM_WORLD);
        MPI_Reduce(end_solidification_z.data(), end_solidification_z_reduced.data(), grid.nz, MPI_FLOAT, MPI_MIN, 0,
                   MPI_COMM_WORLD);

        // Rank 0 - copy to host view
        view_type_float_2d_host start_end_solidification_z_host(
            Kokkos::ViewAllocateWithoutInitializing("start_end_solidification_z_host"), grid.nz, 2);
        if (id == 0) {
            view_type_float_host start_solidification_z_host =
                Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), start_solidification_z_reduced);
            view_type_float_host end_solidification_z_host =
                Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), end_solidification_z_reduced);
            for (int coord_z = 0; coord_z < grid.nz; coord_z++) {
                start_end_solidification_z_host(coord_z, 0) = start_solidification_z_host(coord_z);
                start_end_solidification_z_host(coord_z, 1) = end_solidification_z_host(coord_z);
            }
        }
        return start_end_solidification_z_host;
    }

    // (Optional based on inputs) Set the starting undercooling in the cell for the solidification event that just
    // started. If the cell is still above the liquid, set value to 0
    KOKKOS_INLINE_FUNCTION
    void setStartingUndercooling(const int cycle, const int index) const {
        if (store_undercooling_solidification_start)
            undercooling_solidification_start(index) = Kokkos::fmin(
                0.0, static_cast<float>(cycle - last_time_below_liquidus(index)) * current_cooling_rate(index));
    }

    // (Optional based on inputs) Set the undercooling in the cell at which solidification completed
    KOKKOS_INLINE_FUNCTION
    void setEndingUndercooling(const int cycle, const int index) const {
        if (store_undercooling_current)
            undercooling_current(index) =
                static_cast<float>(cycle - last_time_below_liquidus(index)) * current_cooling_rate(index);
    }

    // Update the solidification event counter for a cell that has not finished the previous solidification event (i.e.,
    // solidification is not complete in this cell as it is not tempsolid or solid type)
    KOKKOS_INLINE_FUNCTION
    void updateSolidificationCounter(const int index) const {
        if (store_solidification_event_counter)
            solidification_event_counter(index)++;
    }

    // Get undercooling of an active cell (init_undercooling is undercooling of domains with a uniform undercooling and
    // no cooling rate)
    KOKKOS_INLINE_FUNCTION
    float getUndercooling(const int cycle, const int index) const {
        return init_undercooling +
               static_cast<float>(cycle - last_time_below_liquidus(index)) * current_cooling_rate(index);
    }

    // Extract the last time step that each cell undergoes melting in the layer. If the cell does not undergo
    // solidification, either print -1 or the specified default value
    template <typename extracted_view_data_type>
    extracted_view_data_type extractTmData(const int domain_size, const int default_val = -1) {
        extracted_view_data_type extracted_data(Kokkos::ViewAllocateWithoutInitializing("extracted_data"), domain_size);
        Kokkos::deep_copy(extracted_data, default_val);
        auto liquidus_cell_list_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), liquidus_cell_list);
        auto cooling_rate_list_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cooling_rate_list);

        for (int n = 0; n < num_liquidus_times_this_layer; n++) {
            if (cooling_rate_list_host(n) < 0) {
                const int index = liquidus_cell_list_host(n);
                extracted_data(index) = liquidus_time_list_host(n);
            }
        }
        return extracted_data;
    }

    // Extract the last time step that each cell cools below the liquidus in the layer. If the cell does not undergo
    // solidification, either print -1 or the specified default value
    template <typename extracted_view_data_type>
    extracted_view_data_type extractTlData(const int domain_size, const int default_val = -1) {
        extracted_view_data_type extracted_data(Kokkos::ViewAllocateWithoutInitializing("extracted_data"), domain_size);
        Kokkos::deep_copy(extracted_data, default_val);
        auto liquidus_cell_list_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), liquidus_cell_list);
        auto cooling_rate_list_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cooling_rate_list);

        for (int n = 0; n < num_liquidus_times_this_layer; n++) {
            if (cooling_rate_list_host(n) >= 0) {
                const int index = liquidus_cell_list_host(n);
                extracted_data(index) = liquidus_time_list_host(n);
            }
        }
        return extracted_data;
    }

    // Extract the cooling rate associated with the final time the cell cools below the liquidus in the layer. If
    // the cell does not undergo solidification, either print -1 or the specified default value
    template <typename extracted_view_data_type>
    extracted_view_data_type extractCrData(const int domain_size, const int default_val = -1) {
        extracted_view_data_type extracted_data(Kokkos::ViewAllocateWithoutInitializing("extracted_data"), domain_size);
        Kokkos::deep_copy(extracted_data, default_val);
        auto liquidus_cell_list_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), liquidus_cell_list);
        auto cooling_rate_list_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cooling_rate_list);

        for (int n = 0; n < num_liquidus_times_this_layer; n++) {
            if (cooling_rate_list_host(n) >= 0) {
                const int index = liquidus_cell_list_host(n);
                extracted_data(index) = cooling_rate_list_host(n);
            }
        }
        return extracted_data;
    }
};

#endif
