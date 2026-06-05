// Copyright Lawrence Livermore National Security, LLC and other ExaCA Project Developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#ifndef GA_REPREGION_HPP
#define GA_REPREGION_HPP

#include "CAconfig.hpp"
#include "CAorientation.hpp"
#include "CAparsefiles.hpp"
#include "CAtypes.hpp"
#include "GAutils.hpp"

#include <Kokkos_Core.hpp>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
// Structure that holds the details of a representative region of the larger microstructure, along which the specific
// analyses to be performed
struct RepresentativeRegion {

    std::string region_name;
    std::string region_type;
    std::string region_orientation;
    std::string units;
    std::string units_dimension;
    std::vector<double> x_bounds_meters = std::vector<double>(2);
    std::vector<double> y_bounds_meters = std::vector<double>(2);
    std::vector<double> z_bounds_meters = std::vector<double>(2);
    double region_size_meters;
    double region_size_microns;
    std::vector<int> x_bounds_cells = std::vector<int>(2);
    std::vector<int> y_bounds_cells = std::vector<int>(2);
    std::vector<int> z_bounds_cells = std::vector<int>(2);
    int region_size_cells;
    std::vector<float> grain_extent_x, grain_extent_y, grain_extent_z;

    // Analysis options for mean quantities
    std::vector<std::string> analysis_options_stats_key = {
        "GrainTypeFractions", // all regions
        "Misorientation", // all regions
        "Size", // all regions
        "BuildTransAspectRatio", // volumes only
        "XExtent", // all regions
        "YExtent", // all regions
        "ZExtent", // all regions
    };
    bool print_stats_yn = false;
    std::vector<bool> analysis_options_stats_yn = std::vector<bool>(7, false);
    // Analysis options for per-grain stats
    std::vector<std::string> analysis_options_per_grain_stats_key = {
        "Misorientation", // all regions
        "Size", // all regions - volume, area, or length
        "XExtent", // all regions
        "YExtent", // all regions
        "ZExtent", // all regions
        "BuildTransAspectRatio", // volumes only
        "IPFZ-RGB" // all regions
    };
    std::vector<bool> analysis_options_per_grain_stats_yn = std::vector<bool>(7, false);
    // Analysis options for per Z coordinate stats
    std::vector<std::string> analysis_options_per_z_stats_key = {
        "MeanGrainArea" // volume only
    };
    bool print_per_grain_stats_yn = false;
    std::vector<bool> analysis_options_per_z_stats_yn = std::vector<bool>(1, false);

    // Analysis options that print separate files
    bool print_exaconstit_yn, print_pole_figure_yn, print_inverse_pole_figure_map_yn;

    // List of grain ID values in the representative region
    std::vector<int> grain_id_vector;
    // List of phase ID values in the representative region
    std::vector<short> phase_id_vector;
    // List of unique grain IDs associated with the region and the number of grains
    int number_of_grains;
    std::vector<int> unique_grain_id_vector;
    // Size (in units of length, area, or volume, depending on region_type) associated with each grain
    std::vector<float> grain_size_vector_microns;

    // Constructor
    template <typename ViewTypeInt3dHost, typename ViewTypeShort3dHost>
    RepresentativeRegion(nlohmann::json analysis_data, std::string region_name, int nx, int ny, int nz, double deltax,
                         std::vector<double> xyz_bounds, ViewTypeInt3dHost grain_id, ViewTypeShort3dHost phase_id) {

        // Data for the specific region of interest
        std::cout << "Parsing data for region " << region_name << std::endl;
        nlohmann::json region_data = analysis_data["Regions"][region_name];
        // Are the bounds given in cells, or in microns? Store both representations
        setUnits(region_data);
        // Obtain the bounds of the region in x, y, and z, in both cells and microns
        convertBounds(region_data, deltax, xyz_bounds, nx, ny, nz);
        // Deduce region type/orientation/size units from the bounds given
        setRegionTypeOrientation();
        setRegionSize(deltax);
        setUnitDimension();

        // Check which overall stats and per grain stats should be printed for this region
        readAnalysisOptionsFromList(region_data, "printAvgStats", analysis_options_stats_key,
                                    analysis_options_stats_yn);
        // print_stats_yn = true if any one of the options are toggled
        int num_analysis_options_stats = analysis_options_stats_yn.size();
        for (int n = 0; n < num_analysis_options_stats; n++) {
            if (analysis_options_stats_yn[n])
                print_stats_yn = true;
        }
        // print_per_grain_stats_yn = true if any one of the options are toggled
        readAnalysisOptionsFromList(region_data, "printPerGrainStats", analysis_options_per_grain_stats_key,
                                    analysis_options_per_grain_stats_yn);
        int num_analysis_options_per_grain_stats = analysis_options_per_grain_stats_yn.size();
        for (int n = 0; n < num_analysis_options_per_grain_stats; n++) {
            if (analysis_options_per_grain_stats_yn[n])
                print_per_grain_stats_yn = true;
        }
        std::cout << "Read analysis options" << std::endl;

        // printPerZCoordinateStats stats are for volumes only - currently, MeanGrainArea is the only option
        if (region_type == "volume")
            readAnalysisOptionsFromList(region_data, "printPerZCoordinateStats", analysis_options_per_z_stats_key,
                                        analysis_options_per_z_stats_yn);
        // Check other Y/N options (false by default or if not an allowed option for the region type)
        readSeparateFileAnalysisOptions(region_data);

        // Lists of grain ID and phase ID values in the representative region
        grain_id_vector = getIDVector<ViewTypeInt3dHost, int>(grain_id);
        phase_id_vector = getIDVector<ViewTypeShort3dHost, short>(phase_id);

        // List of unique grain IDs associated with the region, also initialize the number of grains
        unique_grain_id_vector = getUniqueGrains(grain_id_vector);
        number_of_grains = unique_grain_id_vector.size();
        // Size (in units of length, area, or volume, depending on region_type) associated with each grain
        grain_size_vector_microns = getGrainSizeVector(deltax);
        std::cout << "Loaded analysis options for region " << region_name << std::endl;
    }

    void readSeparateFileAnalysisOptions(nlohmann::json region_data) {

        // ExaConstit print/area series prints for volumes only
        if (region_type == "volume") {
            if (region_data.contains("printExaConstit")) {
                print_exaconstit_yn = region_data["printExaConstit"];
            }
            else
                print_exaconstit_yn = false;
        }
        else
            print_exaconstit_yn = false;

        // Pole figure data print for volumes and areas only
        if ((region_type == "volume") || (region_type == "area")) {
            if (region_data.contains("printPoleFigureData")) {
                print_pole_figure_yn = region_data["printPoleFigureData"];
            }
            else
                print_pole_figure_yn = false;
        }
        else
            print_pole_figure_yn = false;

        // IPF map data for areas only
        if (region_type == "area") {
            if (region_data.contains("printInversePoleFigureData")) {
                print_inverse_pole_figure_map_yn = region_data["printInversePoleFigureData"];
            }
            else
                print_inverse_pole_figure_map_yn = false;
        }
        else
            print_inverse_pole_figure_map_yn = false;
    }

    void setRegionTypeOrientation() {
        bool flat_x = (x_bounds_cells[0] == x_bounds_cells[1]);
        bool flat_y = (y_bounds_cells[0] == y_bounds_cells[1]);
        bool flat_z = (z_bounds_cells[0] == z_bounds_cells[1]);
        if ((!flat_x) && (!flat_y) && (!flat_z)) {
            region_type = "volume";
            region_orientation = "XYZ";
        }
        else if ((flat_x) && (flat_y) && (flat_z))
            throw std::runtime_error("Error: Cannot analyze a single cell");
        else {
            // Either an area or a length
            if ((flat_x) && (flat_y)) {
                region_type = "length";
                region_orientation = "Z";
            }
            else if ((flat_x) && (flat_z)) {
                region_type = "length";
                region_orientation = "Y";
            }
            else if ((flat_y) && (flat_z)) {
                region_type = "length";
                region_orientation = "X";
            }
            else {
                region_type = "area";
                if (flat_x)
                    region_orientation = "YZ";
                else if (flat_y)
                    region_orientation = "XZ";
                else if (flat_z)
                    region_orientation = "XY";
            }
        }
    }

    void readAnalysisOptionsFromList(nlohmann::json region_data, std::string analysis_type,
                                     std::vector<std::string> analysis_key, std::vector<bool> &analysis_yn) {
        int num_options_read = region_data[analysis_type].size();
        int num_options_total = analysis_yn.size();
        for (int i = 0; i < num_options_read; i++) {
            std::string value_read = region_data[analysis_type][i];
            for (int j = 0; j < num_options_total; j++) {
                std::string value_key = analysis_key[j];
                if (value_read == analysis_key[j]) {
                    analysis_yn[j] = true;
                }
            }
        }
    }

    void setUnits(nlohmann::json region_data) {
        // Check that the units provided are Meters or Cells, throwing an error otherwise
        units = region_data["units"];
        if (units != "Meters" && units != "meters" && units != "Cells" && units != "cells")
            throw std::runtime_error("Error: Invalid units, should be Meters or Cells");
    }

    // Get the lower and upper bounds for x, y, or z, in microns, converting from whichever format (cells or microns)
    // was present in the analysis input file
    void convertBounds(nlohmann::json region_data, double deltax, std::vector<double> xyz_bounds, int nx, int ny,
                       int nz) {

        std::vector<std::string> bound_directions = {"x", "y", "z"};
        for (int dir = 0; dir < 3; dir++) {

            // Initialize bounds in both Meters and Cells
            std::vector<int> bounds_cells(2);
            std::vector<double> bounds_meters(2);

            // Get the minimum coordinate along the specified axis
            double coordinate_min_val = xyz_bounds[dir];

            // Two bounds or no bounds should be given
            // If no bounds are given, defaults to the size of the domain along that direction
            std::string twobounds = bound_directions[dir] + "Bounds";
            if (!(region_data.contains(twobounds))) {
                bounds_cells[0] = 0;
                bounds_meters[0] = coordinate_min_val;
                if (dir == 0) {
                    bounds_cells[1] = nx - 1;
                    bounds_meters[1] = xyz_bounds[3];
                }
                else if (dir == 1) {
                    bounds_cells[1] = ny - 1;
                    bounds_meters[1] = xyz_bounds[4];
                }
                else if (dir == 2) {
                    bounds_cells[1] = nz - 1;
                    bounds_meters[1] = xyz_bounds[5];
                }
            }
            else {
                std::vector<double> bounds_read(2);
                bounds_read[0] = region_data[twobounds][0];
                bounds_read[1] = region_data[twobounds][1];
                if ((units == "Meters") || (units == "meters")) {
                    // Bounds provided were in meters, convert to cells
                    bounds_meters[0] = bounds_read[0];
                    bounds_meters[1] = bounds_read[1];
                    bounds_cells[0] = std::round((bounds_read[0] - coordinate_min_val) / deltax);
                    bounds_cells[1] = std::round((bounds_read[1] - coordinate_min_val) / deltax);
                }
                else if ((units == "Cells") || (units == "cells")) {
                    // Bounds provided were in cells, convert to microns
                    bounds_cells[0] = static_cast<int>(bounds_read[0]);
                    bounds_cells[1] = static_cast<int>(bounds_read[1]);
                    bounds_meters[0] = coordinate_min_val + deltax * bounds_read[0];
                    bounds_meters[1] = coordinate_min_val + deltax * bounds_read[1];
                }
            }

            // Store bounds for the given direction
            for (int i = 0; i < 2; i++) {
                if (dir == 0) {
                    x_bounds_meters[i] = bounds_meters[i];
                    x_bounds_cells[i] = bounds_cells[i];
                }
                else if (dir == 1) {
                    y_bounds_meters[i] = bounds_meters[i];
                    y_bounds_cells[i] = bounds_cells[i];
                }
                else if (dir == 2) {
                    z_bounds_meters[i] = bounds_meters[i];
                    z_bounds_cells[i] = bounds_cells[i];
                }
            }
        }
    }

    // Get the size of the region in cells, and in meters/microns (either 1D, 2D, or 3D units depending on region type)
    void setRegionSize(double deltax) {
        region_size_cells = (x_bounds_cells[1] - x_bounds_cells[0] + 1) * (y_bounds_cells[1] - y_bounds_cells[0] + 1) *
                            (z_bounds_cells[1] - z_bounds_cells[0] + 1);
        if (region_type == "length") {
            region_size_meters = region_size_cells * deltax;
            region_size_microns = region_size_meters * Kokkos::pow(10, 6);
        }
        else if (region_type == "area") {
            region_size_meters = region_size_cells * Kokkos::pow(deltax, 2);
            region_size_microns = region_size_meters * Kokkos::pow(10, 12);
        }
        else if (region_type == "volume") {
            region_size_meters = region_size_cells * Kokkos::pow(deltax, 3);
            region_size_microns = region_size_meters * Kokkos::pow(10, 18);
        }
        else
            throw std::runtime_error("Error: Invalid region type in GetRegionSize during region construction");
    }

    // Set the appropriate units for the region based on the dimensions (in microns)
    void setUnitDimension() {
        if (region_type == "length")
            units_dimension = "microns";
        else if (region_type == "area")
            units_dimension = "square microns";
        else if (region_type == "volume")
            units_dimension = "cubic microns";
        else
            throw std::runtime_error("Error: Invalid region type in setRegionSize during region construction");
    }

    // Helper functions used to analyze a region of microstructure data
    // Subroutines starting with "get" return the data specified
    // Subroutines starting with "calc" calculate the quantity specified but do not return it

    // Get the list of Grain/Phase IDs associated with the representative region
    template <typename ViewType3dHost, typename value_type>
    std::vector<value_type> getIDVector(ViewType3dHost id_field) {

        std::vector<value_type> id_vector(region_size_cells);
        int count = 0;
        for (int k = z_bounds_cells[0]; k <= z_bounds_cells[1]; k++) {
            for (int i = x_bounds_cells[0]; i <= x_bounds_cells[1]; i++) {
                for (int j = y_bounds_cells[0]; j <= y_bounds_cells[1]; j++) {
                    id_vector[count] = id_field(k, i, j);
                    count++;
                }
            }
        }
        return id_vector;
    }

    // Given an input vector of integer Grain ID values, return an output vector consisting of the unique Grain ID
    // values, sorted from lowest to highest. Store the number of grains
    std::vector<int> getUniqueGrains(std::vector<int> _grain_id_vector) {
        std::vector<int> _unique_grain_id_vector = _grain_id_vector;
        std::sort(_unique_grain_id_vector.begin(), _unique_grain_id_vector.end());
        std::vector<int>::iterator it;
        it = std::unique(_unique_grain_id_vector.begin(), _unique_grain_id_vector.end());
        _unique_grain_id_vector.resize(std::distance(_unique_grain_id_vector.begin(), it));
        return _unique_grain_id_vector;
    }

    // Given an input vector of integer Grain ID values "grain_id_vector", and an input vector of the
    // unique Grain ID values "unique_grain_id_vector" (of size "number_of_grains"), return a third vector
    // "grain_size_vector" listing the size of each of the "number_of_grains" grains (scaled by the cell size deltax and
    // depending on whether the region is 1D, 2D, or 3D)
    std::vector<float> getGrainSizeVector(double deltax) {

        std::vector<float> grain_size_vector(number_of_grains);
        double conv = convertToMicrons(deltax, region_type);
        for (int n = 0; n < number_of_grains; n++) {
            int grain_size_cells =
                std::count(grain_id_vector.begin(), grain_id_vector.end(), unique_grain_id_vector[n]);
            // convert to either microns, square microns, or cubic microns
            grain_size_vector[n] = static_cast<float>(conv) * static_cast<float>(grain_size_cells);
        }
        return grain_size_vector;
    }

    // Given the 3D grain structure "grain_id", determine the extent in the direction specified of each of the
    // "number_of_grains" unique grains from the volume bounded by [x_low, x_high], [y_low, y_high], [z_low, z_high].
    // Extent is calculated in microns
    template <typename ViewTypeInt3dHost>
    void calcGrainExtent(std::vector<float> &grain_extent, ViewTypeInt3dHost grain_id, std::string direction,
                         double deltax) {

        for (int n = 0; n < number_of_grains; n++) {
            int this_grain_id = unique_grain_id_vector[n];
            int this_grain_size = std::round(grain_size_vector_microns[n] * convertToCells(deltax, region_type));
            std::vector<int> grain_coordinate(this_grain_size);
            int count = 0;
            for (int k = z_bounds_cells[0]; k <= z_bounds_cells[1]; k++) {
                for (int i = x_bounds_cells[0]; i <= x_bounds_cells[1]; i++) {
                    for (int j = y_bounds_cells[0]; j <= y_bounds_cells[1]; j++) {
                        if (grain_id(k, i, j) == this_grain_id) {
                            if (direction == "X")
                                grain_coordinate[count] = i;
                            else if (direction == "Y")
                                grain_coordinate[count] = j;
                            else if (direction == "Z")
                                grain_coordinate[count] = k;
                            count++;
                        }
                    }
                }
            }
            int min_coord = *std::min_element(grain_coordinate.begin(), grain_coordinate.end());
            int max_coord = *std::max_element(grain_coordinate.begin(), grain_coordinate.end());
            grain_extent[n] = (max_coord - min_coord + 1) * convertToMicrons(deltax, "length");
        }
    }

    template <typename ViewTypeInt3dHost>
    void calcNecessaryGrainExtents(ViewTypeInt3dHost grain_id, double deltax) {
        // For the analysis options:
        // If StatsYN[3] or PerGrainStatsYN[5] is toggled, all grain extents are needed. If StatsYN[4] or
        // PerGrainStatsYN[2] is toggled, X extents are needed. If StatsYN[5] or PerGrainStatsYN[3] is toggled, Y
        // extents are needed. If StatsYN[6] or PerGrainStatsYN[4] is toggled, Z extents are needed
        bool calc_extent_x = false;
        bool calc_extent_y = false;
        bool calc_extent_z = false;
        if ((analysis_options_stats_yn[3]) || (analysis_options_stats_yn[4])) {
            calc_extent_x = true;
            calc_extent_y = true;
            calc_extent_z = true;
        }
        else {
            if ((analysis_options_stats_yn[4]) || (analysis_options_per_grain_stats_yn[2]))
                calc_extent_x = true;
            if ((analysis_options_stats_yn[5]) || (analysis_options_per_grain_stats_yn[3]))
                calc_extent_y = true;
            if ((analysis_options_stats_yn[6]) || (analysis_options_per_grain_stats_yn[4]))
                calc_extent_z = true;
        }
        if (calc_extent_x) {
            grain_extent_x.resize(number_of_grains);
            calcGrainExtent(grain_extent_x, grain_id, "X", deltax);
        }
        if (calc_extent_y) {
            grain_extent_y.resize(number_of_grains);
            calcGrainExtent(grain_extent_y, grain_id, "Y", deltax);
        }
        if (calc_extent_z) {
            grain_extent_z.resize(number_of_grains);
            calcGrainExtent(grain_extent_z, grain_id, "Z", deltax);
        }
    }

    // Calculate <100> misorientation relative to the specified cardinal direction for each grain and store in the
    // vector
    template <typename MemorySpace>
    std::vector<float> getCrystalMisorientation(std::string direction, Orientation<MemorySpace> &orientation) {

        std::vector<float> crystal_misorientation_vector(number_of_grains);
        int direction_int;
        if (direction == "X")
            direction_int = 0;
        else if (direction == "Y")
            direction_int = 1;
        else if (direction == "Z")
            direction_int = 2;
        else
            throw std::runtime_error(
                "Error: invalid direction specified in calcCrystalMisorientation: should be X, Y, or Z");
        auto crystal_misorientation = orientation.misorientationCalc(direction_int);
        for (int n = 0; n < number_of_grains; n++) {
            int my_orientation = getCrystalOrientation(unique_grain_id_vector[n], orientation.n_crystal_orientations);
            // Grain misorientation always corresponds to phase 0, even if original crystallographic orientation during
            // initial solidification as phase 1 was different
            float my_misorientation = crystal_misorientation(my_orientation, 0);
            crystal_misorientation_vector[n] = my_misorientation;
        }
        return crystal_misorientation_vector;
    }

    // Create a histogram of orientations for texture determination, using the grain_id values in the volume bounded by
    // [XMin,XMax], [YMin,YMax], [ZMin,ZMax] and excluding and cells that did not undergo melting (layer_id = -1)
    template <typename ViewTypeInt3dHost, typename ViewTypeShort3dHost>
    auto getOrientationHistogram(int n_crystal_orientations, ViewTypeInt3dHost grain_id, ViewTypeShort3dHost layer_id,
                                 ViewTypeShort3dHost phase_id, const int num_phases, bool found_layer_id) {

        // Init histogram values to zero
        if (!found_layer_id)
            std::cout
                << "Note: as LayerID was not given, the grain histogram data may include regions that did not undergo "
                   "melting and solidification as they cannot be differentiated from the regions that were simulated"
                << std::endl;
        using int_type = typename ViewTypeInt3dHost::value_type;
        Kokkos::View<int_type **, Kokkos::HostSpace> go_histogram("go_histogram", n_crystal_orientations, num_phases);
        for (int phase_num = 0; phase_num < num_phases; phase_num++) {
            for (int k = z_bounds_cells[0]; k <= z_bounds_cells[1]; k++) {
                for (int j = y_bounds_cells[0]; j <= y_bounds_cells[1]; j++) {
                    for (int i = x_bounds_cells[0]; i <= x_bounds_cells[1]; i++) {
                        if ((grain_id(k, i, j) != 0) && (layer_id(k, i, j) != -1)) {
                            short my_phase = phase_id(k, i, j);
                            int go_val = getCrystalOrientation(grain_id(k, i, j), n_crystal_orientations);
                            go_histogram(go_val, my_phase)++;
                        }
                    }
                }
            }
        }
        return go_histogram;
    }

    //*****************************************************************************/
    // "print" routines print average quantities to the screen (and to the file stream specified by QoIs)
    // "write" routines write data to specific files
    //*****************************************************************************/
    // Print information about a representative area or volume to the console/qois file
    void printAnalysisHeader(std::ofstream &qois) {

        std::string temp;
        temp = "Stats for " + region_type + " " + region_name + "\n";
        if (region_orientation == "XY") {
            temp += "The representative area is located at Z = " + std::to_string(z_bounds_meters[0]) + " m\n";
            temp += "(in CA units, Z = " + std::to_string(z_bounds_cells[0]) + ")\n";
            temp += "The representative area is bounded by the region spanning X = [" +
                    std::to_string(x_bounds_meters[0]) + "," + std::to_string(x_bounds_meters[1]) + "], Y = [ " +
                    std::to_string(y_bounds_meters[0]) + "," + std::to_string(y_bounds_meters[1]) + "] m\n";
            temp += "(in CA units X = [" + std::to_string(x_bounds_cells[0]) + "," + std::to_string(x_bounds_cells[1]) +
                    "], Y = [" + std::to_string(y_bounds_cells[0]) + "," + std::to_string(y_bounds_cells[1]) + "])\n";
        }
        else if (region_orientation == "XZ") {
            temp += "The representative area is located at Y = " + std::to_string(y_bounds_meters[0]) + " m\n";
            temp += "(in CA units, Y = " + std::to_string(y_bounds_cells[0]) + ")\n";
            temp += "The representative area is bounded by the region spanning X = [" +
                    std::to_string(x_bounds_meters[0]) + "," + std::to_string(x_bounds_meters[1]) + "], Z = [" +
                    std::to_string(z_bounds_meters[0]) + "," + std::to_string(z_bounds_meters[1]) + "] m\n";
            temp += "(in CA units X = [" + std::to_string(x_bounds_cells[0]) + "," + std::to_string(x_bounds_cells[1]) +
                    "], Z = [" + std::to_string(z_bounds_cells[0]) + "," + std::to_string(z_bounds_cells[1]) + "])\n";
        }
        else if (region_orientation == "YZ") {
            temp += "The representative area is located at X = " + std::to_string(x_bounds_meters[0]) + " m\n";
            temp += "(in CA units, X = " + std::to_string(x_bounds_cells[0]) + ")\n";
            temp += "The representative area is bounded by the region spanning Y = [" +
                    std::to_string(y_bounds_meters[0]) + "," + std::to_string(y_bounds_meters[1]) + "], Z = [" +
                    std::to_string(z_bounds_meters[0]) + "," + std::to_string(z_bounds_meters[1]) + "] m\n";
            temp += "(in CA units Y = [" + std::to_string(y_bounds_cells[0]) + "," + std::to_string(y_bounds_cells[1]) +
                    "], Z = [" + std::to_string(z_bounds_cells[0]) + "," + std::to_string(z_bounds_cells[1]) + "])\n";
        }
        else if (region_orientation == "XYZ") {
            temp += "The representative volume specified is bounded by X = [" + std::to_string(x_bounds_meters[0]) +
                    "," + std::to_string(x_bounds_meters[1]) + "], Y = [" + std::to_string(y_bounds_meters[0]) + "," +
                    std::to_string(y_bounds_meters[1]) + "], and Z = [" + std::to_string(z_bounds_meters[0]) + "," +
                    std::to_string(z_bounds_meters[1]) + "] m\n";
            temp += "The representative volume specified is bounded by cells spanning X = [" +
                    std::to_string(x_bounds_cells[0]) + "," + std::to_string(x_bounds_cells[1]) + "], Y = [" +
                    std::to_string(y_bounds_cells[0]) + "," + std::to_string(y_bounds_cells[1]) + "], and Z = [" +
                    std::to_string(z_bounds_cells[0]) + "," + std::to_string(z_bounds_cells[1]) + "]\n";
        }
        else
            throw std::runtime_error("Error: invalid region type");
        dualPrint(temp, std::cout, qois);
    }

    // Print number of cells in the representative region that did not undergo melting, fraction consisting of nucleated
    // grains to the console/qois file
    template <typename ViewTypeInt3dHost, typename ViewTypeShort3dHost>
    void printGrainTypeFractions(std::ofstream &qois, ViewTypeInt3dHost grain_id, ViewTypeShort3dHost layer_id,
                                 ViewTypeShort3dHost phase_id, const int num_phases, bool found_layer_id) {

        std::string temp;
        temp = "-- The representative region consists of " + std::to_string(region_size_cells) + " cells\n";
        if (found_layer_id) {
            // Stats for only the cells that underwent melting
            int number_of_unmelted_cells = 0;
            int number_of_nucleated_grain_cells_melted = 0;
            int number_of_primary_phase_cells_melted = 0;
            for (int k = z_bounds_cells[0]; k <= z_bounds_cells[1]; k++) {
                for (int i = x_bounds_cells[0]; i <= x_bounds_cells[1]; i++) {
                    for (int j = y_bounds_cells[0]; j <= y_bounds_cells[1]; j++) {
                        if (layer_id(k, i, j) == -1)
                            number_of_unmelted_cells++;
                        else {
                            if (grain_id(k, i, j) < 0)
                                number_of_nucleated_grain_cells_melted++;
                            if (phase_id(k, i, j) == 0)
                                number_of_primary_phase_cells_melted++;
                        }
                    }
                }
            }
            temp += "-- The number of cells in the region that did not undergo melting is " +
                    std::to_string(number_of_unmelted_cells) + "\n";
            // Stats for only the portion of the region to undergo melting and solidification
            float vol_fract_nuc_grains_melted =
                divideCast<float>(number_of_nucleated_grain_cells_melted, region_size_cells - number_of_unmelted_cells);
            temp += "-- Considering only the portion of the region to undergo melting/solidification, the volume "
                    "fraction of the region consisting of nucleated grains is " +
                    std::to_string(vol_fract_nuc_grains_melted) + "\n";
            float vol_fract_primary_phase_melted =
                divideCast<float>(number_of_primary_phase_cells_melted, region_size_cells - number_of_unmelted_cells);
            if (num_phases > 1)
                temp += "-- Considering only the portion of the region to undergo melting/solidification, the volume "
                        "fraction of the region that solidified as the primary phase is " +
                        std::to_string(vol_fract_primary_phase_melted) + "\n";
        }
        else
            temp += "-- LayerID was not given in this microstructure dataset, the number of cells in the region that "
                    "did not undergo melting cannot be extracted. Grain volume fractions for the portion of the region "
                    "to undergo melting and resolidification will not be printed\n";
        // Stats for the entire region, regardless of whether the cell melted and solidified during one of the layers
        int number_of_nucleated_grain_cells = 0;
        int number_of_primary_phase_cells = 0;
        for (int k = z_bounds_cells[0]; k <= z_bounds_cells[1]; k++) {
            for (int i = x_bounds_cells[0]; i <= x_bounds_cells[1]; i++) {
                for (int j = y_bounds_cells[0]; j <= y_bounds_cells[1]; j++) {
                    if (grain_id(k, i, j) < 0)
                        number_of_nucleated_grain_cells++;
                    if (phase_id(k, i, j) == 0)
                        number_of_primary_phase_cells++;
                }
            }
        }
        float vol_fract_nuc_grains = divideCast<float>(number_of_nucleated_grain_cells, region_size_cells);
        temp += "-- The volume fraction of the region consisting of nucleated grains is " +
                std::to_string(vol_fract_nuc_grains) + "\n";
        float vol_fract_primary_phase = divideCast<float>(number_of_primary_phase_cells, region_size_cells);
        if (num_phases > 1)
            temp += "-- The volume fraction of the region consisting of the primary phase is " +
                    std::to_string(vol_fract_primary_phase) + "\n";
        dualPrint(temp, std::cout, qois);
    }

    // Print average misorientation for the region relative to each direction
    void printMeanMisorientations(std::ofstream &qois, std::vector<float> crystal_misorientation_x_vector,
                                  std::vector<float> crystal_misorientation_y_vector,
                                  std::vector<float> crystal_misorientation_z_vector) {

        std::vector<std::string> misorientation_direction_labels = {"misorientationX", "misorientationY",
                                                                    "misorientationZ"};
        std::vector<std::string> misorientation_direction_labels_short = {"+X", "+Y", "+Z"};
        float crystal_misorientation_sum_x = 0.0;
        float crystal_misorientation_sum_y = 0.0;
        float crystal_misorientation_sum_z = 0.0;
        for (int n = 0; n < number_of_grains; n++) {
            crystal_misorientation_sum_x += crystal_misorientation_x_vector[n] * grain_size_vector_microns[n];
            crystal_misorientation_sum_y += crystal_misorientation_y_vector[n] * grain_size_vector_microns[n];
            crystal_misorientation_sum_z += crystal_misorientation_z_vector[n] * grain_size_vector_microns[n];
        }
        float avg_misorientation_x = divideCast<float>(crystal_misorientation_sum_x, region_size_microns);
        float avg_misorientation_y = divideCast<float>(crystal_misorientation_sum_y, region_size_microns);
        float avg_misorientation_z = divideCast<float>(crystal_misorientation_sum_z, region_size_microns);
        std::string temp;
        temp = "-- Average misorientation (weighted by size) for grains relative to the X direction (in degrees): " +
               std::to_string(avg_misorientation_x) + "\n";
        temp += "-- Average misorientation (weighted by size) for grains relative to the Y direction (in degrees): " +
                std::to_string(avg_misorientation_y) + "\n";
        temp += "-- Average misorientation (weighted by size) for grains relative to the Z direction (in degrees): " +
                std::to_string(avg_misorientation_z) + "\n";
        dualPrint(temp, std::cout, qois);
    }

    // Print the average grain size and the number of grains in the region
    void printMeanSize(std::ofstream &qois, const double deltax, const int num_phases = 1,
                       const int grain_size_thresh = 6) {

        float avg_size_per_grain = divideCast<float>(region_size_microns, number_of_grains);
        std::string temp = "-- There are " + std::to_string(number_of_grains) + " grains in this " + region_type +
                           " , and the mean grain " + region_type + " is " + std::to_string(avg_size_per_grain) + " " +
                           units_dimension + "\n";
        // Also get average size per grain for regions that exclusively solidified as the second phase
        if (num_phases > 1) {
            int second_phase_cells = 0;
            std::vector<int> grain_id_second_phase(region_size_cells);
            for (int n = 0; n < region_size_cells; n++) {
                if (phase_id_vector[n] == 1) {
                    grain_id_second_phase[second_phase_cells] = grain_id_vector[n];
                    second_phase_cells++;
                }
            }
            grain_id_second_phase.resize(second_phase_cells);
            std::vector<int> unique_grain_id_vector_second_phase = getUniqueGrains(grain_id_second_phase);
            const int num_second_phase_grains = unique_grain_id_vector_second_phase.size();
            // Filter out small grains
            int num_second_phase_grains_above_thresh = 0;
            double conv = convertToMicrons(deltax, region_type);
            for (int n = 0; n < num_second_phase_grains; n++) {
                int grain_size_cells = std::count(grain_id_second_phase.begin(), grain_id_second_phase.end(),
                                                  unique_grain_id_vector_second_phase[n]);
                if (grain_size_cells > grain_size_thresh) {
                    // convert to either microns, square microns, or cubic microns (default threshold of 6 to match MTEX
                    // analysis scripts)
                    num_second_phase_grains_above_thresh++;
                }
            }
            float second_phase_area = static_cast<float>(conv) * static_cast<float>(second_phase_cells);
            // Add area of removed grains into the remaining grains
            float avg_size_per_second_phase_grain =
                divideCast<float>(second_phase_area, num_second_phase_grains_above_thresh);
            temp += "-- For the region that solidified as the second phase, there are " +
                    std::to_string(num_second_phase_grains_above_thresh) + " grains in this " + region_type +
                    " , and the mean grain " + region_type + " is " + std::to_string(avg_size_per_second_phase_grain) +
                    " " + units_dimension + "\n";
        }
        dualPrint(temp, std::cout, qois);
    }

    // Print average grain extent in the specified direction
    void printMeanExtent(std::ofstream &qois, std::string direction) {

        float grain_extent_sum = 0.0;
        std::vector<float> grain_extent_microns;
        if (direction == "X")
            grain_extent_microns = grain_extent_x;
        else if (direction == "Y")
            grain_extent_microns = grain_extent_y;
        else if (direction == "Z")
            grain_extent_microns = grain_extent_z;
        for (int n = 0; n < number_of_grains; n++)
            grain_extent_sum += grain_extent_microns[n];
        float avg_grain_extent = divideCast<float>(grain_extent_sum, number_of_grains);
        std::string temp = "-- The mean grain extent in the " + direction + " direction is " +
                           std::to_string(avg_grain_extent) + " microns\n";
        dualPrint(temp, std::cout, qois);
    }

    // Print average aspect ratio in the build to the average of the transverse directions
    void printMeanBuildTransAspectRatio(std::ofstream &qois) {

        std::vector<float> grain_aspect_ratios(number_of_grains);
        float ar_sum = 0.0;
        float vol_wt_ar_sum = 0.0;
        for (int n = 0; n < number_of_grains; n++) {
            float ar_xz = grain_extent_z[n] / grain_extent_x[n];
            float ar_yz = grain_extent_z[n] / grain_extent_y[n];
            grain_aspect_ratios[n] = 0.5 * (ar_xz + ar_yz);
            ar_sum += grain_aspect_ratios[n];
            vol_wt_ar_sum += grain_aspect_ratios[n] * grain_size_vector_microns[n];
        }
        double representative_region_size_microns = region_size_meters * Kokkos::pow(10, 18);
        std::string temp;
        temp = "-- The mean grain aspect ratio (Z direction to transverse) is " +
               std::to_string(divideCast<float>(ar_sum, number_of_grains)) + "\n";
        temp += "-- The mean volume-weighted grain aspect ratio (Z direction to transverse) is " +
                std::to_string(divideCast<float>(vol_wt_ar_sum, representative_region_size_microns)) + "\n";
        dualPrint(temp, std::cout, qois);
    }

    // Write mean grain areas as a function of build height to file(s)
    template <typename ViewTypeInt3dHost>
    void writeAreaSeries(std::string base_filename, double deltax, ViewTypeInt3dHost grain_id) {

        bool print_mean_areas = analysis_options_per_z_stats_yn[0];
        if (!print_mean_areas)
            return;

        std::string fname = base_filename + "_GrainAreas.csv";
        std::ofstream grainplot;
        std::cout << "Printing file " << fname << " of grain area values (in square microns) for all Z coordinates"
                  << std::endl;
        grainplot.open(fname);
        grainplot << "Zcoordinate(µm),MeanArea(µm2)" << std::endl;

        int layer_area = (x_bounds_cells[1] - x_bounds_cells[0] + 1) * (y_bounds_cells[1] - y_bounds_cells[0] + 1);
        for (int k = z_bounds_cells[0]; k <= z_bounds_cells[1]; k++) {
            // Z coordinate in microns
            const double z_coordinate_microns =
                z_bounds_meters[0] * Kokkos::pow(10, 6) + k * convertToMicrons(deltax, "length");
            // All grain_id values in the representative area at Z = k
            std::vector<int> grain_id_vector_area(region_size_cells);
            int count = 0;
            for (int i = x_bounds_cells[0]; i <= x_bounds_cells[1]; i++) {
                for (int j = y_bounds_cells[0]; j <= y_bounds_cells[1]; j++) {
                    grain_id_vector_area[count] = grain_id(k, i, j);
                    count++;
                }
            }
            // Get the number of grains in the representative area and a list of the unique grain IDs
            std::vector<int> unique_grain_id_vector_area = grain_id_vector_area;
            std::sort(unique_grain_id_vector_area.begin(), unique_grain_id_vector_area.end());
            std::vector<int>::iterator it;
            it = std::unique(unique_grain_id_vector_area.begin(), unique_grain_id_vector_area.end());
            unique_grain_id_vector_area.resize(std::distance(unique_grain_id_vector_area.begin(), it));
            int number_of_grains_area = unique_grain_id_vector_area.size();
            float mean_grain_area_this_layer = divideCast<float>(layer_area, number_of_grains_area);
            if (print_mean_areas)
                grainplot << z_coordinate_microns << ","
                          << mean_grain_area_this_layer * convertToMicrons(deltax, "area") << std::endl;
        }
        grainplot.close();
    }

    // Write pole figure data for this region to a file to be read by MTEX. As the final phase is always 0, the
    // orientations always come from phase 0 regardless of whether the original phase that formed had a different
    // crystallographic orientation
    template <typename MemorySpace, typename ViewTypeInt2DHost>
    void writePoleFigure(std::string base_filename_this_region, Orientation<MemorySpace> &orientation,
                         ViewTypeInt2DHost go_histogram) {

        // Using new format, write pole figure data to "Filename"
        std::string filename = base_filename_this_region + "_PoleFigureData.txt";
        std::ofstream grainplot_pf;
        grainplot_pf.open(filename);
        grainplot_pf << "% MTEX ODF" << std::endl;
        grainplot_pf << "% crystal symmetry: \"m-3m\"" << std::endl;
        grainplot_pf << "% specimen symmetry: \"43\"" << std::endl;
        grainplot_pf << "% phi1    Phi     phi2    value" << std::endl;
        grainplot_pf << std::fixed << std::setprecision(6);
        for (int i = 0; i < orientation.n_crystal_orientations; i++) {
            grainplot_pf << orientation.grain_bunge_euler_host(3 * i, 0) << " "
                         << orientation.grain_bunge_euler_host(3 * i + 1, 0) << " "
                         << orientation.grain_bunge_euler_host(3 * i + 2, 0) << " "
                         << static_cast<float>(go_histogram(i, 0)) << std::endl;
        }
        grainplot_pf.close();
    }

    // From the grain extents in x, y, and z, calculate the aspect ratio for each grain in the build to the average of
    // the transverse directions
    void calcBuildTransAspectRatio(std::vector<float> &build_trans_aspect_ratio) {

        for (int n = 0; n < number_of_grains; n++) {
            float ar_xz = grain_extent_z[n] / grain_extent_x[n];
            float ar_yz = grain_extent_z[n] / grain_extent_y[n];
            build_trans_aspect_ratio[n] = 0.5 * (ar_xz + ar_yz);
        }
    }

    // Determine the R, G, or B value for this grain orientation for the IPF-Z inverse pole figure colormap (Color = 0
    // for R, 1 for G, 2 for B)
    template <typename MemorySpace>
    std::vector<float> getIPFZColor(int color, Orientation<MemorySpace> &orientation) {
        std::vector<float> ipfz_color(number_of_grains);
        for (int n = 0; n < number_of_grains; n++) {
            int my_orientation = getCrystalOrientation(unique_grain_id_vector[n], orientation.n_crystal_orientations);
            ipfz_color[n] = orientation.grain_rgb_ipfz_host(3 * my_orientation + color, 0);
        }
        return ipfz_color;
    }

    // Print data to be read by MTEX to plot the cross-section using the inverse pole figure colormap
    template <typename ViewTypeInt3dHost, typename ViewTypeShort3dHost, typename MemorySpace>
    void writeIPFColoredCrossSection(std::string base_filename_this_region, ViewTypeInt3dHost grain_id,
                                     ViewTypeShort3dHost phase_id, Orientation<MemorySpace> &orientation,
                                     double deltax) {

        // What portion of the area is to be printed?
        int index_1_low, index_1_high, index_2_low, index_2_high, cross_section_out_of_plane_location;
        if (region_orientation == "XY") {
            index_1_low = x_bounds_cells[0];
            index_1_high = x_bounds_cells[1];
            index_2_low = y_bounds_cells[0];
            index_2_high = y_bounds_cells[1];
            cross_section_out_of_plane_location = z_bounds_cells[0];
        }
        else if (region_orientation == "XZ") {
            index_1_low = x_bounds_cells[0];
            index_1_high = x_bounds_cells[1];
            index_2_low = z_bounds_cells[0];
            index_2_high = z_bounds_cells[1];
            cross_section_out_of_plane_location = y_bounds_cells[0];
        }
        else if (region_orientation == "YZ") {
            index_1_low = y_bounds_cells[0];
            index_1_high = y_bounds_cells[1];
            index_2_low = z_bounds_cells[0];
            index_2_high = z_bounds_cells[1];
            cross_section_out_of_plane_location = x_bounds_cells[0];
        }
        else
            throw std::runtime_error(
                "Error: Invalid plane type in writeIPFColoredCrossSection: should be XY, XZ, or YZ");

        std::string fname_ipf = base_filename_this_region + "_IPFCrossSectionData.txt";
        std::ofstream grainplot_ipf;
        grainplot_ipf.open(fname_ipf);
        grainplot_ipf << std::fixed << std::setprecision(6);
        for (int index_1 = index_1_low; index_1 < index_1_high; index_1++) {
            for (int index_2 = index_2_low; index_2 < index_2_high; index_2++) {
                // How do index_1, index_2, and out of plane location correspond to grain_id(Z loc, X loc, y_loc)?
                int z_loc, x_loc, y_loc;
                if (region_orientation == "XY") {
                    x_loc = index_1;
                    y_loc = index_2;
                    z_loc = cross_section_out_of_plane_location;
                }
                else if (region_orientation == "YZ") {
                    x_loc = cross_section_out_of_plane_location;
                    y_loc = index_1;
                    z_loc = index_2;
                }
                else if (region_orientation == "XZ") {
                    x_loc = index_1;
                    y_loc = cross_section_out_of_plane_location;
                    z_loc = index_2;
                }
                else
                    throw std::runtime_error("Error: Unknown region_orientation input for WriteIPFColoredCrossSection: "
                                             "should be XY, YZ, or XZ");
                // What orientation does this grain id correspond to? Should be between 0 and NumberOfOrientations-1
                int go_val = (Kokkos::abs(grain_id(z_loc, x_loc, y_loc)) - 1) % orientation.n_crystal_orientations;
                // Unlike the vtk data which was zero-indexed (the two solid phase ids were 0 and 1), the grain
                // structure used by MTEX is phase "1" or "2" - any unindexed points with GOVal = -1 (which are possible
                // from regions that didn't undergo melting) are assigned phase "0"
                if (go_val == -1)
                    grainplot_ipf << "0 0 0 0 " << index_1 * deltax * Kokkos::pow(10, 6) << " "
                                  << index_2 * deltax * Kokkos::pow(10, 6) << std::endl;
                else {
                    // Note that as the second phase orientations were completely transformed to the corresponding first
                    // phase orientation (or the two phases shared a set of orientations), the second phase orientation
                    // data goes unused here.
                    short phase_id_val = phase_id(z_loc, x_loc, y_loc);
                    grainplot_ipf << orientation.grain_bunge_euler_host(3 * go_val, 0) << " "
                                  << orientation.grain_bunge_euler_host(3 * go_val + 1, 0) << " "
                                  << orientation.grain_bunge_euler_host(3 * go_val + 2, 0) << " " << phase_id_val + 1
                                  << " " << index_1 * deltax * Kokkos::pow(10, 6) << " "
                                  << index_2 * deltax * Kokkos::pow(10, 6) << std::endl;
                }
            }
        }
        grainplot_ipf.close();
    }

    // Write data for the specified RVE for ExaConstit
    template <typename ViewTypeInt3dHost>
    void writeExaConstitRVE(std::string base_filename_this_region, double deltax, ViewTypeInt3dHost grain_id) {

        int x_low = x_bounds_cells[0];
        int x_high = x_bounds_cells[1];
        int y_low = y_bounds_cells[0];
        int y_high = y_bounds_cells[1];
        int z_low = z_bounds_cells[0];
        int z_high = z_bounds_cells[1];
        std::string fname = base_filename_this_region + "_ExaConstit.csv";
        std::ofstream grainplot;
        grainplot.open(fname);
        grainplot << "Coordinates are in CA units, 1 cell = " << deltax << " m. Data is cell-centered. Origin at "
                  << x_low << "," << y_low << "," << z_low << " , domain size is " << x_high - x_low + 1 << " by "
                  << y_high - y_low + 1 << " by " << z_high - z_low + 1 << " cells" << std::endl;
        grainplot << "X coord, Y coord, Z coord, Grain ID" << std::endl;
        for (int k = z_low; k <= z_high; k++) {
            for (int j = y_low; j <= y_high; j++) {
                for (int i = x_low; i <= x_high; i++) {
                    grainplot << i << "," << j << "," << k << "," << grain_id(k, i, j) << std::endl;
                }
            }
        }
        grainplot.close();
    }

    // Write a csv file of stats for each grain
    void writePerGrainStats(std::string output_filename, std::vector<float> crystal_misorientation_x_vector,
                            std::vector<float> crystal_misorientation_y_vector,
                            std::vector<float> crystal_misorientation_z_vector,
                            std::vector<float> build_trans_aspect_ratio, std::vector<float> grain_red,
                            std::vector<float> grain_green, std::vector<float> grain_blue) {

        // Which quantities should be printed?
        std::string stats_fname = output_filename + "_grains.csv";
        std::ofstream grain_stats;
        grain_stats.open(stats_fname);
        grain_stats << "grain_id";
        // Which quantities should be printed for each grain?
        if (analysis_options_per_grain_stats_yn[0])
            grain_stats << ",misorientationX,misorientationY,misorientationZ";
        if (analysis_options_per_grain_stats_yn[1])
            grain_stats << "," << region_type;
        if (analysis_options_per_grain_stats_yn[2])
            grain_stats << ",extentX";
        if (analysis_options_per_grain_stats_yn[3])
            grain_stats << ",extentY";
        if (analysis_options_per_grain_stats_yn[4])
            grain_stats << ",extentZ";
        if (analysis_options_per_grain_stats_yn[5])
            grain_stats << ",buildtransAR";
        if (analysis_options_per_grain_stats_yn[6])
            grain_stats << ",IPFZ_r,IPFZ_g,IPFZ_b";
        grain_stats << std::endl;

        // Print the specified quantities to the csv file
        for (int n = 0; n < number_of_grains; n++) {
            grain_stats << unique_grain_id_vector[n];
            if (analysis_options_per_grain_stats_yn[0])
                grain_stats << "," << crystal_misorientation_x_vector[n] << "," << crystal_misorientation_y_vector[n]
                            << "," << crystal_misorientation_z_vector[n];
            if (analysis_options_per_grain_stats_yn[1])
                grain_stats << "," << grain_size_vector_microns[n];
            if (analysis_options_per_grain_stats_yn[2])
                grain_stats << "," << grain_extent_x[n];
            if (analysis_options_per_grain_stats_yn[3])
                grain_stats << "," << grain_extent_y[n];
            if (analysis_options_per_grain_stats_yn[4])
                grain_stats << "," << grain_extent_z[n];
            if (analysis_options_per_grain_stats_yn[5])
                grain_stats << "," << build_trans_aspect_ratio[n];
            if (analysis_options_per_grain_stats_yn[6])
                grain_stats << "," << grain_red[n] << "," << grain_green[n] << "," << grain_blue[n];
            grain_stats << std::endl;
        }
        grain_stats.close();
    }
};

#endif
