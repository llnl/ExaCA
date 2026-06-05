// Copyright Lawrence Livermore National Security, LLC and other ExaCA Project Developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "CAorientation.hpp"
#include "CAparsefiles.hpp"
#include "GArepresentativeregion.hpp"
#include "GAutils.hpp"

#include "ExaCA.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Given on the command line: Name of analysis input file, name (without file extension) of the ExaCA data
int main(int argc, char *argv[]) {

    // Initialize Kokkos
    Kokkos::initialize();
    {
        using memory_space = typename Kokkos::DefaultExecutionSpace::memory_space;
        // Read command line input to obtain name of analysis file
        std::string analysis_file;
        if (argc < 3) {
            throw std::runtime_error("Error: Full path to analysis file and path to ExaCA log/microstructure file "
                                     "(without file endings) must be given on "
                                     "the command line. Alternatively, paths to log (with .json) and microstructure "
                                     "output (with .vtk) can be given.\n\nUsage:\n  "
                                     "./ExaCA-Analysis analysis/examples/analysis.json output\n\nOR\n\n  "
                                     "./ExaCA-Analysis analysis/examples/analysis.json output.json output.vtk\n");
        }
        analysis_file = argv[1];
        std::string base_filename = argv[2];
        std::string log_file;
        std::string microstructure_file;

        // JSON file (and VTK) were passed
        if (base_filename.find(".json") != std::string::npos) {
            log_file = base_filename;
            base_filename = base_filename.substr(0, base_filename.find(".json"));
            if (argc < 4) {
                throw std::runtime_error(
                    "Error: Full path to ExaCA log file was given: path to microstructure file required."
                    "\n\nUsage:\n  "
                    "./ExaCA-Analysis analysis/examples/analysis.json out.json out.vtk\n");
            }
            microstructure_file = argv[3];
        }
        // Base file was passed
        else {
            log_file = base_filename + ".json";
            microstructure_file = base_filename + ".vtk";
        }
        std::cout << "Performing analysis of " << microstructure_file << " , using the log file " << log_file
                  << " and the options specified in " << analysis_file << std::endl;

        std::vector<std::string> grain_unit_vector_file, phase_names;
        double deltax;
        int nx, ny, nz, number_of_layers, num_phases;
        std::vector<double> xyz_bounds(6);
        parseLogFile(log_file, nx, ny, nz, deltax, number_of_layers, xyz_bounds, grain_unit_vector_file, phase_names,
                     num_phases, false);
        std::cout << "Parsed log file" << std::endl;

        // Allocate memory blocks for grain_id and layer_id data
        Kokkos::View<int ***, Kokkos::HostSpace> grain_id(Kokkos::ViewAllocateWithoutInitializing("grain_id"), nz, nx,
                                                          ny);
        Kokkos::View<short ***, Kokkos::HostSpace> layer_id(Kokkos::ViewAllocateWithoutInitializing("layer_id"), nz, nx,
                                                            ny);
        Kokkos::View<short ***, Kokkos::HostSpace> phase_id(Kokkos::ViewAllocateWithoutInitializing("phase_id"), nz, nx,
                                                            ny);

        // Whether layer_id was given in the microstructure file
        bool found_layer_id = false;

        // Fill arrays with data from paraview file
        initializeData(microstructure_file, nx, ny, nz, grain_id, layer_id, phase_id, found_layer_id);
        std::cout << "Parsed ExaCA grain structure" << std::endl;

        // Grain unit vectors, grain euler angles, RGB colors for IPF-Z coloring
        // (9*NumberOfOrientations,  3*NumberOfOrientations, and 3*NumberOfOrientations in size, respectively)
        Orientation<memory_space> orientation(0, grain_unit_vector_file, true, num_phases);
        std::cout << "Parsed orientation files" << std::endl;

        // Representative region creation
        std::ifstream analysis_data_stream(analysis_file);
        nlohmann::json analysis_data = nlohmann::json::parse(analysis_data_stream);
        nlohmann::json regions_data = analysis_data["Regions"];
        int number_of_regions = analysis_data["Regions"].size();
        std::cout << "There are " << number_of_regions << " regions to analyze" << std::endl;
        for (auto it = regions_data.begin(); it != regions_data.end(); it++) {
            // Create region
            std::string region_name = it.key();
            RepresentativeRegion representativeregion(analysis_data, region_name, nx, ny, nz, deltax, xyz_bounds,
                                                      grain_id, phase_id);
            std::string base_filename_this_region = base_filename + "_" + region_name;

            // Output file stream for quantities of interest
            std::ofstream qois;
            std::string qois_fname = base_filename_this_region + "_QoIs.txt";
            // Print QoIs file if at least one analysis option is turned on
            if (representativeregion.print_stats_yn) {
                qois.open(qois_fname);
                // Header data for qois file
                representativeregion.printAnalysisHeader(qois);
            }

            // Fraction of region consisting of nucleated grains, unmelted material
            if (representativeregion.analysis_options_stats_yn[0])
                representativeregion.printGrainTypeFractions(qois, grain_id, layer_id, phase_id, num_phases,
                                                             found_layer_id);

            // Calculate and if specified, print misorientation data
            std::vector<float> crystal_misorientation_x_vector =
                representativeregion.getCrystalMisorientation("X", orientation);
            std::vector<float> crystal_misorientation_y_vector =
                representativeregion.getCrystalMisorientation("Y", orientation);
            std::vector<float> crystal_misorientation_z_vector =
                representativeregion.getCrystalMisorientation("Z", orientation);
            if (representativeregion.analysis_options_stats_yn[1])
                representativeregion.printMeanMisorientations(qois, crystal_misorientation_x_vector,
                                                              crystal_misorientation_y_vector,
                                                              crystal_misorientation_z_vector);

            // Print mean size data if specified
            if (representativeregion.analysis_options_stats_yn[2])
                representativeregion.printMeanSize(qois, deltax, num_phases);

            // Obtain extents of grains in X, Y, Z if necessary for requested analysis
            representativeregion.calcNecessaryGrainExtents(grain_id, deltax);
            std::vector<float> build_trans_aspect_ratio(representativeregion.number_of_grains);
            if ((representativeregion.analysis_options_stats_yn[3]) ||
                (representativeregion.analysis_options_per_grain_stats_yn[5]))
                representativeregion.calcBuildTransAspectRatio(build_trans_aspect_ratio);
            if (representativeregion.analysis_options_stats_yn[3])
                representativeregion.printMeanBuildTransAspectRatio(qois);

            if (representativeregion.analysis_options_stats_yn[4])
                representativeregion.printMeanExtent(qois, "X");
            if (representativeregion.analysis_options_stats_yn[5])
                representativeregion.printMeanExtent(qois, "Y");
            if (representativeregion.analysis_options_stats_yn[6])
                representativeregion.printMeanExtent(qois, "Z");

            // Determine IPF-Z color of each grain relative to each direction: 0 (red), 1 (green), 2 (blue)
            std::vector<float> grain_red = representativeregion.getIPFZColor(0, orientation);
            std::vector<float> grain_green = representativeregion.getIPFZColor(1, orientation);
            std::vector<float> grain_blue = representativeregion.getIPFZColor(2, orientation);

            // Write grain area data as a function of Z location in the representative volume if the options are
            // toggled, writing to files
            // "[base_filename_this_region]_WeightedGrainAreas.csv" and "[base_filename_this_region]_GrainAreas.csv",
            // respectively
            if ((representativeregion.analysis_options_per_z_stats_yn[0]) ||
                (representativeregion.analysis_options_per_z_stats_yn[1]))
                representativeregion.writeAreaSeries(base_filename_this_region, deltax, grain_id);
            if (representativeregion.print_stats_yn)
                qois.close();

            // Write per-grain stats for the analysis types specified to the file
            // "[base_filename_this_region]_grains.csv"
            if (representativeregion.print_per_grain_stats_yn)
                representativeregion.writePerGrainStats(
                    base_filename_this_region, crystal_misorientation_x_vector, crystal_misorientation_y_vector,
                    crystal_misorientation_z_vector, build_trans_aspect_ratio, grain_red, grain_green, grain_blue);

            // ExaConstit print a file named "[base_filename_this_region]_ExaConstit.csv"
            if (representativeregion.print_exaconstit_yn) {
                representativeregion.writeExaConstitRVE(base_filename_this_region, deltax, grain_id);
            }

            // Pole figure print a file named "[base_filename_this_region]_PoleFigureData.txt"
            if (representativeregion.print_pole_figure_yn) {
                auto go_histogram = representativeregion.getOrientationHistogram(
                    orientation.n_crystal_orientations, grain_id, layer_id, phase_id, num_phases, found_layer_id);
                representativeregion.writePoleFigure(base_filename_this_region, orientation, go_histogram);
            }

            // IPF map for area print a file named "[base_filename_this_region]_IPFCrossSectionData.txt"
            if (representativeregion.print_inverse_pole_figure_map_yn) {
                representativeregion.writeIPFColoredCrossSection(base_filename_this_region, grain_id, phase_id,
                                                                 orientation, deltax);
            }
            std::cout << "Finished analysis for region " << region_name << std::endl;
        } // end loop over all representative regions in analysis file
    } // end scope for kokkos
    // Finalize kokkos and end program
    Kokkos::finalize();
    return 0;
}
