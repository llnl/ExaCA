// Copyright Lawrence Livermore National Security, LLC and other ExaCA Project Developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#ifndef EXACA_INPUTS_HPP
#define EXACA_INPUTS_HPP

#include "CAgrid.hpp"
#include "CAinfo.hpp"
#include "CAinputdata.hpp"
#include "CAtimers.hpp"

#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <nlohmann/json.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct Inputs {

    std::string simulation_type = "", material_filename = "";
    std::vector<std::string> grain_orientation_file = {};
    unsigned long rng_seed = 0.0;
    DomainInputs domain;
    NucleationInputs nucleation;
    InterfacialResponseInputs irf;
    TemperatureInputs temperature;
    SubstrateInputs substrate;
    PrintInputs print;
    std::string file_name;

    // Creates input struct with uninitialized/default values, used in unit tests
    Inputs(){};

    Inputs(const int id, const std::string input_file)
        : file_name(input_file) {

        // Open and read JSON input file
        bool contains_finch_inputs;
        nlohmann::json input_data = getInputDataObject(input_file, contains_finch_inputs);

        // General inputs
        simulation_type = input_data["SimulationType"];
        // Check for valid simulation type.
        if (simulation_type == "FromFunch")
            simulation_type = "FromFinch";
        validSimulationType(simulation_type);

        // Input files that should be present for all problem types
        std::string material_filename_read = input_data["MaterialFileName"];
        // Path to file of materials constants based on install/source location
        material_filename = checkFileInstalled(material_filename_read, id);
        checkFileNotEmpty(material_filename);
        // Get interfacial response function coefficients and freezing range from the material input file
        parseIRF(id);
        // Path to file of grain orientations based on install/source location
        std::vector<std::string> grain_orientation_file_read;
        // For two phase problems, each grain is associated with a crystallographic orientation and a phase
        // One orientation file, no transformation: one orientation per grain, no change during solidification (same as
        // single phase) One orientation file, solidification transformation: randomly selected grain orientation from
        // the file (based on the input RNG seed) corresponds to each grain ID during second phase solidification,
        // transforms into a different one during transformation into primary phase on solidification. Two orientation
        // files, no transformation: unique grain orientation based on the appropriate file for each phase, no change
        // during solidification Two orientation files, solidification transformation: unique grain orientation based on
        // the appropriate file for each phase, second phase grain orientations transform into the primary phase
        // orientations on solidification If one orientation file is given, this is used for any phases in the material.
        // If two are given, this must be a two phase problem and the solidification transformation rule must be
        // selected
        if (input_data["GrainOrientationFile"].size() == 1) {
            grain_orientation_file_read.push_back(input_data["GrainOrientationFile"]);
            if (irf.num_phases == 2)
                grain_orientation_file_read.push_back(input_data["GrainOrientationFile"]);
        }
        else if (input_data["GrainOrientationFile"].size() == 2) {
            if ((irf.num_phases == 1) && (id == 0))
                throw std::runtime_error("Error: Only one grain orientation file should be specified for a material "
                                         "with one interfacial response function");
            if ((irf.transformation == irf.none) && (id == 0))
                throw std::runtime_error("Error: Only one grain orientation file should be specified for a material "
                                         "without a phase transformation rule");
            grain_orientation_file_read.push_back(input_data["GrainOrientationFile"][0]);
            grain_orientation_file_read.push_back(input_data["GrainOrientationFile"][1]);
        }
        else {
            if (id == 0)
                throw std::runtime_error("Error: No more than two grain orientation files should be given");
        }
        for (int phase_num = 0; phase_num < irf.num_phases; phase_num++) {
            std::string grain_orientation_file_tmp = checkFileInstalled(grain_orientation_file_read[phase_num], id);
            checkFileNotEmpty(grain_orientation_file_tmp);
            grain_orientation_file.push_back(grain_orientation_file_tmp);
        }
        // Seed for random number generator (defaults to 0 if not given)
        if (input_data.contains("RandomSeed"))
            rng_seed = input_data["RandomSeed"];

        // Domain inputs:
        // Cell size - given in meters, stored in micrometers
        domain.deltax = input_data["Domain"]["CellSize"];
        domain.deltax = domain.deltax * pow(10, -6);
        // Time step - given in seconds, stored in microseconds
        domain.deltat = input_data["Domain"]["TimeStep"];
        domain.deltat = domain.deltat * pow(10, -6);
        if ((simulation_type == "Directional") || (simulation_type == "SingleGrain")) {
            // Domain size, in cells
            domain.nx = input_data["Domain"]["Nx"];
            domain.ny = input_data["Domain"]["Ny"];
            domain.nz = input_data["Domain"]["Nz"];
        }
        else if ((simulation_type == "FromFile") || (simulation_type == "FromFinch")) {
            domain.number_of_layers = input_data["Domain"]["NumberOfLayers"];
            domain.layer_height = input_data["Domain"]["LayerOffset"];
        }
        else if (simulation_type == "Spot") {
            std::vector<std::string> unused_spot_inputs = {"NSpotsX", "NSpotsY", "SpotOffset", "RSpots"};
            int num_unused_spot_inputs = unused_spot_inputs.size();
            for (int n = 0; n < num_unused_spot_inputs; n++)
                if ((input_data["Domain"].contains(unused_spot_inputs[n])) && (id == 0))
                    std::cout << "Note: Input " << unused_spot_inputs[n] << " is unused in the Spot problem type"
                              << std::endl;
            // Radius is given in micrometers, convert to cells
            domain.spot_radius = input_data["Domain"]["SpotRadius"];
            // Calculate nx, ny, and nz based on spot radius
            domain.nz = domain.spot_radius + 1;
            domain.nx = 2 * domain.spot_radius + 2;
            domain.ny = 2 * domain.spot_radius + 2;
        }

        // Nucleation inputs:
        // Nucleation density (normalized by 10^12 m^-3), mean nucleation undercooling/st dev undercooling(K)
        // SingleGrain problem type does not have nucleation, just a grain of a single orientation in the domain center
        if (simulation_type != "SingleGrain") {
            nucleation.n_max = input_data["Nucleation"]["Density"];
            nucleation.n_max = nucleation.n_max * pow(10, 12);
            nucleation.dtn = input_data["Nucleation"]["MeanUndercooling"];
            nucleation.dtsigma = input_data["Nucleation"]["StDev"];
        }

        // Temperature inputs:
        if ((simulation_type == "FromFile") || (simulation_type == "FromFinch")) {
            // Read/store all temperature files at once (default), or one at a time?
            if (input_data["TemperatureData"].contains("LayerwiseTempRead"))
                temperature.layerwise_temp_read = input_data["TemperatureData"]["LayerwiseTempRead"];
            // Specified region in X and Y for simulation
            if (input_data["TemperatureData"].contains("XRegion")) {
                temperature.temperature_x_bounds[0] = input_data["TemperatureData"]["XRegion"][0];
                temperature.temperature_x_bounds[1] = input_data["TemperatureData"]["XRegion"][1];
                temperature.use_fixed_x_bounds = true;
            }
            if (input_data["TemperatureData"].contains("YRegion")) {
                temperature.temperature_y_bounds[0] = input_data["TemperatureData"]["YRegion"][0];
                temperature.temperature_y_bounds[1] = input_data["TemperatureData"]["YRegion"][1];
                temperature.use_fixed_y_bounds = true;
            }
            if ((input_data["TemperatureData"].contains("HeatTransferCellSize")) && (id == 0))
                std::cout << "Note: Heat transport data cell size is no longer an input used in ExaCA, temperature "
                             "data must be at the same resolution as the CA cell size"
                          << std::endl;
            // Optional translation/mirroring of input temperature data
            if (input_data["TemperatureData"].contains("TranslationCount")) {
                temperature.number_of_copies = input_data["TemperatureData"]["TranslationCount"];
                // Include the original in the total number of copies (number of translations + 1)
                temperature.number_of_copies++;
                std::string offset_direction = input_data["TemperatureData"]["OffsetDirection"];
                // Offsets are given in cells (for consistency with layer height) and microseconds (for consistency
                // with time step) but stored in meters and seconds for consistency with how cell size and time step
                // are treated
                if (offset_direction == "X") {
                    // X offset from input file, Y defaults to 0
                    temperature.x_offset = input_data["TemperatureData"]["SpatialOffset"];
                }
                else if (offset_direction == "Y") {
                    // Y offset from input file, X defaults to 0
                    temperature.y_offset = input_data["TemperatureData"]["SpatialOffset"];
                }
                else
                    throw std::runtime_error("Error: OffsetDirection must be either X or Y");
                temperature.x_offset = temperature.x_offset * domain.deltax;
                temperature.y_offset = temperature.y_offset * domain.deltax;
                temperature.temporal_offset = input_data["TemperatureData"]["TemporalOffset"];
                temperature.temporal_offset = temperature.temporal_offset * pow(10, -6);

                // For the direction not used to offset copies of the temperature data, should the data be mirrored
                // on every other pass (i.e., bidirectional scanning of a laser)?
                bool alternating_direction = input_data["TemperatureData"]["AlternatingDirection"];
                if (alternating_direction) {
                    // False (default) for offset direction, true for other direction
                    if (offset_direction == "X")
                        temperature.mirror_y = true;
                    else
                        temperature.mirror_x = true;
                }
            }
            if (simulation_type == "FromFile") {
                // Get the paths/number of/names of the temperature data files used
                temperature.temp_files_in_series = input_data["TemperatureData"]["TemperatureFiles"].size();
                if (temperature.temp_files_in_series == 0)
                    throw std::runtime_error(
                        "Error: At least one temperature filename must be given for simulation type FromFile");
                else {
                    for (int filename = 0; filename < temperature.temp_files_in_series; filename++)
                        temperature.temp_paths.push_back(input_data["TemperatureData"]["TemperatureFiles"][filename]);
                }
            }
            else if (simulation_type == "FromFinch") {
                // For FromFinch problems, get the number of finch input files (temp_files_in_series) and, if a list of
                // input files are given in the top level input file, store these (temp_paths)
                if (contains_finch_inputs) {
                    // Parse Finch section of the input file for the list of auxiliary files governing Finch simulations
                    std::ifstream input_data_stream(input_file);
                    nlohmann::json input_data_full = nlohmann::json::parse(input_data_stream);
                    if (input_data_full["Finch"].contains("layers")) {
                        // Number of Finch input files
                        temperature.temp_files_in_series = input_data_full["Finch"]["layers"].size();
                    }
                    else {
                        // Finch data in the top level input file only
                        temperature.temp_files_in_series = 1;
                    }
                }
                else {
                    // Only one Finch input file - was given on command line
                    temperature.temp_files_in_series = 1;
                }
                // Option to trim bounds of Finch data around the region that underwent solidification. If not given,
                // defaults to false
                // As an alternative to the TrimUnmeltedRegion option, fixed X or Y bounds can be given to trim the
                // input data
                if ((input_data["TemperatureData"].contains("TrimUnmeltedRegion"))) {
                    if (id == 0)
                        std::cout << "Warning: TrimUnmeltedRegion option is deprecated and will be removed in a future "
                                     "release, please use TrimUnmeltedRegionLateral or TrimUnmeltedRegionBuild options"
                                  << std::endl;
                    temperature.trim_unmelted_region_xy = input_data["TemperatureData"]["TrimUnmeltedRegion"];
                }
                if ((input_data["TemperatureData"].contains("TrimUnmeltedRegionLateral")))
                    temperature.trim_unmelted_region_xy = input_data["TemperatureData"]["TrimUnmeltedRegionLateral"];
                if ((input_data["TemperatureData"].contains("TrimUnmeltedRegionBuild")))
                    temperature.trim_unmelted_region_z = input_data["TemperatureData"]["TrimUnmeltedRegionBuild"];
            }
        }
        else {
            // Temperature data uses fixed thermal gradient (K/m) and cooling rate (K/s)
            temperature.G = input_data["TemperatureData"]["G"];
            temperature.R = input_data["TemperatureData"]["R"];
            // Optional initial undercooling for problem type Directional (required for type SingleGrain). Defaults to 0
            // for problem type Directional if not given, should be a positive number
            if (input_data["TemperatureData"].contains("InitUndercooling"))
                if (input_data["TemperatureData"]["InitUndercooling"] < 0)
                    throw std::runtime_error("Error: optional temperature data argument InitUndercooling should be "
                                             "greater than or equal to zero");
            if (simulation_type == "SingleGrain")
                temperature.init_undercooling = input_data["TemperatureData"]["InitUndercooling"];
            else if ((simulation_type == "Directional") && (input_data["TemperatureData"].contains("InitUndercooling")))
                temperature.init_undercooling = input_data["TemperatureData"]["InitUndercooling"];
            if ((temperature.G > 0) && (Kokkos::fabs(temperature.R) < 0.000001)) {
                // Throw error for edge case where the cooling rate is 0, but cells in the domain would be initialized
                // above the liquidus temperature (i.e., cells that would never solidify)
                int location_init_undercooling;
                if (simulation_type == "Directional")
                    location_init_undercooling = 0;
                else
                    location_init_undercooling = Kokkos::floorf(static_cast<float>(domain.nz) / 2.0);
                int location_liquidus_isotherm =
                    location_init_undercooling +
                    Kokkos::round(temperature.init_undercooling / (temperature.G * domain.deltax));
                if ((temperature.R == 0) && (location_liquidus_isotherm <= domain.nz - 1))
                    throw std::runtime_error("Error: Domain will not fully solidify based on the size in Z, initial "
                                             "undercooling, thermal gradient, and cooling rate in the input file");
            }
        }

        // Substrate inputs:
        if (simulation_type == "Directional") {
            // Must contain inputs corresponding to one of the three modes
            // fract_surface_sites_active only used for mode (i)
            // surface_site_density only used for mode (ii) - given in grains/mm^2
            // grain_locations_x/grain_locations_y/grain_ids only used for mode (iii)
            // Determine surface initialization mode
            bool mode_i_input = input_data["Substrate"].contains("SurfaceSiteFraction");
            bool mode_ii_input = input_data["Substrate"].contains("SurfaceSiteDensity");
            bool mode_iii_input = ((input_data["Substrate"].contains("GrainLocationsX")) &&
                                   (input_data["Substrate"].contains("GrainLocationsY")) &&
                                   (input_data["Substrate"].contains("GrainIDs")));
            // Check for deprecated input FractionSurfaceSitesActive - which used the logic now associated with the
            // SurfaceSiteDensity input
            bool mode_deprecated_input = input_data["Substrate"].contains("FractionSurfaceSitesActive");
            if ((mode_i_input && mode_ii_input) || (mode_i_input && mode_iii_input) ||
                (mode_ii_input && mode_iii_input) || (mode_i_input && mode_ii_input && mode_iii_input))
                throw std::runtime_error(
                    "Error: Conflicting substrate input parameters detected. See examples/README for details");
            if (mode_i_input) {
                substrate.surface_init_mode = "SurfaceSiteFraction";
                substrate.fract_surface_sites_active = input_data["Substrate"]["SurfaceSiteFraction"];
            }
            else if (mode_ii_input) {
                substrate.surface_init_mode = "SurfaceSiteDensity";
                substrate.surface_site_density = input_data["Substrate"]["SurfaceSiteDensity"];
            }
            else if (mode_iii_input) {
                substrate.surface_init_mode = "Custom";
                // Ensure the number of grains is consistent
                std::size_t num_grains = input_data["Substrate"]["GrainLocationsX"].size();
                if ((input_data["Substrate"]["GrainLocationsY"].size() != num_grains) ||
                    (input_data["Substrate"]["GrainIDs"].size() != num_grains))
                    throw std::runtime_error(
                        "Error: GrainLocationsX, GrainLocationsY, and GrainIDs must be the same size");
                else {
                    for (std::size_t graincount = 0; graincount < num_grains; graincount++) {
                        substrate.grain_locations_x.push_back(input_data["Substrate"]["GrainLocationsX"][graincount]);
                        substrate.grain_locations_y.push_back(input_data["Substrate"]["GrainLocationsY"][graincount]);
                        substrate.grain_ids.push_back(input_data["Substrate"]["GrainIDs"][graincount]);
                    }
                }
            }
            else if (mode_deprecated_input) {
                if (id == 0)
                    std::cout
                        << "Warning: FractionSurfaceSitesActive is a deprecated input for problem type C and support "
                           "will be removed in a future release. See README for updated initialization options"
                        << std::endl;
                // The deprecated input uses the logic currently associated with the SurfaceSiteDensity initialization
                // mode, so the fraction should be transformed into a density
                substrate.surface_init_mode = "SurfaceSiteDensity";
                double fract_surface_active_old = input_data["Substrate"]["FractionSurfaceSitesActive"];
                substrate.surface_site_density =
                    fract_surface_active_old / (domain.deltax * domain.deltax * pow(10, 12));
            }
            else
                throw std::runtime_error("Error: either SurfaceSiteFraction, SurfaceSiteDensity, or lists of "
                                         "GrainLocationsX/GrainLocationsY/GrainIDs are "
                                         "required to initialize this problem type");
            if (input_data["Substrate"].contains("FillBottomSurface"))
                substrate.fill_bottom_surface = input_data["Substrate"]["FillBottomSurface"];
        }
        else if (simulation_type == "SingleGrain") {
            // Orientation of the single grain at the domain center
            substrate.single_grain_orientation = input_data["Substrate"]["GrainOrientation"];
        }
        else {
            // Substrate data - should data come from an initial size or a file?
            if ((input_data["Substrate"].contains("SubstrateFilename")) &&
                ((input_data["Substrate"].contains("MeanSize")) ||
                 (input_data["Substrate"].contains("MeanBaseplateGrainSize"))))
                throw std::runtime_error(
                    "Error: only one of baseplate grain size and substrate structure filename should "
                    "be provided in the input file");
            else if (input_data["Substrate"].contains("SubstrateFilename")) {
                substrate.substrate_filename = input_data["Substrate"]["SubstrateFilename"];
                if (substrate.substrate_filename.substr(substrate.substrate_filename.length() - 4) != ".vtk")
                    throw std::runtime_error("Error: Substrate filename must be a vtk file");
                substrate.use_substrate_file = true;
            }
            else if (input_data["Substrate"].contains("MeanSize") ||
                     (input_data["Substrate"].contains("MeanBaseplateGrainSize"))) {
                // Case where only the deprecated input MeanSize was given
                if (!input_data["Substrate"].contains("MeanBaseplateGrainSize")) {
                    substrate.baseplate_grain_spacing = input_data["Substrate"]["MeanSize"];
                    substrate.use_substrate_file = false;
                    if (id == 0)
                        std::cout
                            << "Warning: deprecated input MeanSize has been replaced with MeanBaseplateGrainSize, "
                               "compatibility with MeanSize input will be removed in a future release"
                            << std::endl;
                }
                else {
                    // Case were the new input MeanBaseplateGrainSize was given
                    substrate.baseplate_grain_spacing = input_data["Substrate"]["MeanBaseplateGrainSize"];
                    substrate.use_substrate_file = false;
                    // Warning for case where both the deprecated and new input were given
                    if ((input_data["Substrate"].contains("MeanSize")) && (id == 0))
                        std::cout << "Warning: redundant inputs MeanSize and MeanBaseplateGrainSize were given - "
                                     "deprecated input MeanSize will be ignored"
                                  << std::endl;
                }
                // Check that baseplate_grain_spacing is valid
                if ((id == 0) && (substrate.baseplate_grain_spacing < domain.deltax))
                    throw std::runtime_error(
                        "Error: Input MeanBaseplateGrainSize must be no smaller than the CA cell size");
            }
            if (simulation_type == "Spot") {
                // No powder for this problem type, baseplate top at Z = 0
                substrate.baseplate_top_z = 0.0;
                substrate.baseplate_through_powder = false;
            }
            else {
                // Should the baseplate microstructure be extended through the powder layers? Default is false
                if (input_data["Substrate"].contains("ExtendSubstrateThroughPower"))
                    substrate.baseplate_through_powder = input_data["Substrate"]["ExtendSubstrateThroughPower"];
                // defaults to a unique grain at each site in the powder layers if not given
                if (input_data["Substrate"].contains("PowderDensity")) {
                    // powder density is given as a density per unit volume, normalized by 10^12 m^-3 --> convert this
                    // into a powder grain spacing
                    if (id == 0)
                        std::cout << "Warning: PowderDensity input has been deprecated and will be removed in a future "
                                     "release, please specify MeanPowderGrainSpacing in microns"
                                  << std::endl;
                    // Warning for a case where the old input PowderDensity and new input MeanPowderGrainSize are both
                    // given
                    if (input_data["Substrate"].contains("MeanPowderGrainSize")) {
                        if (id == 0)
                            std::cout << "Warning: Redundant inputs PowderDensity and MeanPowderGrainSize were given - "
                                         "deprecated input PowderDensity will be ignored"
                                      << std::endl;
                    }
                    else {
                        double powder_density_input = input_data["Substrate"]["PowderDensity"];
                        double powder_active_fraction = powder_density_input * pow(10, 12) * pow(domain.deltax, 3);
                        if ((powder_active_fraction <= 0.0) || (powder_active_fraction > 1.0))
                            throw std::runtime_error(
                                "Error: Density of powder surface sites active must be larger than 0 and less "
                                "than 1/(CA cell volume)");
                        // convert powder density into a powder grain spacing, in microns
                        substrate.powder_grain_spacing =
                            pow((1.0 / (powder_density_input * pow(10, 12))), 1.0 / 3.0) * pow(10, 6);
                        if (id == 0)
                            std::cout << "Calculated powder layer grain spacing using PowderDensity is "
                                      << substrate.powder_grain_spacing << " microns" << std::endl;
                    }
                }
                if (input_data["Substrate"].contains("MeanPowderGrainSize")) {
                    substrate.powder_grain_spacing = input_data["Substrate"]["MeanPowderGrainSize"];
                    if ((id == 0) && (substrate.powder_grain_spacing < domain.deltax))
                        throw std::runtime_error(
                            "Error: Input MeanPowderGrainSize must be no smaller than the CA cell size");
                }
                else if ((!input_data["Substrate"].contains("PowderDensity")) &&
                         (!input_data["Substrate"].contains("MeanPowderGrainSize"))) {
                    // Case where neither PowderDensity nor MeanPowderGrainSize inputs are given - defaults to a mean
                    // powder grain size equal to the cell size (in microns)
                    substrate.powder_grain_spacing = domain.deltax * pow(10, 6);
                }
                if ((input_data["Substrate"].contains("PowderFirstLayer")) && (id == 0))
                    std::cout
                        << "Warning: PowderFirstLayer input is no longer used, the top of the first layer must be "
                           "specified using BaseplateTopZ (which will otherwise default to Z = 0)"
                        << std::endl;
                // The top of the baseplate is designated using BaseplateTopZ (assumed to be Z = 0 if not given in input
                // file)
                if (input_data["Substrate"].contains("BaseplateTopZ"))
                    substrate.baseplate_top_z = input_data["Substrate"]["BaseplateTopZ"];
                if ((substrate.baseplate_through_powder) && ((input_data["Substrate"].contains("PowderDensity")) ||
                                                             (input_data["Substrate"].contains("MeanPowderGrainSize"))))
                    throw std::runtime_error(
                        "Error: if the option to extend the baseplate through the powder layers is "
                        "toggled, a powder layer density or powder grain size cannot be given");
            }
            // Optional input for initial size of octhedra when a cell begins solidification (in units of CA cells)
            if (input_data["Substrate"].contains("InitOctahedronSize")) {
                substrate.init_oct_size = input_data["Substrate"]["InitOctahedronSize"];
                if ((substrate.init_oct_size < 0.0) || (substrate.init_oct_size >= 1.0))
                    throw std::runtime_error("Error: InitOctahedronSize should be at least 0, and less than 1");
            }
            if (input_data["Substrate"].contains("BaseplateBottomZ")) {
                substrate.baseplate_bottom_z = input_data["Substrate"]["BaseplateBottomZ"];
                substrate.use_fixed_z_bounds = true;
            }
        }

        // Printing inputs - leave as defaults (all false) if not given in the input file
        if (input_data.contains("Printing"))
            getPrintDataFromInputFile(input_data, id);
        else {
            print.skip_all_printing = true;
            // Need to have at least the last layer in print_layer_number to avoid accessing empty vector, even if no
            // fields are printed
            print.print_layer_number.push_back(domain.number_of_layers - 1);
        }

        // Print information to console about the input file data read
        if (id == 0) {
            std::cout << "Material simulated is " << material_filename << std::endl;
            std::cout << "CA cell size is " << domain.deltax * pow(10, 6) << " microns" << std::endl;
            std::cout << "Nucleation density is " << nucleation.n_max << " per m^3" << std::endl;
            std::cout << "Mean nucleation undercooling is " << nucleation.dtn
                      << " K, standard deviation of distribution is " << nucleation.dtsigma << "K" << std::endl;
            if (simulation_type == "Directional") {
                std::cout << "CA Simulation using a unidirectional, fixed thermal gradient of " << temperature.G
                          << " K/m and a cooling rate of " << temperature.R << " K/s" << std::endl;
                std::cout << "The time step is " << domain.deltat * pow(10, 6) << " microseconds" << std::endl;
                if (substrate.surface_init_mode == "Custom")
                    std::cout << "Input grain locations and ID values will be used at the bottom surface" << std::endl;
                else if (substrate.surface_init_mode == "SurfaceSiteFraction")
                    std::cout << "The fraction of CA cells at the bottom surface that are active is "
                              << substrate.fract_surface_sites_active << std::endl;
                else if (substrate.surface_init_mode == "SurfaceSiteDensity")
                    std::cout << "The density of active CA cells at the bottom surface is "
                              << substrate.surface_site_density << " grains per µm2" << std::endl;
            }
            else if (simulation_type == "Spot") {
                std::cout << "CA Simulation using a radial, fixed thermal gradient of " << temperature.G
                          << " K/m across a hemispherical spot of radius " << domain.spot_radius
                          << " cells, with a cooling rate of " << temperature.R << " K/s" << std::endl;
                std::cout << "The time step is " << domain.deltat * pow(10, 6) << " microseconds" << std::endl;
            }
            else if (simulation_type == "FromFile") {
                std::cout << "CA Simulation using temperature data from file(s)" << std::endl;
                std::cout << "The time step is " << domain.deltat << " seconds" << std::endl;
                std::cout << "The first temperature data file to be read is " << temperature.temp_paths[0]
                          << ", and there are " << temperature.temp_files_in_series << " in the series" << std::endl;
                std::cout << "A total of " << domain.number_of_layers << " layers of solidification offset by "
                          << domain.layer_height << " CA cells will be simulated" << std::endl;
            }
        }
    }

    // Get input file object from either nested ExaCA structure or the input file top level. Also store whether the file
    // contains a Finch data section to parse
    nlohmann::json getInputDataObject(std::string input_file, bool &contains_finch_inputs) {
        std::ifstream input_data_stream(input_file);
        nlohmann::json input_data_raw = nlohmann::json::parse(input_data_stream);
        // Check if data is within a top level "ExaCA" structure and parse JSON object for ExaCA
        contains_finch_inputs = false;
        if (input_data_raw.contains("ExaCA")) {
            nlohmann::json input_data = input_data_raw["ExaCA"];
            if (input_data_raw.contains("Finch"))
                contains_finch_inputs = true;
            return input_data;
        }
        else
            return input_data_raw;
    }

    // Check that all given fields are valid, and all required fields are given
    void checkPrintFieldValidity(nlohmann::json input_data, const int id) {
        const int num_valid_print_fields = print.print_field_label.size();
        std::vector<bool> required_field_found(num_valid_print_fields, false);
        for (auto &[key, value] : input_data["Printing"].items()) {
            bool valid_field = false;
            // Check key read from file against possible matching fields
            for (int field = 0; field < num_valid_print_fields; field++) {
                if (key == print.print_field_label[field]) {
                    valid_field = true;
                    // If this was a required field, record that the field was found
                    if (print.required_print_field[field])
                        required_field_found[field] = true;
                }
            }
            if ((!valid_field) && (id == 0)) {
                std::string error =
                    "Error: field '" + key + "' from Printing section of input file is not recognized by ExaCA";
                throw std::runtime_error(error);
            }
        }
        for (int field = 0; field < num_valid_print_fields; field++) {
            if ((print.required_print_field[field]) && (!required_field_found[field]) && (id == 0)) {
                std::string error = "Error: Required field \"" + print.print_field_label[field] +
                                    "\" from Printing section of input file was not found";
                throw std::runtime_error(error);
            }
        }
    }

    // Updated version, where fields are organized into intralayer and interlayer
    std::vector<bool> getPrintFieldValues(const int id, nlohmann::json input_data, const std::string fieldtype,
                                          const std::vector<std::string> fieldnames_key) {
        int num_fields_key = fieldnames_key.size();
        int num_fields_given = input_data["Printing"][fieldtype]["Fields"].size();
        std::vector<bool> print_fields_given(num_fields_key, false);
        // Check each given field against each possible input field name
        for (int field_given = 0; field_given < num_fields_given; field_given++) {
            bool valid_field = false;
            std::string field_read = input_data["Printing"][fieldtype]["Fields"][field_given];
            for (int field_key = 0; field_key < num_fields_key; field_key++) {
                if (field_read == fieldnames_key[field_key]) {
                    print_fields_given[field_key] = true;
                    valid_field = true;
                }
            }
            if ((!valid_field) && (id == 0)) {
                std::string error = "Error: Field '" + field_read + "' from " + fieldtype +
                                    " Printing section of input file is not recognized by ExaCA";
                throw std::runtime_error(error);
            }
        }
        return print_fields_given;
    }

    // Read the input data file and initialize appropriate variables to non-default values if necessary
    void getPrintDataFromInputFile(nlohmann::json input_data, const int id) {
        // Check that all given fields are valid, and all required fields are given
        checkPrintFieldValidity(input_data, id);
        // Path to output data
        print.path_to_output = input_data["Printing"]["PathToOutput"];
        // Name of output data
        print.base_filename = input_data["Printing"]["OutputFile"];
        if (simulation_type == "Directional")
            if (input_data["Printing"].contains("PrintFrontUndercooling"))
                print.print_front_undercooling = input_data["Printing"]["PrintFrontUndercooling"];
        // Should ASCII or binary be used to print vtk data? Defaults to ASCII if not given
        if (input_data["Printing"].contains("PrintBinary"))
            print.print_binary = input_data["Printing"]["PrintBinary"];
        // Should default ExaConstit output be printed after the simulation? If so, what size RVE?
        // If a size of 0 is given, this is set to false
        if (input_data["Printing"].contains("PrintExaConstitSize")) {
            print.rve_size = input_data["Printing"]["PrintExaConstitSize"];
            if (print.rve_size != 0)
                print.print_default_rve = true;
        }
        if (input_data["Printing"].contains("Intralayer")) {
            // Fields to be printed during a simulation or during a layer of a simulation - if increment
            if (input_data["Printing"]["Intralayer"]["Increment"] == 0) {
                // Files only printed for the initial state of a given layer
                print.intralayer_increment = INT_MAX;
            }
            else {
                // Files to be printed at some interval during each layer
                print.intralayer_increment = input_data["Printing"]["Intralayer"]["Increment"];
                // Default is false for printing intermediate frames if no changes in the simulation since the prior
                // printed frame
                if (input_data["Printing"]["Intralayer"].contains("PrintIdleFrames"))
                    print.intralayer_idle_frames = input_data["Printing"]["Intralayer"]["PrintIdleFrames"];
            }
            // Which fields should be printed during the layers?
            std::vector<bool> print_fields_intralayer =
                getPrintFieldValues(id, input_data, "Intralayer", print.fieldnames_key);
            if (print_fields_intralayer[0])
                print.intralayer_grain_id = true;
            if (print_fields_intralayer[1])
                print.intralayer_layer_id = true;
            if (print_fields_intralayer[2])
                print.intralayer_phase_id = true;
            if (print_fields_intralayer[3])
                print.intralayer_grain_misorientation = true;
            if (print_fields_intralayer[4])
                print.intralayer_undercooling_current = true;
            if (print_fields_intralayer[5])
                print.intralayer_undercooling_solidification_start = true;
            if (print_fields_intralayer[6])
                print.intralayer_melt_pool_edge = true;
            if (print_fields_intralayer[7])
                print.intralayer_melt_time_step = true;
            if (print_fields_intralayer[8])
                print.intralayer_crit_time_step = true;
            if (print_fields_intralayer[9])
                print.intralayer_undercooling_change = true;
            if (print_fields_intralayer[10])
                print.intralayer_cell_type = true;
            if (print_fields_intralayer[11])
                print.intralayer_diagonal_length = true;
            if (print_fields_intralayer[12])
                print.intralayer_solidification_event_counter = true;
            if (print_fields_intralayer[13])
                print.intralayer_number_of_solidification_events = true;

            // True if any fields are printed
            int num_print_intralayer_inputs = print_fields_intralayer.size();
            for (int n = 0; n < num_print_intralayer_inputs; n++) {
                if (print_fields_intralayer[n])
                    print.intralayer = true;
            }
            // Will fields other than grain misorientations be printed?
            for (int n = 0; n < num_print_intralayer_inputs; n++) {
                if ((n != 3) && (print_fields_intralayer[n]))
                    print.intralayer_non_misorientation_fields = true;
            }
        }
        // List of layers following which interlayer data should be printed (will always print after last layer by
        // default)
        if (input_data["Printing"]["Interlayer"].contains("Layers")) {
            if ((id == 0) && (input_data["Printing"]["Interlayer"].contains("Increment")))
                std::cout << "Warning: A list of layers to print and a layer increment were both present in the "
                             "input file print options, the layer increment will be ignored"
                          << std::endl;
            int num_print_layers = input_data["Printing"]["Interlayer"]["Layers"].size();
            for (int n = 0; n < num_print_layers; n++) {
                int print_layer_val = input_data["Printing"]["Interlayer"]["Layers"][n];
                if (print_layer_val >= domain.number_of_layers) {
                    if (id == 0)
                        std::cout << "Note: adjusting layer value of " << print_layer_val << " to "
                                  << domain.number_of_layers - 1 << " as the simulation only contains layers 0 through "
                                  << domain.number_of_layers << std::endl;
                    print_layer_val = domain.number_of_layers - 1;
                }
                print.print_layer_number.push_back(print_layer_val);
            }
            // Make sure files print after last layer, even if it wasn't listed
            if (print.print_layer_number[num_print_layers - 1] != domain.number_of_layers - 1)
                print.print_layer_number.push_back(domain.number_of_layers - 1);
        }
        else if (input_data["Printing"]["Interlayer"].contains("Increment")) {
            // Print layer numbers starting at 0 and at interlayer_increment, always including the last layer
            int interlayer_increment = input_data["Printing"]["Interlayer"]["Increment"];
            for (int n = 0; n < domain.number_of_layers - 1; n += interlayer_increment)
                print.print_layer_number.push_back(n);
            print.print_layer_number.push_back(domain.number_of_layers - 1);
        }
        else
            print.print_layer_number.push_back(domain.number_of_layers - 1);

        // Which fields should be printed during the layers?
        std::vector<bool> print_fields_interlayer =
            getPrintFieldValues(id, input_data, "Interlayer", print.fieldnames_key);
        if (print_fields_interlayer[0])
            print.interlayer_grain_id = true;
        if (print_fields_interlayer[1])
            print.interlayer_layer_id = true;
        if (print_fields_interlayer[2])
            print.interlayer_phase_id = true;
        if (print_fields_interlayer[3])
            print.interlayer_grain_misorientation = true;
        if (print_fields_interlayer[4])
            print.interlayer_undercooling_current = true;
        if (print_fields_interlayer[5])
            print.interlayer_undercooling_solidification_start = true;
        if (print_fields_interlayer[6])
            print.interlayer_melt_pool_edge = true;
        if (print_fields_interlayer[7])
            print.interlayer_melt_time_step = true;
        if (print_fields_interlayer[8])
            print.interlayer_crit_time_step = true;
        if (print_fields_interlayer[9])
            print.interlayer_undercooling_change = true;
        if (print_fields_interlayer[10])
            print.interlayer_cell_type = true;
        if (print_fields_interlayer[11])
            print.interlayer_diagonal_length = true;
        if (print_fields_interlayer[12])
            print.interlayer_solidification_event_counter = true;
        if (print_fields_interlayer[13])
            print.interlayer_number_of_solidification_events = true;
        if ((print.interlayer_grain_id) || (print.interlayer_phase_id) || (print.interlayer_undercooling_current) ||
            (print.interlayer_undercooling_solidification_start))
            print.interlayer_full = true;
        // First 7 inputs are full domain inputs - check if any of the others were toggled
        int num_interlayer_current_inputs = print_fields_interlayer.size();
        for (int n = 7; n < num_interlayer_current_inputs; n++) {
            if (print_fields_interlayer[n])
                print.interlayer_current = true;
        }
        // Should starting undercooling for solidification be stored?
        if ((print.intralayer_undercooling_solidification_start) ||
            (print.interlayer_undercooling_solidification_start) || (print.print_front_undercooling))
            print.store_solidification_start = true;
        // Should the indicator of whether a cell is at a melt pool edge be stored?
        if ((print.intralayer_melt_pool_edge) || (print.interlayer_melt_pool_edge))
            print.store_melt_pool_edge = true;

        // If no ExaConstit data is to be printed, no intralayer nor interlayer fields are to be printed, and no front
        // undercooling data is to be printed, throw an error (likely an input file typo, ExaCA should at the very least
        // output some feature about the simulated data)
        if (id == 0)
            std::cout << "Successfully parsed data printing options from input file" << std::endl;
    }

    // Print a log file for this ExaCA run in json file format, containing information about the run parameters used
    // from the input file as well as the decomposition scheme
    void printExaCALog(const int id, const int np, const int cycle, const Grid grid, Timers timers,
                       const float vol_fraction_nucleated) {

        std::vector<int> ny_local_allranks(np);
        std::vector<int> y_offset_allranks(np);
        MPI_Gather(&grid.ny_local, 1, MPI_INT, ny_local_allranks.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gather(&grid.y_offset, 1, MPI_INT, y_offset_allranks.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (id == 0) {
            std::string FName = print.path_to_output + print.base_filename + ".json";
            std::cout << "Printing ExaCA log file" << std::endl;
            std::ofstream exaca_log;
            exaca_log.open(FName);
            exaca_log << "{" << std::endl;
            exaca_log << "   \"ExaCAVersion\": \"" << version() << "\", " << std::endl;
            exaca_log << "   \"ExaCACommitHash\": \"" << gitCommitHash() << "\", " << std::endl;
            exaca_log << "   \"KokkosVersion\": \"" << kokkosVersion() << "\", " << std::endl;
            exaca_log << "   \"InputFile\": \"" << file_name << "\", " << std::endl;
            exaca_log << "   \"TimeStepOfOutput\": " << cycle << "," << std::endl;
            exaca_log << "   \"SimulationType\": \"" << simulation_type << "\"," << std::endl;
            exaca_log << "   \"MaterialFileName\": \"" << material_filename << "\"," << std::endl;
            if (irf.num_phases == 1) {
                exaca_log << "   \"PhaseName\": \"" << irf.phase_names[0] << "\"," << std::endl;
                exaca_log << "   \"GrainOrientationFile\": \"" << grain_orientation_file[0] << "\"," << std::endl;
            }
            else if (irf.num_phases == 2) {
                exaca_log << "   \"PhaseName\": [\"" << irf.phase_names[0] << "\", \"" << irf.phase_names[1] << "\"],"
                          << std::endl;
                if (irf.transformation == irf.solidification)
                    exaca_log << "   \"GrainOrientationFile\": [\"" << grain_orientation_file[0] << "\", \""
                              << grain_orientation_file[1] << "\"]," << std::endl;
                else
                    exaca_log << "   \"GrainOrientationFile\": \"" << grain_orientation_file[0] << "\"," << std::endl;
            }
            exaca_log << "   \"Domain\": {" << std::endl;
            exaca_log << "      \"Nx\": " << grid.nx << "," << std::endl;
            exaca_log << "      \"Ny\": " << grid.ny << "," << std::endl;
            exaca_log << "      \"Nz\": " << grid.nz << "," << std::endl;
            exaca_log << "      \"CellSize\": " << grid.deltax << "," << std::endl;
            exaca_log << "      \"TimeStep\": " << domain.deltat << "," << std::endl;
            exaca_log << "      \"XBounds\": [" << grid.x_min << "," << grid.x_max << "]," << std::endl;
            exaca_log << "      \"YBounds\": [" << grid.y_min << "," << grid.y_max << "]," << std::endl;
            exaca_log << "      \"ZBounds\": [" << grid.z_min << "," << grid.z_max << "]";
            if (simulation_type == "FromFile") {
                exaca_log << "," << std::endl;
                exaca_log << "      \"NumberOfLayers\": " << grid.number_of_layers << "," << std::endl;
                exaca_log << "      \"LayerOffset\": " << grid.layer_height;
            }
            else if (simulation_type == "Spot") {
                exaca_log << "," << std::endl;
                exaca_log << "      \"SpotRadius\": " << domain.spot_radius;
            }
            exaca_log << std::endl;
            exaca_log << "   }," << std::endl;
            exaca_log << "   \"Nucleation\": {" << std::endl;
            exaca_log << "      \"Density\": " << nucleation.n_max << "," << std::endl;
            exaca_log << "      \"MeanUndercooling\": " << nucleation.dtn << "," << std::endl;
            exaca_log << "      \"StDevUndercooling\": " << nucleation.dtsigma << "," << std::endl;
            exaca_log << "      \"VolFractionNucleated\": " << vol_fraction_nucleated << std::endl;
            exaca_log << "   }," << std::endl;
            exaca_log << "   \"TemperatureData\": {" << std::endl;
            if (simulation_type == "FromFile") {
                exaca_log << "       \"TemperatureFiles\": [";
                for (int i = 0; i < temperature.temp_files_in_series - 1; i++) {
                    exaca_log << "\"" << temperature.temp_paths[i] << "\", ";
                }
                exaca_log << "\"" << temperature.temp_paths[temperature.temp_files_in_series - 1] << "\"]" << std::endl;
            }
            else {
                exaca_log << "      \"G\": " << temperature.G << "," << std::endl;
                exaca_log << "      \"R\": " << temperature.R << std::endl;
            }
            exaca_log << "   }," << std::endl;
            exaca_log << "   \"Substrate\": {" << std::endl;
            if (simulation_type == "Directional")
                exaca_log << "       \"SurfaceSiteFraction\": " << substrate.fract_surface_sites_active << std::endl;
            else if (simulation_type == "SingleGrain")
                exaca_log << "       \"GrainOrientation\": " << substrate.single_grain_orientation << std::endl;
            else {
                if (substrate.use_substrate_file)
                    exaca_log << "       \"SubstrateFilename\": "
                              << "\"" << substrate.substrate_filename << "\"" << std::endl;
                else {
                    exaca_log << "       \"MeanSize\": " << substrate.baseplate_grain_spacing << "," << std::endl;
                    exaca_log << "       \"MeanBaseplateGrainSize\": " << substrate.baseplate_grain_spacing << ","
                              << std::endl;
                    exaca_log << "       \"MeanPowderGrainSize\": " << substrate.powder_grain_spacing << std::endl;
                }
            }
            exaca_log << "   }," << std::endl;
            exaca_log << "   \"NumberMPIRanks\": " << np << "," << std::endl;
            exaca_log << "   \"Decomposition\": {" << std::endl;
            exaca_log << "       \"SubdomainYSize\": [";
            for (int i = 0; i < np - 1; i++)
                exaca_log << ny_local_allranks[i] << ",";
            exaca_log << ny_local_allranks[np - 1] << "]," << std::endl;
            exaca_log << "       \"SubdomainYOffset\": [";
            for (int i = 0; i < np - 1; i++)
                exaca_log << y_offset_allranks[i] << ",";
            exaca_log << y_offset_allranks[np - 1] << "]" << std::endl;
            exaca_log << "   }," << std::endl;
            exaca_log << timers.printLog() << std::endl;
            exaca_log << "}" << std::endl;
            exaca_log.close();
        }
    }

    // Helper function to phase coefficients, functional form, and freezing range for a specified phase
    void parseIRF_function_form(nlohmann::json irf_phase_data, const int phase_num = 0) {
        irf.A[phase_num] = irf_phase_data["coefficients"]["A"];
        irf.B[phase_num] = irf_phase_data["coefficients"]["B"];
        irf.C[phase_num] = irf_phase_data["coefficients"]["C"];
        std::string functionform = irf_phase_data["function"];
        if (functionform == "cubic") {
            irf.D[phase_num] = irf_phase_data["coefficients"]["D"];
            irf.function[phase_num] = irf.cubic;
        }
        else if ((functionform == "quadratic") || (functionform == "power") || (functionform == "exponential")) {
            // D should not have been given, this functional form only takes 3 input fitting parameters
            if (irf_phase_data["coefficients"]["D"] != nullptr) {
                std::string error = "Error: functional form of this type takes only A, B, and C as inputs";
                throw std::runtime_error(error);
            }
            if (functionform == "quadratic")
                irf.function[phase_num] = irf.quadratic;
            else if (functionform == "power")
                irf.function[phase_num] = irf.power;
            else if (functionform == "exponential")
                irf.function[phase_num] = irf.exponential;
        }
        else
            throw std::runtime_error("Error: Unrecognized functional form for interfacial response function, currently "
                                     "supported options are quadratic, cubic, power, and exponential");
        irf.freezing_range[phase_num] = irf_phase_data["freezing_range"];

        if (irf_phase_data.contains("velocity_cap"))
            irf.velocity_cap[phase_num] = irf_phase_data["velocity_cap"];
    }

    // Get interfacial response function coefficients and freezing range from the material input file and all listed
    // phases
    void parseIRF(const int id) {
        if (id == 0)
            std::cout << "Parsing material file using json input format" << std::endl;
        std::ifstream material_data(material_filename);
        nlohmann::json data = nlohmann::json::parse(material_data);

        // Determine if this material has 1 or 2 phases
        if (data.contains("phases")) {
            irf.num_phases = 2;
            // Get phase labels
            const int num_phases_file = data["phases"].size();
            if (num_phases_file != 2)
                if (id == 0)
                    throw std::runtime_error(
                        "Error: Material file containing 'phases' list must requires two phase names");
            irf.phase_names.push_back(data["phases"][0]);
            irf.phase_names.push_back(data["phases"][1]);
            // For both phases, obtain coefficients, freezing range, functional form
            for (int phase_num = 0; phase_num < irf.num_phases; phase_num++) {
                // IRF data for this phase
                nlohmann::json irf_phase_data = data[irf.phase_names[phase_num]];
                parseIRF_function_form(irf_phase_data, phase_num);
            }
            // Transformation rule for secondary phase -> primary phase
            std::string transformation_read = data[irf.phase_names[1]]["transformation"];
            if (transformation_read == "none")
                irf.transformation = irf.none;
            else if (transformation_read == "solidification")
                irf.transformation = irf.solidification;
            else if (id == 0)
                throw std::runtime_error(
                    "Error: transformation input for second phase must have the value 'none' or 'solidification'");
        }
        else {
            // Only using a single phase, no label
            irf.phase_names.push_back("default");
            // IRF data for the default phase
            parseIRF_function_form(data);
            // Unused second phase values
            irf.A[1] = 0.0;
            irf.B[1] = 0.0;
            irf.C[1] = 0.0;
            irf.freezing_range[1] = 0.0;
        }

        MPI_Barrier(MPI_COMM_WORLD);
        if (id == 0)
            std::cout << "Successfully parsed material input file" << std::endl;
    }
};

#endif
