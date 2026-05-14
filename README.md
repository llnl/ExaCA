# ExaCA
An exascale-capable cellular automaton for nucleation and grain growth
ExaCA is a cellular automata (CA) code for grain growth under additive
manufacturing conditions, created by ExaAM within the Exascale Computing Project.

## License

ExaCA is distributed under an [MIT license](LICENSE).

## Citing ExaCA

If you use ExaCA in your work, please cite the following [paper](CITATION.bib).
In addition, cite the current release or version used from
[Zenodo](https://doi.org/10.5281/zenodo.6908176).

### Publications using ExaCA

In addition to the primary ExaCA [citation](CITATION.bib),
[this list of articles](PUBLICATIONS.bib) use ExaCA in their work.

## Contributing

We encourage you to contribute to ExaCA. Please check the
[contribution guidelines](CONTRIBUTING.md).

## Build
ExaCA uses Kokkos and MPI for parallelism, JSON for input files, and CMake to build.

### Package managers

ExaCA is integrated with the following package managers. Detailed build instructions are available below as well.
 - spack: see the [spack documentation](https://spack.io/) for installation and use
    - `spack install exaca` will install with the default options
    - all ExaCA options and hardware details are available through spack, e.g. `spack install exaca +finch +cuda cuda_arch=90`
 - nix: see [NIX.md](./envs/nix/NIX.md) for building ExaCA using Nix
    - the Nix installation currently only supports the Serial Kokkos backend and does not support Finch

### Dependencies

|Dependency | Version | Required | Details|
|---------- | ------- |--------  |------- |
|[CMake](https://cmake.org/download/)       | 3.11+    | Yes      | Build system
|[Kokkos](https://github.com/kokkos/kokkos) | 4.0+    | Yes      | Portable on-node parallelism
|MPI        | GPU Aware if enabled | Yes     | Multi-node parallelism
|[json](https://github.com/nlohmann/json) | 3.10+| Yes       | Input parsing
|[GoogleTest](https://github.com/google/googletest) | 1.10+   | No       | Unit test framework
|CUDA       | 9+      | No       | Programming model for NVIDIA GPUs
|HIP        | 3.5+    | No       | Programming model for AMD GPUs
|[Finch](https://github.com/ORNL-MDF/Finch) | main | No | Heat transport model for coupled simulation

CMake must be available to build ExaCA and Kokkos. The underlying parallel programming models and MPI are available on most systems and can generally be found automatically by CMake. Note these dependencies must all be installed first, if not available. Kokkos is also available on many systems; if not, obtain the desired version:
```
git clone https://github.com/kokkos/kokkos.git --branch 4.0.00
```

### Obtaining ExaCA

ExaCA is available on GitHub, by default starting from the current `master` branch:
```
git clone https://github.com/LLNL/ExaCA.git
```

### Backend options
Note that ExaCA runs with the default enabled Kokkos backend
 (see https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html#backend-selection).

ExaCA has been tested with the Serial, OpenMP, Threads, CUDA, and HIP backends.

### Building Kokkos

#### Host build

If Kokkos is not already installed on your system, configure, build, and install Kokkos:
```
# Change this path to desired Kokkos installation location
export KOKKOS_INSTALL_DIR=`pwd`/install/kokkos
# Change this path to Kokkos source
cd ./kokkos
# Configure Kokkos
cmake \
  -B build \
  -D CMAKE_BUILD_TYPE="Release" \
  -D CMAKE_INSTALL_PREFIX=$KOKKOS_INSTALL_DIR \
  -D Kokkos_ENABLE_OPENMP=ON
# Build Kokkos
cmake --build build
# Install Kokkos
cmake --install build
cd ../
```
Note that there are other host backends available. Kokkos architecture flags can also be set to improve performance and must match the hardware you run on (e.g. -DKokkos_ARCH_POWER9=ON); see [Kokkos architecture flag details](https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html#keywords-arch).

#### CUDA build

Similar to the CPU build above, Kokkos can instead be configured and built for NVIDIA GPUs:
```
# Change this path to desired Kokkos installation location
export KOKKOS_INSTALL_DIR=`pwd`/install/kokkos
# Change this path to Kokkos source
cd ./kokkos
# Check the GPU architecture flag matches the hardware
# Configure Kokkos
cmake \
  -B build \
  -D CMAKE_BUILD_TYPE="Release" \
  -D CMAKE_INSTALL_PREFIX=$KOKKOS_INSTALL_DIR \
  -D Kokkos_ENABLE_CUDA=ON \
  -D Kokkos_ARCH_VOLTA70=ON
# Build Kokkos
cmake --build build
# Install Kokkos
cmake --install build
cd ../
```
Note the two flags needed for the `Kokkos::Cuda` backend. The Kokkos architecture flag must match the hardware you run on. Kokkos will automatically redirect the (default) host compiler to `nvcc` in the example above. By default, the host will use `Kokkos::Serial`; other parallel host backends can also be used, e.g. by adding `-D Kokkos_ENABLE_OPENMP` as was done above.

#### HIP Build
To build Kokkos for HIP the `hipcc` compiler must be explicitly passed, along with architecture and backend flags analogous to the previous examples:
```
# Change this path to desired Kokkos installation location
export KOKKOS_INSTALL_DIR=`pwd`/install/kokkos
cd ./kokkos
# Configure Kokkos
cmake \
  -B build \
  -D CMAKE_BUILD_TYPE="Release" \
  -D CMAKE_CXX_COMPILER=hipcc \
  -D CMAKE_INSTALL_PREFIX=$KOKKOS_INSTALL_DIR$ \
  -D Kokkos_ENABLE_HIP=ON \
  -D Kokkos_ARCH_VEGA908=ON
# Build Kokkos
cmake --build build
# Install Kokkos
cmake --install build
cd ../
```
-------
### Build ExaCA

Once Kokkos and MPI are installed, ExaCA can be built:
```
# Change this path to desired ExaCA installation location
export EXACA_INSTALL_DIR=`pwd`/install/exaca
# Change this path to Kokkos installation location
export KOKKOS_INSTALL_DIR=`pwd`/install/kokkos

# Change this path to ExaCA source
# Configure ExaCA
cd ./ExaCA
cmake \
  -B build \
  -D CMAKE_BUILD_TYPE="Release" \
  -D CMAKE_PREFIX_PATH=$KOKKOS_INSTALL_DIR \
  -D CMAKE_INSTALL_PREFIX=$EXACA_INSTALL_DIR
# Build ExaCA
cmake --build build
# Install ExaCA
cmake --install build
cd ../
```

Kokkos will forward the compilation flags and details to ExaCA automatically.

### Building with external JSON
By default, ExaCA will download the JSON library dependency used for input files. This automatic download does not work on all systems; a separate build of this library can be done instead. As with the dependencies described above, first obtain the source:
```
git clone https://github.com/nlohmann/json
```

And then build the json library (header only):
```
# Change this path to desired JSON installation location
export JSON_INSTALL_DIR=`pwd`/install/json

cd json
# Configure json
cmake \
  -B build \
  -D CMAKE_BUILD_TYPE="Release" \
  -D CMAKE_INSTALL_PREFIX=$JSON_INSTALL_DIR \
  -D JSON_BuildTests=OFF
# Build json
cmake --build build
# Install json
cmake --install build
cd ../
```
Then add this install path to the ExaCA configuration (example above) together with the path to Kokkos `-D CMAKE_PREFIX_PATH="$KOKKOS_INSTALL_DIR;$JSON_INSTALL_DIR"` and build ExaCA. Note that quotes are necessary for multiple paths.

### Building with Finch
ExaCA can be compiled with Finch, a finite difference-based heat transport solver, for coupled heat transport and solidification simulation without the need to read time-temperature history data from file(s). The Finch source code and build instructions are available at https://github.com/ORNL-MDF/Finch. To compile ExaCA with Finch, include the path to the Finch install in the `CMAKE_INSTALL_PREFIX`. To require that ExaCA is compiled with Finch, add `ExaCA_REQUIRE_FINCH=ON`.

## Testing ExaCA

Unit tests can be run if the `ExaCA_ENABLE_TESTING` CMake option is enabled in the build and if the GoogleTest framework is available on the system or built locally with the install path passed to ExaCA (see the previous section describing the JSON build and pointing ExaCA to the installation).

After building, tests can be run with `cmake --build build --target test` from the source directory (presuming `build/` is the relative location of the build folder). Tests are automatically generated for all enabled Kokkos backend.

## Running ExaCA

ExaCA runs using an input file, passed on the command line. Example problems are provided in the `examples/` directory - a separate [examples/README.md](examples/README.md) file goes into more detail on the problem types, the optional and required arguments needed for each problem type, and additional files used by ExaCA. The example input files present in this repository are:
 * `Inp_DirSolidification.json`: simulates grain growth from a surface with a fixed thermal gradient and cooling rate
 * `Inp_SmallDirSolidification.json`: a smaller and simpler version of the previous
 * `Inp_SpotMelt.json`: simulates overlapping spot melts with fixed a fixed thermal gradient and cooling rate
 * `Inp_SmallSpotMelt.json`: a smaller and simpler version of the previous
 * `Inp_SmallSpotMelt_Steel2phase.json`: a modified version of the small spot melt problem for stainless steel 316L where two solidification phases can nucleate and grow - the initial substrate is set to the primary phase (austenite), but nucleation of new grains is assigned one of the phase (austenite and ferrite) based on the faster-solidifying phase at the nucleation undercooling. Each phase solidifies using a unique interfacial response function relating solidification velocity to local undercooling.
 * `Inp_SmallSpotMelt_Steel1phase.json`: a control version of the previous problem where only the austenite phase for stainless steel 316L is considered.

Example problems only possible with [external data](https://github.com/LLNL/ExaCA-Data):
 * `Inp_SingleLine.json`: simulates melting and solidification of a single line of melt pool data
 * `Inp_SingleLineTranslate.json`: using the same single line data as in `Inp_SingleLine.json`, create 3 overlapping line melt pools using specified bounds in the Y direction
 * `Inp_TwoLineTwoLayer.json`: simulates two layers consisting of segments of two overlapping melt pools

Run by calling the created executable with an ExaCA input file:
```
mpiexec -n 1 ./build/install/bin/ExaCA examples/Inp_DirSolidification.json
```
Alternatively, the `Finch-ExaCA` executable can be used for coupled Finch-ExaCA simulations. Compilation with `ExaCA_ENABLE_FINCH=ON` is required for these examples with [Finch in-memory coupling](https://github.com/ORNL-MDF/Finch):
 * `Inp_SmallFinch.json`: simulates melting and solidification of a small melt pool segment at a coarse resolution
 * `Inp_Finch.json`: simulates melting and solidification of a small melt pool segment at a fine resolution, repeated for 3 layers
 * `Inp_FinchTranslate.json`: simulates melting and solidification of the small melt pool segment at the fine resolution translated in space to form 3 overlapping segments
 * `Inp_FinchVariedLayers.json`: performs 3 Finch simulations of simulates melting and solidification of a small melt pool segment at the fine resolution, where each simulation uses a unique value for laser absorption, then uses ExaCA to simulate the solidification microstructure
For simulation using the `Finch-ExaCA` executable Finch-ExaCA, both the Finch and ExaCA inputs are given in a combined input file given on the command line (see `examples/README.md` for more details on the format of this file, and the Finch README for the required Finch inputs). Alternatively, if the Finch inputs are not present in the combined Finch-ExaCA input file, and the combined simulation requires a single Finch heat transport simulation, separate input files for Finch and ExaCA can be given on the command line with the Finch input file listed first:
```
mpiexec -n 1 ./build/install/bin/Finch-ExaCA $PATH_TO_FINCH/examples/single_line/inputs_small.json examples/Inp_SmallFinch.json
```
For coupled Finch-ExaCA runs, caution should be taken such that the cell size given in the CA input file matches that used for heat transport as given in the Finch input file or the Finch-ExaCA combined input file. Additionally, `scan_path_file` in the Finch input file or in the Finch-ExaCA combined input file (e.g. for `Inp_Finch.json` this is `examples/single_line/inputs_small_refined.json` in the Finch repository) should be modified to represent an accurate file path.

## Output and post-processing analysis

As detailed further in [examples/README.md](examples/README.md), ExaCA can print output fields from the simulation at either a specified increment during a simulation (or during a given layer of a multilayer problem), or at the end of the simulation (or after specified layers of a multilayer problem). A list of output options is given below, along with descriptions and whether each field is stored for all layers of a multilayer problem or only the current simulated layer.

| Output field | Stored for | Details |
|--------------| -----------|---------|
| GrainID      | All layers | A unique identifier for each grain in the ExaCA simulation. Positive numbers represent grains from either the initial substrate/baseplate or from a powder layer, while negative numbers represent grains formed via a nucleation event. GrainIDs of 0 are used to identify cells from a powder layer that do not have any associated material (i.e., voids or gas in the powder layer). Each GrainID is linked to a grain orientation through the substrate input file - GrainID = 1 or -1 is associated with the first orientation in the file, GrainID = 2 or -2 with the second orientatation, etc, with orientations reused when the number of grains exceeds the number of orientations in the file (for example, [examples/Substrate/`GrainOrientationVectors.csv](examples/Substrate/GrainOrientationVectors.csv) has 10,000 grain orientations, so a GrainID of 10,001 will share its orientation with the GrainID of 1).
| LayerID      | All layers | The final (or most recent) layer in which a given cell undergoes/underwent melting and solidification. Layer numbers start at 0, with -1 used for cells that have not undergone melting/solidification.
| PhaseID      | All layers | The solid phase associated with the most recent solidification in a cell. For single phase solidification problems, this will always be 0, while for two phase solidification problems, this will be 0 for cells that solidified as the primary phase and 1 for cells that solidified as the second phase
| GrainMisorientation  | All layers | The orientation (in degrees) of the grain's nearest <100> direction to the Z axis (the thermal gradient direction for directional solidification problems, the build/layer offset direction for other problems). Epitaxial grains (from the initial grain structure or powder layer) are assigned (rounded) integer values between 0 and 55, while nucleated grains (not present in the initial grain structure) are assigned values between 100 and 155 (the offset of 100 is simply used to ensure the two types of grains are differentiated, but a nucleated grain with a value of 135 actually has <100> misoriented with Z by 35 degrees). All cells belonging to the same grain (same GrainID value) will have the same GrainMisorientation, as no intra-granular misorientations are currently simulated. Also of note is that for intermediate output files where liquid cells are present, these cells are assigned a value of -1, and that unmelted material (or non-existent material in a powder layer) is assigned a value of 200.
| UndercoolingCurrent  | All layers | The current undercooling of a cell (if liquid or active), or the undercooling when a cell was most recently active (if solid). Superheated liquid cells or cells that have not undergone melting and solidification will have a value of 0.
| UndercoolingSolidificationStart  | All layers | For cells that are currently undergoing solidification or have finished solidification, the undercooling when solidification started. Will have a value of 0 for all other cells
| MeltPoolEdge  | All layers | Value of 1 for cells that were at the edge of the time-temperature history data and converted to active cells, value of 0 for all other cells including cells that became active via a cell capture or nucleation event
| MeltTimeStep  | Current layer | The time step at which a cell will go above the liquidus for the final time (either overall, or for a multilayer problem, in the current layer).  Will have a value of -1 for cell that will not go above the liquidus.
| CritTimeStep  | Current layer | The time step at which a cell will go below the liquidus for the final time (either overall, or for a multilayer problem, in the current layer).  Will have a value of -1 for cell that will not go above the liquidus.
| UndercoolingChange  | Current layer | The rate (in K per time step) at which a cell will cool below the liquidus when it undergoes solidification for the final time (either overall, or for a multilayer problem, in the current layer).  Will have a value of -1 for cell that will not go above the liquidus.
| CellType  | Current layer | The state of a given ExaCA cell, mapped from integers to values according to the types given in `src/CAtypes.hpp`
| DiagonalLength  | Current layer | The size of the octahedral envelope used to track the solidification front (will be 0 if a cell has not yet/will not undergo solidification).
| SolidificationEventCounter  | Current layer | The number of times a cell has undergone solidification (either overall, or for a multilayer problem, in the current layer).
| NumberOfSolidificationEvents | Current layer | The number of times a cell will undergo solidification (either overall, or for a multilayer problem, in the current layer).

Analysis of ExaCA vtk data from the final state of a problem can be performed if the vtk data contains, at a minimum, the LayerID and GrainID fields. For example, using output for the test problem `Inp_DirSolidification.json` (output files `TestProblemDirS.vtk` and `TestProblemDirS.json`), analysis can be performed using the `ExaCA-GrainAnalysis` executable (installed in the same location as `ExaCA`), with two command line arguments: the first being the path to/name of the analysis input file, and the second being the path to and filename (excluding extensions) of the .vtk and .json files associated with the data set of interest.

```
./build/install/bin/ExaCA-GrainAnalysis analysis/examples/AnalyzeDirS.json TestProblemDirS
```
Within the [analysis/examples/](analysis/examples/) directory, there are example analysis input files. Note that the microstructure data files `TestProblemDirS.vtk` and `TestProblemDirS.json` must both be in the location given on the command line.

The analysis executable, in addition to outputting grain statistics, can also output files that can be further post-processing in Matlab using the MTEX toolbox to generate pole figures, inverse pole figures, and inverse pole figure-colored cross-sections. More details on this are provided in [analysis/README.md](analysis/README.md)

## Automated input file generation using Tasmanian (https://tasmanian.ornl.gov/)

Within the [utilities/](utilities/) directory, an example python script including automated input file generation, launching of an ensemble of ExaCA simulations and analysis of the simulation results, the plotting of surrogate model response surfaces for outputs of interest, and the calibration of the model input parameters to produce target outputs can be run using the python script [utilities/TASMANIAN/DirSolCalibration.py](utilities/TASMANIAN/DirSolCalibration.py). Python dependencies to run this script are given in [utilities/TASMANIAN/requirements.txt](utilities/TASMANIAN/requirements.txt). This example is based on the directional solidification test problem from the `examples` directory, where thermal gradient and heterogeneous nucleation density are the two inputs varied across the simulation ensemble members and mean grain area at the top surface and mean grain extent in the Z direction are the two outputs used for calibration. 

## Release

LLNL-CODE-821827
