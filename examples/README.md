# ExaCA problem types and auxiliary files
ExaCA currently can model five types of problems:

* Problem type `Directional` is a directional solidification problem, with the bottom surface initialized with some fraction of sites home to epitaxial grains and at the liquidus temperature and a positive thermal gradient in the +Z direction. The domain is then cooled at a constant rate. 
* Problem type `Spot` is a hemispherical spot, with fixed thermal gradient magnitude and cooling rate as solidification proceeds from the outer edge of the spot towards the center. The ability to simulate multilayer arrays of overlapping spots was deprecated in version 1.2 and removed after version 1.3
* Problem type `SingleGrain` is an initial nuclei at the domain center growing each time step until a domain edge is reached
* Problem type `FromFile` is a custom solidification problem using time-temperature history file(s) (default location is `examples/Temperatures`). Again note that some example problems require [external data](https://github.com/LLNL/ExaCA-Data). The format of these files are as follows:
    * The first line should be the names of the columns: x, y, z, tm, tl, cr
    * Each line following the first should have six comma-separated values corresponding to x, y, z, tm, tl, cr. x, y, and z are cell coordinates, in meters, of a given location in the simulation. The spacing between locations should correspond to a Cartesian grid, with a cell size equivalent to that specified in the input file. For each time that an x,y,z coordinate went above and below the liqiuidus temperature of the alloy during a heat transport simulation, a tm (time at which the point went above the liquidus), tl (time at which the point went below the liquidus), and cr (instantaneous cooling rate at the liquidus) should be recorded. As meters and seconds are the units used, and the cell size and time step tend to be on the order of micrometers and microseconds, it is recommended that this data be given as double precision values to avoid truncation of values
    * If an x,y,z coordinate melted and solidified multiple times, it should appear in the file multiple times on separate lines. The order of the lines do not matter, except that the header line must be before any data.
    * The top surface (the largest Z coordinate in a file) is assumed to be flat. Additionally, if multiple temperature files are being used (for example, a scan pattern consisting of 10 layers of repeating even and odd file data), the Z coordinate corresponding to this flat top surface should be the same for all files.
    * Alternatively, if a time-temperature history file has the extension `.catemp`, it will be parsed as a binary string. The binary form for these files does not contain commas, newlines, nor a header, but consists of sequential x,y,z,tm,tl,cr,x,y,z,tm,tl,cr... data as double precision values (little endian). This is often a significantly smaller file size than the standard format, and will be faster to read during initialization.
    * The M is no longer required in the problem type, as problem type R is now the same as RM (it now includes multiple melting and solidification events per cell)
* Problem type `FromFinch` uses time-temperature history data generated from Finch, and requires Finch (and as a result, Cabana) as a dependency as well as an input file containing inputs for both codes. Running using this problem type requires use of the `Finch-ExaCA` executable.

# ExaCA top level input files
The .json files in the examples subdirectory are provided on the command line to let ExaCA know which problem is being simulated. The json files contain different input sections, with certain inputs only used for certain problem types. For optional inputs, the default value used is noted. There are two types of input files for ExaCA: "singular" (ExaCA) input files and "combined" (Finch-ExaCA). For combined input files, the top level input sections are `Finch` and `ExaCA`, with subsections `SimulationType`, `MaterialFileName`, `GrainOrientationsFile`, `RandomSeed`, `Domain`, `Nucleation`, `TemperatureData`, `Substrate`, and `Printing` in the ExaCA portion of the file and subsections as given in the Finch README for the Finch portion of the file. Note that for the Finch section of these files, inputs that vary on different layers (for multilayer simulations) can be given in a `layers` list; see `examples/Inp_FinchVariedLayers.json` for an example of this. For singular input files, the aforemenioned ExaCA subsections are top level inputs. Also note that the file path to the Finch scan path file(s) given in the input file may need to be adjusted depending on the location of your Finch build and associated input files.

## Top level inputs (individual input files) or ExaCA level inputs (combined input files)
|Input                   | Details |
|------------------------|---------|
| SimulationType         |
|                        | `Directional` for directional solidification (thermal gradient in build direction, fixed cooling rate)
|                        | `Spot` for spot melt problem (fixed thermal gradient/constant cooling rate for a single hemispherical spot)
|                        | `FromFile` for use of temperature data provided in the appropriate format ([see below](#temperature-inputs))
|                        | `SingleGrain` for solidification of a single grain at the domain center, continuing until it reaches a domain edge
|                        | `FromFinch` for solidification driven by time-temperature history data from a Finch heat transport simulation
| MaterialFileName       | Name of material file in examples/Materials used ([see section](#material-file))
| GrainOrientationFile   | File listing rotation matrix components used in assigning orientations to grains ([see below](#grain-orientation-files)
| RandomSeed             | Value of type double used as the seed to generate baseplate, powder, and nuclei details (default value is 0.0 if not provided)
| Domain                 | Section for parameters that describe the simulation domain for the given problem type ([see below](#domain-inputs))
| Nucleation             | Section for parameters that describe nucleation ([see below](#nucleation-inputs))
| TemperatureData        | Section for parameters/files governing the temperature field for the given problem type ([see below](#temperature-inputs)). Section is unused if temperature data is given from Finch
| Substrate              | Section for parameters/files governing the edge boundary conditions ([see below](#substrate-inputs))
| Printing               | Section for parameters/file names for output data ([see below](#printing-inputs)). If this section is not given, only the log file from the run will be printed

## Domain inputs
| Input        |Relevant problem type(s)| Details |
|--------------|------------------------|---------|
|CellSize      | All                    | CA cell size, in microns
|TimeStep      | All                    | CA time step, in microseconds (note previously for problem type Directional, this was derived from deltax, G, and R)
|Nx            | Directional, SingleGrain    | Domain size in x, in cells
|Ny            | Directional, SingleGrain    | Domain size in y, in cells
|Nz            | Directional, SingleGrain    | Domain size in z, in cells
|NumberOfLayers| FromFile, FromFinch    | Number of layers for which the temperature pattern will be repeated
|LayerOffset   | FromFile, FromFinch    | If NumberOfLayers > 1, the offset (in cells) in the +Z direction for each layer of the temperature pattern
|SpotRadius    | Spot                   | Spot radius, in microns

## Nucleation inputs
| Input            |Relevant problem type(s)| Details |
|------------------|------------------------|---------|
|Density           | Directional, Spot, FromFile, FromFinch  | Density of heterogeneous nucleation sites in the liquid (evenly distributed among cells that are liquid or undergo melting), normalized by 1 x 10^12 m^-3
|MeanUndercooling  | Directional, Spot, FromFile, FromFinch  | Mean nucleation undercooling (relative to the alloy liquidus temperature) for activation of nucleation sites (Gaussian distribution)
|StDevUndercooling | Directional, Spot, FromFile, FromFinch  | Standard deviation of nucleation undercooling (Gaussian distribution), in K

## Temperature inputs
| Input        | Relevant problem type(s)| Details |
|--------------|-------------------------|---------|
|G             | Directional, Spot, SingleGrain    | Thermal gradient in the build (+Z) directions, in K/m
|R             | Directional, Spot, SingleGrain    | Cooling rate (uniform across the domain), in K/s
|LayerwiseTempRead | FromFile, FromFinch           | If set to true, the appropriate temperature data will be read (FromFile) or generated (FromFinch) during each layer's initialization, stored temporarily, and discarded. If set to false, temperature data for all layers will be read/generated and stored during code initialization, and initialization of each layer will be performed using this stored temperature data. Setting this to true is only recommended if a large quantity of temperature data is read by ExaCA (for example, a 10 layer simulation where each layer's temperature data comes from a different file or Finch simulation).
|TemperatureFiles | FromFile             | List of files corresponding to each layer's temperature data, in the form ["filename1.csv","filename2.csv",...]. If the number of entries is less than NumberOfLayers, the list is repeated. Note that if the Z coordinate of the top surface for each data set has the layer offset applied, LayerOffset in the "Domain" section of the input file should be set to 0, to avoid offsetting the layers twice.
|InitUndercooling | Directional, SingleGrain       | For SingleGrain, this is the undercooling at the location of the seeded grain. For problem type Directional, this is an optional argument (defaulting to zero) for the initial undercooling at the domain's bottom surface
|TrimUnmeltedRegion     | FromFinch             | Should ExaCA use only the bounds of the region in the Finch data that underwent solidification? ("true" or "false"). Selecting "false" will give ExaCA the same simulation bounds as Finch, which will allocate a larger domain and may be slower to simulate; however, this may be desirable in the case of a series of simulations that should all use the same fixed XYZ bounds
|TranslationCount       | FromFile, FromFinch             | Number of translations of the Finch temperature data in the direction specified by `OffsetDirection`. Defaults to 0 (does not copy or translate data)
|XRegion                | FromFile, FromFinch             | Optional: Explicit bounds in X (given in the form [xlow,xhigh], in units of meters) for the simulation domain
|YRegion                | FromFile, FromFinch             | Optional: Explicit bounds in Y (given in the form [ylow,yhigh], in units of meters) for the simulation domain
|OffsetDirection        | FromFile, FromFinch             | Direction ("X" or "Y") to translate the temperature data in space
|SpatialOffset          | FromFile, FromFinch             | Distance (in number of cells) to offset the translations of the temperature data in the offset direction
|TemporalOffset         | FromFile, FromFinch             | Amount of time (in microseconds) between the liquidus time values for a given point between translations 
|AlternatingDirection   | FromFile, FromFinch             | Whether or not (true or false) to mirror the data from every other translation about the direction transverse to the offset direction (i.e., to create a bidirectional scan profile from a single line)

## Substrate inputs
| Input              | Relevant problem type(s))| Details |
|--------------------|--------------------------|---------|
|SurfaceSiteFraction | Directional      | What fraction of cells at the bottom surface of the domain are the source of a grain? (see note (b-i))
|SurfaceSiteDensity  | Directional      | Density, in nuclei/µm^2, of grains at the bottom surface of the domain (see note (b-ii))
|GrainLocationsX     | Directional      | List of grain locations in X on the bottom surface of the domain (see note (b-iii))
|GrainLocationsY     | Directional      | List of grain locations in Y on the bottom surface of the domain (see note (b-iii))
|GrainIDs            | Directional      | GrainID values for each grain in (X,Y) (see note (b3))
|FillBottomSurface   | Directional      | Optionally assign all cells on the bottom surface the grain ID of the closest grain (defaults to false)
|MeanBaseplateGrainSize | Spot, FromFile, FromFinch       | Mean spacing between grain centers in the baseplate/substrate (in microns) (see note (a))
|SubstrateFilename   |  Spot, FromFile, FromFinch  | Path to and filename for substrate data in vtk format (see note (a))
|MeanPowderGrainSize | Spot, FromFile, FromFinch    | Mean spacing between grain centers in the powder layer (in microns). Defaults to one grain per cell
|ExtendSubstrateThroughPowder| FromFile, FromFinch  | true/false value: Whether to use the baseplate microstructure as the boundary condition for the entire height of the simulation (defaults to false) (see note (a))
| BaseplateTopZ      | FromFile, FromFinch          | The Z coordinate (in meters) that marks the top of the baseplate/boundary of the baseplate with the powder. If not given, Z = 0 meters will be assumed to be the baseplate top if ExtendSubstrateThroughPowder = false (If ExtendSubstrateThroughPowder = true, the entire domain will be initialized with the baseplate grain structure)
| BaseplateBottomZ   | FromFile, FromFinch          | The Z coordinate (in meters) that marks the bottom of the baseplate. If not given, the bottom of the baseplate is assumed to correspond to the smallest Z coordinate with temperature data. BaseplateBottomZ must be smaller than the smallest Z coordinate with temperature data (i.e., temperature data cannot exist below the baseplate grain structure)
|GrainOrientation | SingleGrain           | Which orientation from the orientation's file is assigned to the grain (starts at 0). Default is 0 
|InitOctahedronSize | All           | Initial size of the octahedra that represent the solid-liquid interface when solidifiation first begins locally. Given as a fraction of a cell size, must be at least 0 and smaller than 1. Default is 0.01

(a) One of these inputs must be provided, but not both
(b) These represent 3 different ways of initializing the interface. Only the inputs corresponding to one mode (i, ii, or iii) should be given. Mode (i) assigns a GrainID value to randomly selected cells at the domain's bottom surface according to the input fixed fraction of sites active. However, this does not guarantee the same grains to be at the same physical (x,y) location if the cell size or domain size are changed. Mode (ii) determines a number of grains based on the input density, and assigns physical (x,y) locations. As only one GrainID per cell is allowed, a large density with a large cell size may lead to underresolution of the desired density - however, this approach does guarantee that the same grains will be initialized at the same physical (x,y) locations in the simulation with changes to cell size or domain size. Mode (iii) requires 3 inputs and allows manual initialization of the substrate with of lists of cell coordinates in X, cell coordinates in Y, and GrainID values.

## Printing inputs
| Input        | Relevant problem type(s))| Details |
|--------------| -------------------------|---------|
| PathToOutput | All                      | File path location for the output files (required)
| OutputFile   | All                      | All output files will begin with the string specified on this line (required)
| PrintBinary  | All                      | Whether or not ExaCA vtk output data should be printed as big endian binary data, or as ASCII characters (defaults to false)
| PrintFrontUndercooling | Directional         | Whether or not ExaCA will store and print the undercooling at the solidification front when solidification begins and ends at each Z coordinate
| PrintExaConstitSize    | FromFile          | Length of the cubic representative volume element (RVE) data for ExaConstit, taken from the domain center in X and Y, and at the domain top in Z excluding the final layer's grain structure. If not given (or given a value of 0), the RVE will not be printed 
| Intralayer   | All | Optional section for printing the state of the simulation during a given layer of a multilayer problem/during a single layer problem
| Intralayer: Increment | All | Increment, in time steps, at which intermediate output should be printed. If 0, will only print the state of the system at the start of each layer
| Intralayer: Fields | All | Fields to print during intralayer increments (see list of fields and descriptions in top-level [README](README.md))
| Intralayer: PrintIdleFrames | All | Whether or not ExaCA should print intermediate output regardless of whether the simulation has changed from the last frame. Defaults to false
| Interlayer   | All | List of options for printing the state of the system following a given layer, or at the end of the run
| Interlayer: Layers | All | List of layers (starting at 0 and through "NumberOfLayers-1") following which the state of the simulation should be printed. If not given (or for non-multilayer problems), defaults to printing only after the full simulation has completed
| Interlayer: Increment | All | If "Interlayer: Layers" is not given, this option enables printing of interlayer output starting at layer 0 and repeating at the specified increment. The full simulation results following the final layer will always be printed as long as the "Interlayer" section is present in the input file".
| Interlayer: Fields | All | Fields to print following layers (see list of fields and descriptions in top-level [README](README.md))

# ExaCA additional input files

|Input                      | Required?      |
|---------------------------|----------------|
| Material file             | Yes            |
| Grain orientations file(s)| Yes            |
| Temperature file(s)       | For problem type `FromFile` only |
| Substrate file            | No             |

## Material file

This file describes the interfacial response function, i.e., the solidification velocity V as a function of undercooling dT in the liquid. Allowed functional forms are "cubic" (V = A(dT)^3 + B(dT)^2 + C(dT) + D), "quadratic" (V = A(dT)^2 + B(dT) + C), "power" (V = A(dT)^b), and "exponential" (V = A exp^(B(dT))). If the "velocity_cap" input is set to true, the solidification velocity will be capped at the value V = V(freezing_range). The material file can consist of an interfacial response for a single solid phase (for example,`examples/Materials/Inconel625.json`) or two solid phases (for example,`examples/Materials/SS316L_AusteniteFerrite.json`). In the case of two phase solidification, the names of the two phases must be given and separate functions must be specified for each phase. For the second phase listed in the file, if the "transformation" option is set to "solidification", solidification that occurs from cells that solidified as the second phase will proceed as the primary phase (however, the fact that solidification initially occured as the second phase will be tracked by the "PhaseID" variable). This phase transformation will be associated either with a random change in crystallographic orientation (if a single grain orientation file was given in the top level input file) or map to a specific new orientation (if two grain orientation files were given in the top level input file). The material file examples are installed with the executable, making it possible to simplify use the file name in the input file. Custom files must either be added to the ExaCA CMake build, use an absolute file path, or a path relative to the ExaCA source. 

## Grain orientation files

This file describes the mapping of "GrainID" values to crystallographic orientations. An example is
`examples/Substrate/GrainOrientationVectors.csv`: the first line is the
number of orientations (10000), and each additional line is a list of unit
vectors corresponding to a cubic grain's <001> directions in the form 'x1, y1,
z1, x2, y2, z2, x3, y3, z3', where the coordinate system used is taken as the
ExaCA reference frame. The distribution of orientations is approximately even.
Like the material file, the orientation file could be swapped out with one
consisting of more (or fewer) orientations, following `GrainOrientationVectors.csv` as a template. 
For two phase solidification (multiple phase names specified in the Material file), two grain 
orientation filenames can be given. In this case, if the "transformation" option from the Material file 
was set to "none", the first orientation file maps "GrainID" values for the primary phase and the 
second orientation file maps "GrainID" values for the second phase. Alternatively, if the 
"transformation" option from the Material file was set to "solidification", the crystallographic
 orientation from second phase solidification (assigned from the second orientation file) will 
 transform into the corresponding crystallographic orientation from the first orientation file upon 
 solidification. Orientation file examples are installed with the executable, making it possible
to simplify use the file name in the input file. Custom files must either be
added to the ExaCA CMake build, use an absolute file path, or a path relative
to the ExaCA source. It should also be noted that if using a custom file of 
grain orientation vectors in an ExaCA simulations, corresponding files that 
list the orientations in bunge Euler angle form (analogous to 
`examples/Substrate/GrainOrientationEulerAnglesBungeZXZ.csv`) and as RGB values 
corresponding to the orientations' inverse pole figure-mapped RGB values 
(analogous to `examples/Substrate/GrainOrientationRGB_IPF-Z.csv`) will be 
necessary to run the analysis executable; see `analysis/README.md` for more 
details.
