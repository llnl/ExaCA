import numpy as np
import pandas as pd
import sys
import subprocess
import csv
import numpy as np
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
from scipy.stats import mode as scipymode
import json
import Tasmanian
from Tasmanian import Optimization as Opt, DREAM
import matplotlib.pyplot as plt

def writeExaCAInputFiles(grid_points, phys_bounds, num_points):
    # Min, max thermal gradient
    thermal_grad_min = phys_bounds[0][0]
    thermal_grad_max = phys_bounds[1][0]

    # Min, max thermal gradient
    nuc_density_min = phys_bounds[0][1]
    nuc_density_max = phys_bounds[1][1]

    # Read directional solidification example as template
    with open("examples/Inp_DirSolidification.json", 'r') as file:
        input_template_data = json.load(file)

    # Write input files
    for file_number in range(num_points):
        # Thermal gradient for this ensemble member
        # Scale from 0 to 1 (originally between -1 and 1)
        thermal_grad_scaled = (grid_points[file_number, 0] + 1.0) / 2.0
        thermal_grad_file_number = thermal_grad_min + thermal_grad_scaled * (thermal_grad_max - thermal_grad_min)
        input_template_data["TemperatureData"]["G"] = thermal_grad_file_number
        # Cooling rate (K/s) = isotherm velocity (m/s) * thermal gradient (K/m)
        input_template_data["TemperatureData"]["R"] = thermal_grad_file_number * isotherm_velocity

        # Heterogeneous nucleation density for this ensemble member
        # Scale from 0 to 1 (originally between -1 and 1)
        nuc_density_scaled = (grid_points[file_number, 1] + 1.0) / 2.0
        nuc_density_file_number = nuc_density_min + nuc_density_scaled * (nuc_density_max - nuc_density_min)
        # ExaCA input file takes a density normalized by 10^12 m^-3
        input_template_data["Nucleation"]["Density"] = nuc_density_file_number / 1e12

        # Assign case number to output file
        input_template_data["Printing"]["OutputFile"] = "Case" + str(file_number)

        # Write modified json template data
        modified_input_filename = "examples/Inp_Case" + str(file_number) + ".json"
        with open(modified_input_filename, 'w') as file:
            json.dump(input_template_data, file, indent=4)

# Read CA data from log files and return array of results
def readExaCAOutput(num_points):
    predictions = np.zeros([num_points, 2], dtype=float)
    search_strings = ["mean grain area is ", "mean grain extent in the Z direction is"]
    for sim_number in range(num_points):
        filename = "Case" + str(sim_number) + "_MidCrossSectionXZ_QoIs.txt"
        with open(filename, 'r') as file:
            for line in file:
                # Check if the search string is in the current line
                for search_string in search_strings:
                    if search_string in line:
                        # Find the index where the search string ends and take the rest of the line
                        start_index = line.find(search_string) + len(search_string)
                        remaining_text = line[start_index:]
                        value = remaining_text.split()[0]
                        idx = search_strings.index(search_string)
                        predictions[sim_number][idx] = float(value)
    return predictions

def performCalibration(grid, phys_bounds, target_outputs, target_stdev = 0.2):

    # Transform grid ranges to map to physical input bounds
    grid.setDomainTransform(np.array(phys_bounds).T)
    # Calculate mean and standard deviation for each dimension
    target_y = np.array([target_outputs[0], target_outputs[1]]).T
    target_output_stdev = np.array([target_stdev*target_outputs[0], target_stdev*target_outputs[1]]).T
    target_variance = np.power(target_output_stdev, 2.0)
    
    # Select m and k for sampling
    points = grid.getPoints()
    min_points = np.min(points, axis=0)
    max_points = np.max(points, axis=0)
    print(f"Using prior uniform[{min_points},{max_points}]")

    # Calibration parameters
    num_chains = 50
    num_burnup_iterations = 50000
    num_collection_iterations = 50000

    state = DREAM.State(num_chains, 2)
    state.setState(DREAM.genUniformSamples(min_points, max_points, state.getNumChains()))

    # --- vectorized likelihood over a batch of parameter vectors ---
    def likelihood(x_batch):
        # ensure x_batch is an (nChains × dim) NumPy array
        X = np.atleast_2d(x_batch).astype(np.float64)

        # evaluate all at once; returns shape (nChains, nOutputs)
        Y_pred = grid.evaluateBatch(X)

        # compute per-chain, joint Gaussian likelihood
        # sq has shape (nChains, 2)
        sq    = ((Y_pred - target_y[np.newaxis, :])**2) / target_variance[np.newaxis, :]
        numer = np.exp(-0.5 * np.sum(sq, axis=1))
        denom = np.prod(np.sqrt(2 * np.pi * target_variance))
        return numer / denom

    DREAM.Sample(num_burnup_iterations,
                    num_collection_iterations,
                DREAM.PosteriorEmbeddedLikelihood(lambda x: likelihood(x),
                                                  prior='uniform'),
                DREAM.Domain("hypercube", min_points, max_points),
                state,
                DREAM.IndependentUpdate('uniform', 0.05),
                DREAM.DifferentialUpdate(90))

    a_samples = state.getHistory()

    mean, a_variance = state.getHistoryMeanVariance()
    mode = state.getApproximateMode()

    # Save samples to the specified file
    np.savetxt("samples.txt", a_samples)
    # --- Configuration for a single-column Acta Materialia figure (square) ---
    fig_width_in = 4
    params = {
        'axes.labelsize': 8,
        'font.size': 8,
        'mathtext.default': 'regular',
        'axes.titlesize': 8,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'figure.figsize': [fig_width_in, fig_width_in],
        'lines.linewidth': 1,
        'axes.axisbelow': True,
        'figure.dpi': 300,
    }
    plt.rcParams.update(params)
    
    # Load the 2D sample data
    x_data = a_samples[:, 0]
    y_data = a_samples[:, 1]

    # --- Create Figure and GridSpec Layout ---
    fig = plt.figure()
    # 2×2 grid: main joint at [1,0], marginals at [0,0] & [1,1]
    gs = gridspec.GridSpec(
        nrows=2, ncols=2,
        width_ratios=(4, 1), height_ratios=(1, 4),
        wspace=0.05, hspace=0.05
    )

    ax_joint = fig.add_subplot(gs[1, 0])
    ax_marg_x = fig.add_subplot(gs[0, 0], sharex=ax_joint)
    ax_marg_y = fig.add_subplot(gs[1, 1], sharey=ax_joint)

    xy = np.vstack([x_data, y_data])
    kde = gaussian_kde(xy)

    x_grid, y_grid = np.mgrid[x_data.min():x_data.max():100j, y_data.min():y_data.max():100j]
    positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
    z_grid = np.reshape(kde(positions).T, x_grid.shape)
    z_grid /= np.max(z_grid)

    max_index = np.argmax(z_grid)

    print("Thermal gradient calibrated to ", positions[0,max_index], " K/m, nucleation density to ", "{:.3e}".format(positions[1, max_index])," per meter cubed")

    # Plot the filled contour plot on the joint axis
    quad = ax_joint.contourf(
         x_grid, y_grid, z_grid,
         levels=11, cmap="Blues"
    )
    
    # --- Dedicated Colorbar Axis to the right of the whole figure ---
    cbar_ax = fig.add_axes([0.98, 0.12, 0.02, 0.7])
    fig.colorbar(quad, cax=cbar_ax, label='Normalized Probability Density')
    cbar_ax.yaxis.label.set_size(params['ytick.labelsize'])
    cbar_ax.tick_params(labelsize=params['ytick.labelsize'])

    # --- Marginal histograms (with Matplotlib) ---
    
    ax_marg_x.hist(
        x_data,
        bins=30, histtype="stepfilled", alpha=0.7,
        density=True, color='#336699'
    )
    ax_marg_y.hist(
        y_data,
        orientation='horizontal',
        bins=30, histtype="stepfilled", alpha=0.7,
        density=True, color='#336699'
    )
        
    # --- Clean up marginals ---
    ax_marg_x.tick_params(axis="x", labelbottom=False)
    ax_marg_x.set_yticks([])
    ax_marg_x.set_ylabel("")

    ax_marg_y.tick_params(axis="y", labelleft=False)
    ax_marg_y.set_xticks([])
    ax_marg_y.set_xlabel("")
    
    # --- Main plot labels & grid ---
    ax_joint.set_xlabel("Thermal gradient (K/m)")
    ax_joint.set_ylabel("Nucleation density ($m^{-3}$)")
    ax_joint.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.5)

    # --- Reference point ---
    ax_joint.scatter(
       positions[0,max_index], positions[1,max_index],
       color='k', marker='o', s=25, facecolor='white', zorder=10
    )
    # --- Final layout tweaks ---
    fig.subplots_adjust(
        left=0.10, top=0.96,
    )
    fig.savefig(
        'joint_probability_distribution.png',
        dpi=300, bbox_inches='tight'
    )

def plotSurrogate(grid, phys_bounds, results, target_outputs):

    # generate surrogate results on a grid
    surrogate_grid_nx = 40
    surrogate_grid_ny = 40
    x = np.linspace(-1, 1, surrogate_grid_nx)
    y = np.linspace(-1, 1, surrogate_grid_ny)
    XX, YY = np.meshgrid(x, y)
    grid_points = np.array([np.array([x,y]) for x,y in zip(XX.flatten(), YY.flatten())])
    dual_surrogate = grid.evaluateBatch(grid_points)
    mean_grain_area_surrogate = dual_surrogate[:,0]
    mean_grain_area_surrogate = mean_grain_area_surrogate.reshape(XX.shape)
    grain_extent_z_surrogate = dual_surrogate[:,1]
    grain_extent_z_surrogate = grain_extent_z_surrogate.reshape(XX.shape)

    # Plot input ranges
    xs = phys_bounds[0][0] + (phys_bounds[1][0] - phys_bounds[0][0])*(x - np.min(XX))/(np.max(x) - np.min(x))
    ys = phys_bounds[0][1] + (phys_bounds[1][1] - phys_bounds[0][1])*(y - np.min(YY))/(np.max(y) - np.min(y))
    column_minimums = np.min(results, axis=0)
    column_maximums = np.max(results, axis=0)

    # Figure for fishscale area fraction
    fig, ax1 = plt.subplots(ncols=1, nrows=1, constrained_layout=True)

    # Set the fixed upper and lower bounds for the color scale.
    vmin_fixed_1 = column_minimums[0]
    vmax_fixed_1 = column_maximums[0]

    # Define the levels for the fixed color scale.
    num_levels = 41
    levels_1 = np.linspace(vmin_fixed_1, vmax_fixed_1, num_levels)
    im1 = ax1.contourf(
        xs, ys, mean_grain_area_surrogate,
        cmap='viridis',
        levels=levels_1
    )
    plt.colorbar(im1)
    ax1.contour(xs, ys, mean_grain_area_surrogate, colors='k', levels=[target_outputs[0]], linewidths=1.5)
    ax1.set_xlabel("Thermal gradient (K/m)")
    ax1.set_ylabel("Nucleation density ($m^{-3}$)")
    ax1.set_title("Mean grain area ($µm^2$)")
    plt.savefig("surrogate_mean_grain_area.png", dpi=300, bbox_inches='tight')
    #plt.show()

    # Figure for fishscale area fraction
    fig, ax2 = plt.subplots(ncols=1, nrows=1, constrained_layout=True)

    # Set the fixed upper and lower bounds for the color scale.
    vmin_fixed_2 = column_minimums[1]
    vmax_fixed_2 = column_maximums[1]

    # Define the levels for the fixed color scale - same num as first plot
    levels_2 = np.linspace(vmin_fixed_2, vmax_fixed_2, num_levels)

    im2 = ax2.contourf(
       xs, ys, grain_extent_z_surrogate,
       cmap='viridis',
       levels=levels_2,    # Use fixed levels
       vmin=vmin_fixed_2,  # Use fixed vmin
       vmax=vmax_fixed_2   # Use fixed vmax
    )
    plt.colorbar(im2)
    ax2.contour(xs, ys, grain_extent_z_surrogate, colors='k', levels=[target_outputs[1]], linewidths=1.5)

    ax2.set_xlabel("Thermal gradient (K/m)")
    ax2.set_ylabel("Nucleation density ($m^{-3}$)")
    ax2.set_title("Mean grain size in Z (µm)")
    plt.savefig("surrogate_mean_grain_extent_z.png", dpi=300, bbox_inches='tight')
    #plt.show()

def plot_contours(grid, phys_bounds):
    surrogate_grid_nx = 40
    surrogate_grid_ny = 40
    x = np.linspace(-1, 1, surrogate_grid_nx)
    y = np.linspace(-1, 1, surrogate_grid_ny)
    XX, YY = np.meshgrid(x, y)
    grid_points = np.array([np.array([x,y]) for x,y in zip(XX.flatten(), YY.flatten())])
    dual_surrogate = grid.evaluateBatch(grid_points)
    mean_grain_area_surrogate = dual_surrogate[:,0]
    mean_grain_area_surrogate = mean_grain_area_surrogate.reshape(XX.shape)
    grain_extent_z_surrogate = dual_surrogate[:,1]
    grain_extent_z_surrogate = grain_extent_z_surrogate.reshape(XX.shape)
    xs = phys_bounds[0][0] + (phys_bounds[1][0] - phys_bounds[0][0])*(x - np.min(XX))/(np.max(x) - np.min(x))
    ys = phys_bounds[0][1] + (phys_bounds[1][1] - phys_bounds[0][1])*(y - np.min(YY))/(np.max(y) - np.min(y))

    fig, ax = plt.subplots(ncols=1, nrows=1, constrained_layout=True)
    ax.contour(xs, ys, mean_grain_area_surrogate, colors='k', levels=[0.68], linewidths=1.5)
    ax.contour(xs, ys, grain_extent_z_surrogate, colors='k', levels=[exp_mean_grain_area], linewidths=1.5)
    ax.set_xlabel("Nucleation density ($m^{-3}$)")
    ax.set_ylabel("Mean nucleation undercooling (K)")
    ax.set_title("Area fraction of as-solidified ferrite")
    plt.savefig("/Users/65y/Desktop/AMMT_FY25/Calibration/Contours.png", dpi=300, bbox_inches='tight')
    plt.show()

# Main function
if __name__ == "__main__":

    # Ensemble of simulations with varied nucleation density and thermal gradient
    # Fixed isotherm velocity (m/s)
    isotherm_velocity = 0.1

    # Target mean grain area and mean grain extent in Z
    target_ouputs = [100.0, 15.0]
    # Tasmanian uniform grid
    num_inputs = 2
    num_outputs = 2
    grid_res = 1
    level_limits = num_inputs*[grid_res]
    depth = sum(level_limits)
    grid = Tasmanian.makeLocalPolynomialGrid(
        iDimension=num_inputs,
        iOutputs=num_outputs,
        iDepth=depth,
        iOrder=1,
        sRule="localp",
        liLevelLimits=level_limits)
    
    # Values from -1 to 1 for each of the inputs
    grid_points = grid.getPoints()
    # Number of simulations to run
    num_points = len(grid_points)
    
    phys_bounds = [
         [2e5, 1e14], # lower bounds for thermal gradient (K/m), nucleation density (m^-3)
         [2e6, 1e15] # upper bounds for thermal gradient, nucleation density
    ]

    # Write ExaCA input files for the ensemble of simulations
    writeExaCAInputFiles(grid_points, phys_bounds, num_points)

    # Run ExaCA simulations
    print("Starting ExaCA simulations")
    result = subprocess.run(["bash","utilities/TASMANIAN/LaunchEnsemble.sh",str(num_points)], capture_output=True, text=True)
    # Print captured output to terminal
    print("Done with ExaCA simulations")
    # Get ExaCA predictions
    predictions = readExaCAOutput(num_points)
    # Load values into tasmanian grid
    grid.loadNeededValues(predictions)
    # Plot response surfaces for outputs
    plotSurrogate(grid, phys_bounds, predictions, target_ouputs)
    # Calibration of inputs to target outputs
    performCalibration(grid, phys_bounds, target_ouputs)
