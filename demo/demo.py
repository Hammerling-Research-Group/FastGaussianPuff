import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from FastGaussianPuff import GaussianPuff as GP

def main():
    # set simulation parameters
    # IMPORTANT: obs_dt must be a positive integer multiple of sim_dt
    obs_dt, sim_dt, puff_dt = 60, 1, 10 # [seconds]

    # start and end times need to be timezone-aware.
    # start/end are the times you want to simulate between. time_zone is the location you want to simulate in.
    # you can input start and end times in any timezone and they'll get converted to the time_zone you specify.
    # (e.g. input start/end in UTC and set time_zone to "America/Denver" to simulate in mountain time)
    start = pd.to_datetime("2022-01-01 12:00:00-06:00")
    end = pd.to_datetime("2022-01-01 13:00:00-06:00")
    time_zone = "America/Denver"  # alternative: "US/Mountain"

    # fabricated wind data
    fake_times = np.linspace(0,10,61)
    wind_speeds = [3]*61
    wind_directions = 120*np.abs(np.cos(fake_times))
    wind_directions[30:60] -= 40*np.abs(np.sin(6*fake_times[30:60]))

    wind_speeds = np.array(wind_speeds)
    wind_directions = np.array(wind_directions)

    # number of grid points
    x_num, y_num, z_num = 51, 51, 11

    # grid coordinates
    grid_coords = np.array([0, 0, 0, 50, 50, 10]) # format is (x_min, y_min, z_min, x_max, y_max, z_max) in [m]

    # location and emission rate for emitting source
    source_coordinates =  np.array([[25, 25, 5]]) # format is [[x0,y0,z0]] in [m]. needs to be nested list for compatibility with multi source (coming soon)
    emission_rate = np.array([3]) # emission rate for the single source above, [kg/hr]

    gp = GP(
        obs_dt=obs_dt,
        sim_dt=sim_dt,
        puff_dt=puff_dt,
        simulation_start=start,
        simulation_end=end,
        source_coordinates=source_coordinates,
        emission_rates=emission_rate,
        wind_speeds=wind_speeds,
        wind_directions=wind_directions,
        grid_coordinates=grid_coords,
        nx=x_num,
        ny=y_num,
        nz=z_num,
        time_zone=time_zone,
    )

    print("STARTING SIMULATION")
    gp.simulate()
    print("SIMULATION FINISHED")
    print("MAKING PLOTS")

    visualize_puff(gp)

    print("CHECK FILE grid_puff.gif")

def visualize_puff(gp):
    # Reshape ch4_sim to 4D array (time, x, y, z)
    ch4_obs_reshaped = gp.ch4_obs.reshape((gp.n_out, *gp.grid_dims))

    # Just look at one slice of concentration data at height of 4m
    # note: z domain goes from [0,10] and there are 11 grid points, so the 5th grid point is at 4m due to zero indexing
    ch4_2D = ch4_obs_reshaped[:, :, :, 5] 

    # Function to update the figure for each frame
    def update(frame):
        im.set_array(ch4_2D[frame])
        ax.set_title(f't=' + str(frame) + " min", fontsize=20)
        return im,

    # Create initial plot
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(ch4_2D[0], cmap="Greens", origin='upper', aspect="auto")
    fig.subplots_adjust(top=0.93, left=0.07,right=1.05, wspace=None, hspace=None)

    cbar = fig.colorbar(im)
    cbar.set_label("Concentration (ppm)", fontsize=20)
    im.set_clim(0, 60)
    ax.set_xlabel('X', fontsize=20)
    ax.set_ylabel('Y', fontsize=20)

    # Create animation
    ani = FuncAnimation(fig, update, frames=gp.n_out, blit=True)

    # Save animation as GIF
    ani.save("grid_puff.gif", dpi=100)

    plt.close()


# def custom_cmap(gp):
#     mako_cmap = plt.cmap("mako_r", as_cmap=True)
#     # Define the range of the colormap
#     cmin, cmax = gp.ch4_obs.min(), gp.ch4_obs.max()
#     # Normalize the range of the colormap
#     norm = plt.Normalize(vmin=cmin, vmax=cmax)
#     # Get the colormap colors
#     cmap_colors = mako_cmap(norm(np.linspace(cmin, cmax, 256)))
#     # Set the values below 0 to white
#     # cmap_colors[0] = (1, 1, 1, 1)  # RGB values for white

#     custom_cmap = plt.cm.colors.ListedColormap(cmap_colors)
#     return custom_cmap


if __name__ == "__main__":
    main()
