#include <iostream>
#include <cmath>
#include <vector>
#include <time.h>
#include <chrono>
#include <functional>

#include <algorithm>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/chrono.h>

#include <Eigen/Core>
#include <Eigen/Dense>

typedef Eigen::Vector2d Vector2d;
typedef Eigen::VectorXd Vector;
typedef Eigen::Ref<Vector> RefVector;
typedef Eigen::MatrixXd Matrix;
typedef Eigen::Ref<Matrix, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> RefMatrix;

typedef std::vector<std::vector<std::vector<int> > > vec3d;
typedef std::chrono::system_clock::time_point TimePoint;

class CGaussianPuff{
protected:
    const Vector X, Y, Z;
    Vector X_centered, Y_centered;
    int N_points;

    double sim_dt, puff_dt;
    double puff_duration;

    const Vector wind_speeds, wind_directions;

    double thresh_xy_max, thresh_z_max;

    time_t sim_start, sim_end;
    Matrix source_coordinates;
    Vector emission_strengths;

    double x0, y0, z0; // current iteration's source coordinates
    double x_min, y_min; // current mins for the grid centered at the current source
    double x_max, y_max; 

    std::function<double(double)> exp;

    double conversion_factor;
    double exp_tol;

    bool quiet;

    const double one_over_two_pi_three_halves = 1/std::pow(2*M_PI, 1.5);
    double cosine; // store value of cosine/sine so we don't have to evaluate it across different functions
    double sine;

public:
    /* Constructor.
    Inputs:
        X, Y, Z: Flattened vectors holding (x,y,z) coordinates of points to simulate [m]
        N: total number of grid points
        sim_dt: time between simulation time steps [s]
        puff_dt: time between creation of two puffs [s]
        puff_duration: maximum amount of time a puff can be alive for. Helps prevent unphysical effects [s]
        sim_start, sim_end: datetime stamps for when to start and end the emission scenario
        wind_speeds, wind_directions: timeseries for wind speeds [m/s] and directions [degrees] at sim_dt resolution
        source_coordinates: source coordinates in (x,y,z) format for each source. size- (n_sources, 3) [m]
        emission_strengths: emission rates for each source [kg/hr]. length: n_sources
        conversion_factor: conversion ratio between kg/m^3 to ppm for CH4
        exp_tol: tolerance for the exponential thresholding applied to the Gaussians. Lower tolerance means less accuracy
        but a faster execution.
        unsafe: enables unsafe math operations that are faster but less accurate due to the approximations used.
        quiet: false if you want output for simulation completeness. true for silent simulation.
    */
    CGaussianPuff(Vector X, Vector Y, Vector Z, int N,
                    double sim_dt, double puff_dt, double puff_duration,
                    TimePoint sim_start, TimePoint sim_end,
                    Vector wind_speeds, Vector wind_directions,
                    Matrix source_coordinates, Vector emission_strengths,
                    double conversion_factor, double exp_tol,
                    bool unsafe, bool quiet)

    : X(X), Y(Y), Z(Z), N_points(N), 
    sim_dt(sim_dt), puff_dt(puff_dt), puff_duration(puff_duration), wind_speeds(wind_speeds), wind_directions(wind_directions),
    source_coordinates(source_coordinates), emission_strengths(emission_strengths),
    conversion_factor(conversion_factor), exp_tol(exp_tol), quiet(quiet) {

        if(unsafe){
            if (!quiet) std::cout << "RUNNING IN UNSAFE MODE\n";
            this->exp = &fastExp; 
        } else {
            this->exp = [](double x){return std::exp(x);};
        }

        this->sim_start = std::chrono::system_clock::to_time_t(sim_start);
        this->sim_end = std::chrono::system_clock::to_time_t(sim_end);
    }

    /* Simulation time loop
    Inputs:
        ch4: 2d array. First index represents simulation time steps, second index is flattened spatial index
    Returns:
        none, but the concentration is added directly to the ch4 array for all time steps
    */
    void simulate(RefMatrix ch4){

        double emission_length = difftime(sim_end, sim_start);
        int n_puffs = ceil(emission_length/puff_dt);

        // later, for multisource: iterate over source coords
        setSourceCoordinates(0);
        double q = emission_strengths[0]/3600; // convert to kg/s

        double emission_per_puff = q*puff_dt;

        tm puff_start = *localtime(&sim_start);
        int current_hour = puff_start.tm_hour;
        double current_min = puff_start.tm_min;
        double puff_dt_minute = static_cast<double>(puff_dt)/60.0; // puff dt in minutes

        double report_ratio = 0.1;

        int puff_lifetime = ceil(puff_duration/sim_dt); // number of time steps a puff can live for
        int puff_to_sim_ratio = round(puff_dt/sim_dt); // number of simulation steps for every puff

        for(int p = 0; p < n_puffs; p++){

            if (PyErr_CheckSignals() != 0) throw pybind11::error_already_set(); // catches ctrl+c signal from python

            // keeps track of current hour. needed to compute stability class
            current_min += puff_dt_minute;
            if(current_min > 60.0){
                current_min = current_min - 60.0; // maintain fractional part over 60
                current_hour += 1;
                if(current_hour > 23) current_hour = 0;
            }

            // bounds check on time
            if(p*puff_to_sim_ratio + puff_lifetime >= ch4.rows()) puff_lifetime = ch4.rows()-p*puff_to_sim_ratio;

            double theta = windDirectionToAngle(wind_directions[p]);

            // computes concentration timeseries for this puff
            concentrationPerPuff(emission_per_puff, theta, wind_speeds[p], 
                                    current_hour, ch4.middleRows(p*puff_to_sim_ratio, puff_lifetime));
            
            if(!quiet && floor(n_puffs*report_ratio) == p){
                std::cout << "Simulation is " << report_ratio*100 << "\% done\n";
                report_ratio += 0.1;
            }
        }
    }

private:

    /* Computes the concentration timeseries for a single puff.
    Inputs:
        q: Total emission corresponding to this puff (kg)
        ws, theta: wind speed (m/s) and wind direction (radians)
        hour: current hour of day (int)
        ch4: 2D concentration array. First index is time, second index is the flattened spatial index.
    Returns:
        None. The concentration is added directly into the ch4 array in GaussianPuffEquation()
    */
    void concentrationPerPuff(double q, double theta, double ws, int hour,
                                RefMatrix ch4){

        // cache cos/sin so they can get reused in other calls
        cosine = cos(theta);
        sine = sin(theta);


        std::vector<char> stability_class = stabilityClassifier(ws, hour);

        GaussianPuffEquation(q, ws,stability_class,ch4);
    }

    void getPuffCenteredSigmas(std::vector<double>& sigma_y, std::vector<double>& sigma_z,
                                         int n_time_steps, double ws, std::vector<char> stability_class){
        std::vector<double> downwind_dists = std::vector<double>(n_time_steps+1);

        double shift_per_step = ws*sim_dt;
        downwind_dists[0] = 0;
        for(int i = 1; i <= n_time_steps; i++){
            downwind_dists[i] = downwind_dists[i-1] + shift_per_step;
        }

        getSigmaCoefficients(sigma_y, sigma_z, stability_class, downwind_dists);
    }

    Vector2d calculateExitLocation(){

        Vector2d box_min(x_min, y_min);
        Vector2d box_max(x_max, y_max);
        Vector2d origin(0,0);
        Vector2d rayDir(cosine, -sine);
        Vector2d invRayDir = rayDir.cwiseInverse();

        Vector2d exit_times = AABB(box_min, box_max, origin, invRayDir);

        return exit_times[1]*rayDir;
    }

    /* Evaluates the Gaussian Puff equation on the grids. 
    Inputs:
        q: Total emission corresponding to this puff (kg)
        ws, wind speed (m/s)
        X_rot, Y_rot: rotated X and Y grids. The Z grid isn't rotated so the member variable is used repeatedly.
        ts: time series the puff is live for. 
        c: 2D concentration array. The first index represents the time step, the second index represents the flattened
        spatial index.
    Returns:
        none, but the concentrations are added into the concentration array.
    */
    void GaussianPuffEquation(
        double q, double ws, std::vector<char> stability_class,
        RefMatrix ch4) {

        Vector2d exit_location = calculateExitLocation();
        double max_downwind_dist = sqrt(exit_location[0]*exit_location[0] + exit_location[1]*exit_location[1]);

        std::vector<double> temp(1);
        std::vector<double> temp_y(1);
        std::vector<double> temp_z(1);
        temp[0] = max_downwind_dist;

        getSigmaCoefficients(temp_y, temp_z, stability_class, temp); // get maximum sigma coeffs
        double sigma_y_max = temp_y[0];
        double sigma_z_max = temp_z[0];


        // compute the maximum plume size
        double prefactor = (q * conversion_factor * one_over_two_pi_three_halves) / (sigma_y_max * sigma_y_max * sigma_z_max);
        double threshold = std::log(exp_tol / (2 * prefactor));
        double thresh_constant = std::sqrt(-2 * threshold);

        thresh_xy_max = sigma_y_max * thresh_constant;
        thresh_z_max = sigma_z_max * thresh_constant;

        // find out when puff will leave the domain
        double t = calculatePlumeTravelTime(thresh_xy_max, ws); // number of seconds til plume leaves grid

        int n_time_steps = ceil(t / sim_dt); // rescale to unitless number of timesteps

        std::vector<double> sigma_y(n_time_steps+1);
        std::vector<double> sigma_z(n_time_steps+1);

        // gets the dispersion coefficient for the puff at each location on its path
        getPuffCenteredSigmas(sigma_y, sigma_z, n_time_steps, ws, stability_class);

        // bound check on time
        if (n_time_steps >= ch4.rows()) {
            n_time_steps = ch4.rows() - 1;
        }

        double shift_per_step = ws*sim_dt;
        double x_shift_per_step = cosine*shift_per_step;
        double y_shift_per_step = -sine*shift_per_step;
        double wind_shift  = 0;
        double x_shift = 0;
        double y_shift = 0;

        // prefactor excluding the sigmas since those change each time step
        prefactor = q*conversion_factor*one_over_two_pi_three_halves;

        // start at 1 since plume hasn't moved any on it's 0th time step so it's effective dispersion is 0
        for (int i = 1; i <= n_time_steps; i++) {
            // recompute threshold we use every step with current plume dispersion coefficient
            double one_over_sig_y = 1/sigma_y[i];
            double one_over_sig_z = 1/sigma_z[i]; 
            double local_prefactor = prefactor*one_over_sig_y*one_over_sig_y*one_over_sig_z;

            double temp = std::log(exp_tol / (2 * local_prefactor));
            double local_thresh = std::sqrt(-2 * temp);

            // wind_shift is distance [m] plume has moved from source
            wind_shift += shift_per_step;
            x_shift += x_shift_per_step;
            y_shift += y_shift_per_step;

            std::vector<int> indices = coarseSpatialThreshold(wind_shift, local_thresh, sigma_y[i], sigma_z[i]);
 
            for (int j : indices) {

                // Skips upwind cells since upwind diffusion is ignored
                if(sigma_y[i] <= 0 || sigma_z[i] <= 0){
                    continue;
                }

                double t_xy = sigma_y[i] * local_thresh; // local threshold

                // Exponential thresholding conditionals
                if (std::abs(X_centered[j] - x_shift) >= t_xy) {
                    continue;
                }

                if (std::abs(Y_centered[j] - y_shift) >= t_xy) {
                    continue;
                }

                double t_z = sigma_z[i] * local_thresh; // local threshold

                if (std::abs(Z[j] - z0) >= t_z) {
                    continue;
                }

                // terms are written in a way to minimize divisions and exp evaluations
                double y_dist_from_cent = (Y_centered[j] - y_shift);
                double x_dist_from_cent = (X_centered[j] - x_shift);
                double z_minus_by_sig = (Z[j] - z0) * one_over_sig_z;
                double z_plus_by_sig = (Z[j] + z0) * one_over_sig_z;

                double one_over_sig_y_sq = one_over_sig_y*one_over_sig_y;

                double term_4_a_arg = z_minus_by_sig * z_minus_by_sig;
                double term_4_b_arg = z_plus_by_sig * z_plus_by_sig;
                double term_3_arg = (y_dist_from_cent * y_dist_from_cent + x_dist_from_cent * x_dist_from_cent)*one_over_sig_y_sq;

                double term_4 = this->exp(-0.5 * (term_3_arg + term_4_a_arg)) + this->exp(-0.5 * (term_3_arg + term_4_b_arg));

                ch4(i, j) += local_prefactor * term_4;
            }
        }
    }

    /* Computes the time step when the plume will exit the computational grid. 
    Inputs:
        thresh_xy: Gaussian threshold on xy
        ws, wind speed (m/s)
    Returns:
        the time when the plume will be fully off the grid.
    */
    double calculatePlumeTravelTime(double thresh_xy, 
                                    double ws){

        Vector2d box_min(-thresh_xy, -thresh_xy);
        Vector2d box_max(thresh_xy, thresh_xy);

        Vector2d grid_min(x_min, y_min);
        Vector2d grid_max(x_max, y_max);

        Vector2d origin(0,0);

        Vector2d rayDir(cosine, -sine);
        Vector2d invRayDir = rayDir.cwiseInverse();

        // finding the last corner of the threshold box to leave the grid
        Vector2d box_times = AABB(box_min, box_max, origin, invRayDir); // find where ray intersects box
        Vector2d backward_collision = box_times[0]*rayDir; // where backwards ray intersects with an edge of the box
        Vector2d box_corner = findNearestCorner(box_min, box_max, backward_collision);

        // find the corner of the grid that the threshold must pass based on the wind direction
        Vector2d grid_middle = 0.5*(grid_max-grid_min).array() + grid_min.array();
        Vector2d grid_times = AABB(grid_min, grid_max, grid_middle, invRayDir);
        Vector2d forward_collision = grid_times[1]*rayDir + grid_middle;
        Vector2d grid_corner = findNearestCorner(grid_min, grid_max, forward_collision);

        // compute travel time between the two corners
        Vector2d distance = (grid_corner-box_corner).cwiseAbs();
        invRayDir = invRayDir.cwiseAbs();
        double travelDistance = (distance.array()*invRayDir.array()).minCoeff();
        double travelTime = travelDistance/ws;

        return travelTime;
    }

    /* Axis Aligned Bounding Box algorithm. Used to compute the intersections between a ray (wind direction) and a square.
    See https://tavianator.com/2022/ray_box_boundary.html for details.
    Inputs:
        box_min, box_max: 2D vectors containing the minimum and maximum corners of a square in the xy plane.
        origin: starting location of the ray.
        invRayDir: elementwise inverse of the ray direction. While the ray direction could be used instead as the input,
            using the inverse saves on computing multiple divisions.
    Returns:
        2D vector containing the times of the ray intersection. For the fast Gaussian Puff algorithm, the ray's origin
        is always within the box. As such, the returns will have tmin < 0 be the backwards intersection and tmax > 0 
        be the forward intersection, where the directions refer to traveling in the positive and negative ray direction.
    */
    Vector AABB(Vector box_min, Vector box_max, Vector origin, Vector invRayDir){

        // casts to arrays make it an elementwise product
        Vector t0 = (box_min-origin).array()*invRayDir.array();
        Vector t1 = (box_max-origin).array()*invRayDir.array();

        double tmax = (t0.cwiseMax(t1)).minCoeff();
        double tmin = (t0.cwiseMin(t1)).maxCoeff();

        return Vector2d(tmin, tmax);
    }
 
    /*  Given a point on the edge of a square, finds the nearest of the four corners of the square.
    Inputs:
        min_corner, max_corner:  minimum and maximum corners of a square (e.g. lower left and top right).
        point: a point on the edge of the square.
    Returns:
        2D Vector containing the coordinates to the nearest corner of the square. 
    */
    Vector findNearestCorner(Vector min_corner, Vector max_corner, Vector point){
        
        Vector2d corner;

        if(abs(min_corner(0)-point(0)) < abs(max_corner(0)-point(0))){
            corner(0) = min_corner(0);
        }else{
            corner(0) = max_corner(0);
        }

        if(abs(min_corner(1)-point(1)) < abs(max_corner(1)-point(1))){
            corner(1) = min_corner(1);
        }else{
            corner(1) = max_corner(1);
        }

        return corner;
    }

    /* Computes Pasquill stability class
    Inputs:
        wind_speed: [m/s]
        hour: current hour of day
    Returns:
        stability_class: character A-F representing a Pasquill stability class
            note: multiple classes can be returned. In this case, they get averaged when computing dispersion coefficients.
    */
    std::vector<char> stabilityClassifier(double wind_speed, int hour, int day_start=7, int day_end=18) {

        bool is_day = (hour >= day_start) && (hour <= day_end);
        if (wind_speed < 2) {
            return is_day ? std::vector<char>{'A', 'B'} : std::vector<char>{'E', 'F'};
        } else if (wind_speed < 3) {
            return is_day ? std::vector<char>{'B'} : std::vector<char>{'E', 'F'};
        } else if (wind_speed < 5) {
            return is_day ? std::vector<char>{'B', 'C'} : std::vector<char>{'D', 'E'};
        } else if (wind_speed < 6) {
            return is_day ? std::vector<char>{'C', 'D'} : std::vector<char>{'D'};
        } else {
            return std::vector<char>{'D'};
        }
    }


    /* Gets dispersion coefficients (sigma_{y,z}) for the entire grid.
        sigma_z = a*x^b, x in km,
        sigma_y = 465.11628*x*tan(THETA) where THETA = 0.017453293*(c-d*ln(x)) where x in km
        Note: sigma_{y,z} = -1 if x < 0 due to there being no upwind dispersion.
    Inputs:
        sigma_y, sigma_z: vectors to fill with dispersion coefficients
        stability_class: a char in A-F from Pasquill stability classes 
        X_rot: rotated version of the X grid
    Returns:
        None, but sigma_y and sigma_z are filled with the dispersion coefficients.
    */
    void getSigmaCoefficients(std::vector<double>& sigma_y, std::vector<double>& sigma_z, std::vector<char> stability_class, std::vector<double>& downwind_dists){

        int n_stab = stability_class.size();

        for(int i = 0; i < downwind_dists.size(); i++){

            double x = downwind_dists[i]*0.001;

            double sigma_y_temp;
            double sigma_z_temp;

            // note: if there are multiple stability classes, we average the dispersion coefficients
            for(int j = 0; j < n_stab; j++){
                char stab = stability_class[j];
                int flag = 0;
                double a, b, c, d;

            if (x <= 0) {
                sigma_y[i] = -1;
                sigma_z[i] = -1;
                break; // don't need to average if upwind
            } else {
                if (stab == 'A') {
                    if (x < 0.1) {
                        a = 122.800;
                        b = 0.94470;
                    } else if (x < 0.15) {
                        a = 158.080;
                        b = 1.05420;
                    } else if (x < 0.20) {
                        a = 170.220;
                        b = 1.09320;
                    } else if (x < 0.25) {
                        a = 179.520;
                        b = 1.12620;
                    } else if (x < 0.30) {
                        a = 217.410;
                        b = 1.26440;
                    } else if (x < 0.40) {
                        a = 258.890;
                        b = 1.40940;
                    } else if (x < 0.50) {
                        a = 346.750;
                        b = 1.72830;
                    } else if (x < 3.11) {
                        a = 453.850;
                        b = 2.11660;
                    } else {
                        flag = 1;
                    }
                    c = 24.1670;
                    d = 2.5334;
                } else if (stab == 'B') {
                    if (x < 0.2) {
                        a = 90.673;
                        b = 0.93198;
                    } else if (x < 0.4) {
                        a = 98.483;
                        b = 0.98332;
                    } else {
                        a = 109.300;
                        b = 1.09710;
                    }
                    c = 18.3330;
                    d = 1.8096;
                } else if (stab == 'C') {
                    a = 61.141;
                    b = 0.91465;
                    c = 12.5000;
                    d = 1.0857;
                } else if (stab == 'D') {
                    if (x < 0.3) {
                        a = 34.459;
                        b = 0.86974;
                    } else if (x < 1) {
                        a = 32.093;
                        b = 0.81066;
                    } else if (x < 3) {
                        a = 32.093;
                        b = 0.64403;
                    } else if (x < 10) {
                        a = 33.504;
                        b = 0.60486;
                    } else if (x < 30) {
                        a = 36.650;
                        b = 0.56589;
                    } else {
                        a = 44.053;
                        b = 0.51179;
                    }
                    c = 8.3330;
                    d = 0.72382;
                } else if (stab == 'E') {
                    if (x < 0.1) {
                        a = 24.260;
                        b = 0.83660;
                    } else if (x < 0.3) {
                        a = 23.331;
                        b = 0.81956;
                    } else if (x < 1) {
                        a = 21.628;
                        b = 0.75660;
                    } else if (x < 2) {
                        a = 21.628;
                        b = 0.63077;
                    } else if (x < 4) {
                        a = 22.534;
                        b = 0.57154;
                    } else if (x < 10) {
                        a = 24.703;
                        b = 0.50527;
                    } else if (x < 20) {
                        a = 26.970;
                        b = 0.46173;
                    } else if (x < 40) {
                        a = 35.420;
                        b = 0.37615;
                    } else {
                        a = 47.618;
                        b = 0.29592;
                    }
                    c = 6.2500;
                    d = 0.54287;
                } else if (stab == 'F') {
                    if (x < 0.2) {
                        a = 15.209;
                        b = 0.81558;
                    } else if (x < 0.7) {
                        a = 14.457;
                        b = 0.78407;
                    } else if (x < 1) {
                        a = 13.953;
                        b = 0.68465;
                    } else if (x < 2) {
                        a = 13.953;
                        b = 0.63227;
                    } else if (x < 3) {
                        a = 14.823;
                        b = 0.54503;
                    } else if (x < 7) {
                        a = 16.187;
                        b = 0.46490;
                    } else if (x < 15) {
                        a = 17.836;
                        b = 0.41507;
                    } else if (x < 30) {
                        a = 22.651;
                        b = 0.32681;
                    } else if (x < 60) {
                        a = 27.074;
                        b = 0.27436;
                    } else {
                        a = 34.219;
                        b = 0.21716;
                    }
                    c = 4.1667;
                    d = 0.36191;
                } else {
                    throw std::invalid_argument("Invalid stability class.");
                }
                double Theta = 0.017453293 * (c - d * std::log(x)); // in radians
                sigma_y_temp = 465.11628 * x * std::tan(Theta); // in meters
    
                if (flag == 0) {
                    sigma_z_temp = a * std::pow(x, b); // in meters
                    sigma_z_temp = std::min(sigma_z_temp, 5000.0);
                } else {
                    sigma_z_temp = 5000.0;
                }
                }
                sigma_y[i] += sigma_y_temp;
                sigma_z[i] += sigma_z_temp;
            }
            sigma_y[i] /= n_stab;
            sigma_z[i] /= n_stab;
        }
    }

    const double deg_to_rad_factor = M_PI/180.0;

    // somewhat hacky way of making this act as an abstract function
    // okay with this for now since it's private and can't get called from python due to lack of bindings
    virtual std::vector<int> coarseSpatialThreshold(double, double, double, double){ throw std::logic_error("Not implemented"); }

    static double fastExp(double x){
        constexpr double a = (1ll << 52) / 0.6931471805599453;
        constexpr double b = (1ll << 52) * (1023 - 0.04367744890362246);
        x = a * x + b;

        constexpr double c = (1ll << 52);
        if (x < c)
            x = 0.0;

        uint64_t n = static_cast<uint64_t>(x);
        std::memcpy(&x, &n, 8);
        return x;
    }

    void setSourceCoordinates(int source_index){
        x0 = source_coordinates(source_index, 0);
        y0 = source_coordinates(source_index, 1);
        z0 = source_coordinates(source_index, 2);

        // center the x,y grids at the source
        X_centered = X.array() - x0;
        Y_centered = Y.array() - y0;

        x_min = X.minCoeff() - x0;
        y_min = Y.minCoeff() - y0;
        x_max = X.maxCoeff() - x0;
        y_max = Y.maxCoeff() - y0;
    }

    // convert wind direction (degrees) to the angle (radians) between the wind vector and positive x-axis
    double windDirectionToAngle(double wd){
        double theta = wd-270;
        theta = theta*deg_to_rad_factor;

        return theta;
    }

};

class GridGaussianPuff : public CGaussianPuff {

    int nx, ny, nz;
    double dx, dy, dz; // grid spacings
public:

    /* Constructor. See CGaussianPuff constructor for information on parameters not mentioned here.
    nx,ny,nz: number of grid points along each axis respectively
    */
    GridGaussianPuff(Vector X, Vector Y, Vector Z, 
                    int nx, int ny, int nz, 
                    double sim_dt, double puff_dt, double puff_duration,
                    TimePoint sim_start, TimePoint sim_end,
                    Vector wind_speeds, Vector wind_directions,
                    Matrix source_coordinates, Vector emission_strengths,
                    double conversion_factor, double exp_tol,
                    bool unsafe, bool quiet)

        : CGaussianPuff(X, Y, Z, nx*ny*nz,
                        sim_dt, puff_dt, puff_duration,
                        sim_start, sim_end,
                        wind_speeds, wind_directions,
                        source_coordinates, emission_strengths,
                        conversion_factor, exp_tol,
                        unsafe, quiet), 

        nx(nx), ny(ny), nz(nz)
    {

        std::vector<double> gridSpacing = computeGridSpacing();
        dx = gridSpacing[0];
        dy = gridSpacing[1];
        dz = gridSpacing[2];

        // declares empty 3D vector of integers of size (nx, ny, nz)
        vec3d map_table(ny, std::vector<std::vector<int>>(nx, std::vector<int>(nz)));

        // precomputes the map from the 3D meshgrid index to the 1D raveled index.
        // precomputed because the divisions in map() are too expensive to do repeatedly. 
        for(int i = 0; i < nx; i++){
            for(int j = 0; j < ny; j++){
                for(int k = 0; k < nz; k++){
                    // (i,j) index flipped since numpy's 'ij' indexing is being used on the meshgrids
                    map_table[j][i][k] = map(i,j,k);
                }
            }
        }
        this->map_table = map_table;

    }

private:
    vec3d map_table; // precomputed map from the 3D meshgrid index to the 1D raveled index.
    

    // The grid-based spatial thresholding uses inequalities based on grid indices 
    std::vector<int> coarseSpatialThreshold(double wind_shift, double thresh_constant, 
                        double sig_y_current,  double sig_z_current) override {

        std::vector<int> indices = getValidIndices(thresh_xy_max, thresh_z_max, wind_shift);

        return indices;
    }

    /*  Computes bounds on the grid indices based on where the Gaussian is located.
    Inputs:
        thresh_xy, thresh_z: Gaussian thresholds on x and y together, and z separately. These are based on the
            dispersion coefficients of the Gaussian. For the loosest possible bounds, use the largest coefficients.
        ws, wind speed (m/s)
        t_i: time step (s)

    Returns:
        A vector of six doubles containing the lower and upper bounds on the i, j, and k indices. Note that these are
        not rounded to integers as they're used in an intermediate calculation (see calculatePlumeTravelTime) and
        roundind them early creates a rounding error.
    */
    std::vector<double> computeIndexBounds(double thresh_xy, double thresh_z,
                                            double wind_shift){

        Eigen::Matrix2d R;
        R << cosine, -sine,
            sine, cosine;

        Eigen::Vector2d X0;
        X0 << x_min, y_min;

        Eigen::Vector2d v = R.col(0);
        Eigen::Vector2d vp = R.col(1);

        Eigen::Vector2d tw;
        tw << wind_shift, 0;

        Eigen::Vector2d X0_r = R*X0;
        auto X0_rt = X0_r - tw;

        double Xrt_dot_v = X0_rt.dot(v);
        double Xrt_dot_vp = X0_rt.dot(vp);

        double one_over_dx = 1/dx;
        double one_over_dy = 1/dy;
        double one_over_dz = 1/dz;

        double i_lower = (-Xrt_dot_v - thresh_xy - 1)*one_over_dx;
        double i_upper = (-Xrt_dot_v + thresh_xy + 1)*one_over_dx;

        double j_lower = (-Xrt_dot_vp - thresh_xy - 1)*one_over_dy;
        double j_upper = (-Xrt_dot_vp + thresh_xy + 1)*one_over_dy;

        double k_lower = (-thresh_z + z0)*one_over_dz;
        double k_upper = (thresh_z + z0)*one_over_dz;

        return std::vector<double>{i_lower, i_upper, j_lower, j_upper, k_lower, k_upper};
    }

    /* Computes a list of indices to be evaluated based on the location of the Gaussian on the grid.
    Inputs:
        thresh_xy, thresh_z: Gaussian thresholds on x and y together, and z separately. These are based on the
            dispersion coefficients of the Gaussian. For the loosest possible bounds, use the largest coefficients.
        wind_shift: meters that the plume is shifted downwind at the current timestep
        x0, y0, z0: coordinates of source (m)
    Returns:
        A list of indices to the flattened grids where the Gaussian equation should be evaluated.
    */
    std::vector<int> getValidIndices(double thresh_xy, double thresh_z,
                                        double wind_shift){
        
        std::vector<double> indexBounds = computeIndexBounds(thresh_xy, thresh_z,
                                                            wind_shift);

        int i_lower = floor(indexBounds[0]);
        int i_upper = ceil(indexBounds[1]);
        int j_lower = floor(indexBounds[2]);
        int j_upper = ceil(indexBounds[3]);
        int k_lower = floor(indexBounds[4]);
        int k_upper = ceil(indexBounds[5]);

        // makes sure the computed index bounds are sensical and computes total number of cells in bounds
        if(i_lower < 0) i_lower = 0;
        if(i_upper > nx-1) i_upper = nx-1;

        if(j_lower < 0) j_lower = 0;
        if(j_upper > ny-1) j_upper = ny-1;

        if(k_lower < 0) k_lower = 0;
        if(k_upper > nz-1) k_upper = nz-1;

        int i_count, j_count, k_count;
        if(i_upper < i_lower || i_lower > i_upper){
            return std::vector<int>(0);
        } else{
            i_count = i_upper-i_lower+1;
        }

        if(j_upper < j_lower || j_lower > j_upper){
            return std::vector<int>(0);
        } else{
            j_count = j_upper-j_lower+1;
        }

        if(k_upper < k_lower){
            return std::vector<int>(0);
        } else{
            k_count = k_upper-k_lower+1;
        }

        int cellCount = i_count*j_count*k_count;

        std::vector<int> indices(cellCount);
        int currentCell = 0;
        for(int i = i_lower; i <= i_upper; i++){
            for(int j = j_lower; j <= j_upper; j++){
                for(int k = k_lower; k <= k_upper; k++){
                    indices[currentCell] = map_table[j][i][k];
                    currentCell++;
                }
            }
        }

        return indices;
    }

    std::vector<double> computeGridSpacing(){

        std::vector<double> gridSpacing(3); 

        gridSpacing[0] = abs(X[nz] - X[0]); // dx
        gridSpacing[1] = abs(Y[nz*nx] - Y[0]); // dy
        gridSpacing[2] = abs(Z[1] - Z[0]); // dz

        return gridSpacing;
    }

    // maps 3d index to 1d raveled index in numpy 'ij' format meshgrids
    int map(int i, int j, int k){
        return j*nz*nx + i*nz + k;
    }
    
};

class SensorGaussianPuff : public CGaussianPuff {

    std::vector<int> indices;
    
public:

    /* Constructor. See CGaussianPuff constructor for information about parameters not given here.
    N_sensors: Number of sensors being simulated.
    */
    SensorGaussianPuff(Vector X, Vector Y, Vector Z, 
                    int N_sensors,
                    double sim_dt, double puff_dt, double puff_duration,
                    TimePoint sim_start, TimePoint sim_end,
                    Vector wind_speeds, Vector wind_directions,
                    Matrix source_coordinates, Vector emission_strengths,
                    double conversion_factor, double exp_tol,
                    bool unsafe, bool quiet) :
        CGaussianPuff(X, Y, Z, N_sensors,
                        sim_dt, puff_dt, puff_duration,
                        sim_start, sim_end,
                        wind_speeds, wind_directions,
                        source_coordinates, emission_strengths,
                        conversion_factor, exp_tol,
                        unsafe, quiet)
    {

        std::vector<int> inds(N_points);
        for(int i = 0; i < N_sensors; i++){
            inds[i] = i;
        }

        this->indices = inds;
    }
private:
    // mostly a stub, can add a precomputed spatial threshold later
    std::vector<int> coarseSpatialThreshold(double wind_shift, double thresh_constant, 
                        double sig_y_current,  double sig_z_current) override {
        return indices;
    }

};

using namespace pybind11::literals;
namespace py = pybind11;

PYBIND11_MODULE(CGaussianPuff, m) {
    py::class_<GridGaussianPuff>(m, "GridGaussianPuff")
    .def(py::init<Vector, Vector, Vector, int, int, int, double, double, double,
                    TimePoint, TimePoint, 
                    Vector, Vector, Matrix, Vector, double, double, bool, bool>())
    .def("simulate", &CGaussianPuff::simulate);

    py::class_<SensorGaussianPuff>(m, "SensorGaussianPuff")
    .def(py::init<Vector, Vector, Vector, int, double, double, double,
                    TimePoint, TimePoint, 
                    Vector, Vector, Matrix, Vector, double, double, bool, bool>())
    .def("simulate", &CGaussianPuff::simulate);
}
