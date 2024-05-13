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
    double sim_dt, puff_dt;
    double puff_duration;

    const Vector X, Y, Z;
    Vector X_rot, Y_rot;
    Matrix stackedGrid;
    int N_points;

    const Vector wind_speeds, wind_directions;
    Vector sigma_y, sigma_z;

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

        stackedGrid.resize(2, X.size());

        if(unsafe){
            if (!quiet) std::cout << "RUNNING IN UNSAFE MODE\n";
            this->exp = &fastExp; 
        } else {
            this->exp = [](double x){return std::exp(x);};
        }

        sigma_y = Vector(N_points);
        sigma_z = Vector(N_points);

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

        Vector X_rot(X.size());
        Vector Y_rot(Y.size());

        // rotates X and Y grids, stores in X_rot and Y_rot
        rotatePoints(X_rot, Y_rot);

        char stability_class = stabilityClassifier(ws, hour);

        // gets sigma coefficients and stores in sigma_{y,z} class member vars
        getSigmaCoefficients(stability_class, X_rot);

        GaussianPuffEquation(q, ws,
                            X_rot, Y_rot,
                            ch4);
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
        double q, double ws,
        RefVector X_rot, RefVector Y_rot,
        RefMatrix ch4) {

        double sigma_y_max = sigma_y.maxCoeff();
        double sigma_z_max = sigma_z.maxCoeff();

        // compute thresholds
        double prefactor = (q * conversion_factor * one_over_two_pi_three_halves) / (sigma_y_max * sigma_y_max * sigma_z_max);
        double threshold = std::log(exp_tol / (2 * prefactor));
        double thresh_constant = std::sqrt(-2 * threshold);

        thresh_xy_max = sigma_y_max * thresh_constant;
        thresh_z_max = sigma_z_max * thresh_constant;

        double t = calculatePlumeTravelTime(thresh_xy_max, ws); // number of seconds til plume leaves grid

        int n_time_steps = ceil(t / sim_dt); // rescale to unitless number of timesteps

        // bound check on time
        if (n_time_steps >= ch4.rows()) {
            n_time_steps = ch4.rows() - 1;
        }

        for (int i = n_time_steps; i >= 0; i--) {

            // wind_shift is distance [m] plume has moved from source
            double wind_shift = ws * (i * sim_dt); // i*sim_dt is # of seconds on current time step

            std::vector<int> indices = coarseSpatialThreshold(wind_shift, thresh_constant);

            for (int j : indices) {

                // Skips upwind cells since sigma_{y,z} = -1 for upwind points
                if (sigma_y[j] < 0 || sigma_z[j] < 0) {
                    continue;
                }

                double t_xy = sigma_y[j] * thresh_constant; // local threshold

                // Exponential thresholding conditionals
                if (std::abs(X_rot[j] - wind_shift) >= t_xy) {
                    continue;
                }

                if (std::abs(Y_rot[j]) >= t_xy) {
                    continue;
                }

                double t_z = sigma_z[j] * thresh_constant; // local threshold

                if (std::abs(Z[j] - z0) >= t_z) {
                    continue;
                }

                // terms are written in a way to minimize divisions and exp evaluations
                double one_over_sig_y = 1 / sigma_y[j];
                double one_over_sig_z = 1 / sigma_z[j];

                double y_by_sig = Y_rot[j] * one_over_sig_y;
                double x_by_sig = (X_rot[j] - wind_shift) * one_over_sig_y;
                double z_minus_by_sig = (Z[j] - z0) * one_over_sig_z;
                double z_plus_by_sig = (Z[j] + z0) * one_over_sig_z;

                double term_4_a_arg = z_minus_by_sig * z_minus_by_sig;
                double term_4_b_arg = z_plus_by_sig * z_plus_by_sig;
                double term_3_arg = (y_by_sig * y_by_sig + x_by_sig * x_by_sig);

                double term_1 = q * one_over_two_pi_three_halves * one_over_sig_y * one_over_sig_y * one_over_sig_z;
                double term_4 = this->exp(-0.5 * (term_3_arg + term_4_a_arg)) + this->exp(-0.5 * (term_3_arg + term_4_b_arg));

                ch4(i, j) += term_1 * term_4 * conversion_factor;
            }
        }
    }



    /* Rotates the X and Y grids based on the current wind direction and source location.
    Inputs:
        X_rot, Y_rot: vectors the same size as the grid to get filled with rotated coordinates.
    Returns:
        None, but fills X_rot and Y_rot with the rotated grids.
    */
    void rotatePoints(RefVector X_rot, RefVector Y_rot){

        Eigen::Matrix2d R;
        R << cosine, -sine,
            sine, cosine;

        Matrix R_g = R*stackedGrid;

        X_rot = R_g.row(0);
        Y_rot = R_g.row(1);
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
        2D Vector containing the coordintes to the nearest corner of the square. 
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
    */
    char stabilityClassifier(double wind_speed, int hour, int day_start=7, int day_end=19) {
        bool is_day = (hour >= day_start) && (hour <= day_end);
        char stability_class;

        if (wind_speed < 2.0) {
            stability_class = is_day ? 'A' : 'E';
        } else if (wind_speed < 5.0) {
            stability_class = is_day ? 'B' : 'E';
        } else {
            stability_class = 'D';
        }

        return stability_class;
    }

    /* Gets dispersion coefficients (sigma_{y,z}) for the entire grid.
        sigma_z = a*x^b, x in km,
        sigma_y = 465.11628*x*tan(THETA) where THETA = 0.017453293*(c-d*ln(x)) where x in km
        Note: sigma_{y,z} = -1 if x < 0 due to there being no upwind dispersion.
    Inputs:
        stability_class: a char in A-F from Pasquill stability classes 
        X_rot: rotated version of the X grid
    Returns:
        None, but sigma_y and sigma_z class variables are filled with the dispersion coefficients.
    */
    void getSigmaCoefficients(char stability_class, Vector X_rot){
        X_rot = X_rot.array() * 0.001; // convert to km

        for(int i = 0; i < X_rot.size(); i++){
            int flag = 0;
            double a, b, c, d;

            double x = X_rot[i];

            if (x <= 0) {
                sigma_y[i] = -1;
                sigma_z[i] = -1;
            } else {
                if (stability_class == 'A') {
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
                } else if (stability_class == 'B') {
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
                } else if (stability_class == 'C') {
                    a = 61.141;
                    b = 0.91465;
                    c = 12.5000;
                    d = 1.0857;
                } else if (stability_class == 'D') {
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
                } else if (stability_class == 'E') {
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
                } else if (stability_class == 'F') {
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

                if (flag == 0) {
                    double Theta = 0.017453293 * (c - d * std::log(x)); // in radians
                    sigma_y[i] = 465.11628 * x * std::tan(Theta); // in meters
                    sigma_z[i] = a * std::pow(x, b); // in meters
                    sigma_z[i] = std::min(sigma_z[i], 5000.0);
                } else {
                    sigma_z[i] = 5000.0;
                }
            }
        }
    }

    const double deg_to_rad_factor = M_PI/180.0;

    // somewhat hacky way of making this act as an abstract function
    // okay with this for now since it's private and can't get called from python due to lack of bindings
    virtual std::vector<int> coarseSpatialThreshold(double, double){ throw std::logic_error("Not implemented"); }

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

        Vector X_shift = X.array() - x0;
        Vector Y_shift = Y.array() - y0;

        stackedGrid << X_shift.transpose(), Y_shift.transpose();

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
    std::vector<int> coarseSpatialThreshold(double wind_shift, double thresh_constant) override {

        std::vector<int> indices = getValidIndices(thresh_xy_max, thresh_z_max, wind_shift);

        if(!indices.empty()){
            // shrinks the thresholds
            double box_max_sig_y = sigma_y(indices).maxCoeff(); // max sigma of the current valid indices
            double box_max_sig_z = sigma_z(indices).maxCoeff();
            thresh_xy_max = box_max_sig_y*thresh_constant;
            thresh_z_max = box_max_sig_z*thresh_constant;
        }

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
        double norm_sq_Xrt = X0_rt.dot(X0_rt);

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
    std::vector<int> coarseSpatialThreshold(double wind_shift, double thresh_constant) override {
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