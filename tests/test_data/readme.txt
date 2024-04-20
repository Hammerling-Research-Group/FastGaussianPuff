Data was generated with original python code.

Until I know if we can keep some data in the repository, data can be found in dropbox under fastGaussianPuff/data.

NEW uses the plume-centered sigmas
OLD uses the whole grid sigmas

Filename convention: n-N-exp-DATE.csv
    N: number of points. All computed on cube grids, so N=nx*ny*nz=nx^3
    DATE: start time of the METEC experiment
    sim: sim_dt that was used
    puff: puff_dt that was used
    
N=750 was generated with 
    x_num = 10
    y_num = 15
    z_num = 5
to test non-square grids
