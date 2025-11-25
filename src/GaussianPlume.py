import numpy as np
from FastGaussianPuff import interface_helpers as ih
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from numba import njit, prange


class GaussianPlume:

    def __init__(
        self,
        simulation_timestamp,
        time_zone,
        source_coordinates,
        emission_rate,
        wind_speed,
        wind_direction,
        X,
        Y,
        Z,
    ) -> None:
        try:
            time_zone = ZoneInfo(time_zone)
        except ZoneInfoNotFoundError:
            raise ValueError(f"Invalid timezone: {time_zone}")
        self.local_hour = ih.ensure_utc(simulation_timestamp).tz_convert(time_zone).hour

        self.source_coordinates = ih.parse_source_coords(source_coordinates)[0]
        self.emission_rate = emission_rate
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction
        self.X = X
        self.Y = Y
        self.Z = Z

        # this uses the usual conversion factor (see GaussianPuff) but needs the extra
        # factor of 3600 since emission rate is kg/hr and wind speed is m/s
        self.conversion_factor = 1e6 * 1.524 / 3600.0

    def simulate(self):
        stability_class = np.array(
            self.stabilityClassifier(self.wind_speed, self.local_hour)
        )
        theta = self.windDirectionToAngle(self.wind_direction)

        x0 = self.source_coordinates[0]
        y0 = self.source_coordinates[1]
        X_rot, Y_rot = self.rotate_points(self.X, self.Y, x0, y0, theta)

        stab_arr = np.array([ord(c) for c in stability_class], dtype=np.int32)
        sigmaY, sigmaZ = self.getSigmaCoefficients(stab_arr, X_rot)

        sigmaY = sigmaY.reshape(self.X.shape)
        sigmaZ = sigmaZ.reshape(self.X.shape)
        X_rot = X_rot.reshape(self.X.shape)
        Y_rot = Y_rot.reshape(self.Y.shape)

        concentration = self.GaussianPlume(
            X_rot,
            Y_rot,
            self.Z,
            self.emission_rate,
            self.wind_speed,
            self.source_coordinates[2],
            sigmaY,
            sigmaZ,
        )

        return concentration

    def GaussianPlume(self, X, Y, Z, Q, u, z0, sigmaY, sigmaZ):
        inds = np.where((sigmaY > 0) & (sigmaZ > 0) & (X > 0))
        if len(inds[0]) == 0:
            return np.zeros_like(X)

        C = np.zeros_like(X)
        C[inds] = (
            (Q / (2 * np.pi * u * sigmaY[inds] * sigmaZ[inds]))
            * np.exp(-0.5 * (Y[inds] / sigmaY[inds]) ** 2)
            * (
                np.exp(-0.5 * ((Z[inds] - z0) / sigmaZ[inds]) ** 2)
                + np.exp(-0.5 * ((Z[inds] + z0) / sigmaZ[inds]) ** 2)
            )
        )
        return C * self.conversion_factor

    def rotate_points(self, X, Y, x0, y0, theta):
        cosine = np.cos(theta)
        sine = np.sin(theta)
        Xs = X - x0
        Ys = Y - y0

        R = np.array([[cosine, -sine], [sine, cosine]])
        stacked = np.vstack((Xs.flatten(), Ys.flatten()))
        X_rot, Y_rot = R @ stacked

        return X_rot, Y_rot

    def windDirectionToAngle(self, wd):
        deg_to_rad_factor = np.pi / 180
        theta = wd - 270
        theta *= deg_to_rad_factor
        return theta

    def stabilityClassifier(self, wind_speed, hour, day_start=7, day_end=18):
        is_day = day_start <= hour <= day_end

        if wind_speed < 2:
            return ["A", "B"] if is_day else ["E", "F"]
        elif wind_speed < 3:
            return ["B"] if is_day else ["E", "F"]
        elif wind_speed < 5:
            return ["B", "C"] if is_day else ["D", "E"]
        elif wind_speed < 6:
            return ["C", "D"] if is_day else ["D"]
        else:
            return ["D"]

    @staticmethod
    @njit(parallel=True)
    def getSigmaCoefficients(stab_classes, downwind_dists):
        """
        stab_classes: 1D np.int32 array of ord('A')..ord('F')
        downwind_dists: 1D np.float64 array (meters)
        returns: sigmaY, sigmaZ as np.float64 arrays
        """
        n = downwind_dists.shape[0]
        nStab = stab_classes.shape[0]

        sigmaY = np.empty(n, dtype=np.float64)
        sigmaZ = np.empty(n, dtype=np.float64)

        for i in prange(n):
            x = downwind_dists[i] * 0.001  # km

            if x <= 0.0:
                sigmaY[i] = -1.0
                sigmaZ[i] = -1.0
                continue

            sumY = 0.0
            sumZ = 0.0

            for s in range(nStab):
                stab = stab_classes[s]

                a = 0.0
                b = 0.0
                c = 0.0
                d = 0.0
                flag = 0

                if stab == ord("A"):
                    if x < 0.1:
                        a, b = 122.800, 0.94470
                    elif x < 0.15:
                        a, b = 158.080, 1.05420
                    elif x < 0.20:
                        a, b = 170.220, 1.09320
                    elif x < 0.25:
                        a, b = 179.520, 1.12620
                    elif x < 0.30:
                        a, b = 217.410, 1.26440
                    elif x < 0.40:
                        a, b = 258.890, 1.40940
                    elif x < 0.50:
                        a, b = 346.750, 1.72830
                    elif x < 3.11:
                        a, b = 453.850, 2.11660
                    else:
                        flag = 1
                    c, d = 24.1670, 2.5334

                elif stab == ord("B"):
                    if x < 0.2:
                        a, b = 90.673, 0.93198
                    elif x < 0.4:
                        a, b = 98.483, 0.98332
                    else:
                        a, b = 109.300, 1.09710
                    c, d = 18.3330, 1.8096

                elif stab == ord("C"):
                    a, b = 61.141, 0.91465
                    c, d = 12.5, 1.0857

                elif stab == ord("D"):
                    if x < 0.3:
                        a, b = 34.459, 0.86974
                    elif x < 1.0:
                        a, b = 32.093, 0.81066
                    elif x < 3.0:
                        a, b = 32.093, 0.64403
                    elif x < 10.0:
                        a, b = 33.504, 0.60486
                    elif x < 30.0:
                        a, b = 36.650, 0.56589
                    else:
                        a, b = 44.053, 0.51179
                    c, d = 8.3330, 0.72382

                elif stab == ord("E"):
                    if x < 0.1:
                        a, b = 24.260, 0.83660
                    elif x < 0.3:
                        a, b = 23.331, 0.81956
                    elif x < 1.0:
                        a, b = 21.628, 0.75660
                    elif x < 2.0:
                        a, b = 21.628, 0.63077
                    elif x < 4.0:
                        a, b = 22.534, 0.57154
                    elif x < 10.0:
                        a, b = 24.703, 0.50527
                    elif x < 20.0:
                        a, b = 26.970, 0.46173
                    elif x < 40.0:
                        a, b = 35.420, 0.37615
                    else:
                        a, b = 47.618, 0.29592
                    c, d = 6.25, 0.54287

                elif stab == ord("F"):
                    if x < 0.2:
                        a, b = 15.209, 0.81558
                    elif x < 0.7:
                        a, b = 14.457, 0.78407
                    elif x < 1.0:
                        a, b = 13.953, 0.68465
                    elif x < 2.0:
                        a, b = 13.953, 0.63227
                    elif x < 3.0:
                        a, b = 14.823, 0.54503
                    elif x < 7.0:
                        a, b = 16.187, 0.46490
                    elif x < 15.0:
                        a, b = 17.836, 0.41507
                    elif x < 30.0:
                        a, b = 22.651, 0.32681
                    elif x < 60.0:
                        a, b = 27.074, 0.27436
                    else:
                        a, b = 34.219, 0.21716
                    c, d = 4.1667, 0.36191

                logx = np.log(x)
                theta = 0.017453293 * (c - d * logx)
                sigmaY_tmp = 465.11628 * x * np.tan(theta)

                if flag == 1:
                    sigmaZ_tmp = 5000.0
                else:
                    sigmaZ_tmp = a * np.exp(b * logx)
                    if sigmaZ_tmp > 5000.0:
                        sigmaZ_tmp = 5000.0

                sumY += sigmaY_tmp
                sumZ += sigmaZ_tmp

            sigmaY[i] = sumY / float(nStab)
            sigmaZ[i] = sumZ / float(nStab)

        return sigmaY, sigmaZ
