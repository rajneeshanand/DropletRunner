#!/usr/bin/env python3

#OCamCalib camera model for DropletRunner.
#Implements Scaramuzza's omnidirectional camera model using calibration data from calib_results.txt.

#Two main functions: 1. cam2world(pixel) for 3D ray direction 2. world2cam(3D_point) for pixel coordinates

import numpy as np
import os


class OCamModel:
    def __init__(self, calib_file=None):
                    
        if calib_file is None:
            possible = [
                os.path.join(os.path.dirname(__file__), '..', 'calib', 'calib_results.txt'),
                os.path.expanduser('~/droplet_runner_ws/src/droplet_state/calib/calib_results.txt'),
            ]
            calib_file = None
            for p in possible:
                if os.path.exists(p):
                    calib_file = p
                    break
            if calib_file is None:
                raise FileNotFoundError(f'calib_results.txt not found')

        self._parse_calib(calib_file)

        #Calibration was done at 1024x768 but Camera runs at 1280x720
        self.calib_width = 1024
        self.calib_height = 768
        self.cam_width = 1280
        self.cam_height = 720

        self.sx = self.cam_width / self.calib_width    # 1.25,    # Scale factors
        self.sy = self.cam_height / self.calib_height   # 0.9375

        self.cx_scaled = self.cx * self.sx        #scaling of center point to camera resolution
        self.cy_scaled = self.cy * self.sy

    def _parse_calib(self, filepath):
        with open(filepath, 'r') as f:
            lines = f.readlines()

        data_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                data_lines.append(line)

        #direct mapping polynomial (cam2world)
        parts = data_lines[0].split()
        n_ss = int(parts[0])
        self.ss = np.array([float(x) for x in parts[1:n_ss+1]])

        #inverse mapping polynomial (world2cam)
        parts = data_lines[1].split()
        n_invpol = int(parts[0])
        self.invpol = np.array([float(x) for x in parts[1:n_invpol+1]])

        #center (row, col) where row=y, col=x
        parts = data_lines[2].split()
        self.cy = float(parts[0])  # row = y
        self.cx = float(parts[1])  # col = x

        #affine parameters c, d, e
        parts = data_lines[3].split()
        self.c = float(parts[0])
        self.d = float(parts[1])
        self.e = float(parts[2])

        #image size (height, width)
        parts = data_lines[4].split()
        self.height = int(parts[0])
        self.width = int(parts[1])

    def cam2world(self, pixel_x, pixel_y):           #Convert pixel coordinates to 3D ray direction.
        x = pixel_x / self.sx
        y = pixel_y / self.sy

        xc = x - self.cx
        yc = y - self.cy

        det = self.c - self.d * self.e
        xp = (xc - self.d * yc) / det
        yp = (-self.e * xc + self.c * yc) / det

        rho = np.sqrt(xp**2 + yp**2)

        z = np.polyval(self.ss[::-1], rho)

        norm = np.sqrt(xp**2 + yp**2 + z**2)
        if norm < 1e-10:
            return np.array([0.0, 0.0, 1.0])

        return np.array([xp / norm, yp / norm, z / norm])

    def world2cam(self, point_3d):              #project 3D point to pixel coordinates.
        X, Y, Z = point_3d
        norm_xy = np.sqrt(X**2 + Y**2)

        if norm_xy < 1e-10:
            # Point on optical axis
            return self.cx_scaled, self.cy_scaled

        theta = np.arctan2(-Z, norm_xy)

        #evaluate inverse polynomial
        rho = np.polyval(self.invpol[::-1], theta)

        xp = X / norm_xy * rho
        yp = Y / norm_xy * rho

        x_calib = xp * self.c + yp * self.d + self.cx
        y_calib = xp * self.e + yp + self.cy

        pixel_x = x_calib * self.sx
        pixel_y = y_calib * self.sy

        return pixel_x, pixel_y

    def undistort_point(self, pixel_x, pixel_y, z_plane=0.0):

        #Undistort a pixel coordinate by projecting through the camera
        #model and intersecting with a plane at height z_plane. This corrects lens distortion for points on the board surface.
        
        ray = self.cam2world(pixel_x, pixel_y)

        if abs(ray[2]) < 1e-10:
            return pixel_x, pixel_y  # fallback

        # Intersect ray with z=z_plane and Parametric: P = t * ray, find t where P_z = z_plane
        # For z_plane=0, we just use the x,y components of the ray, normalized by z
        corrected_x = ray[0] / ray[2]
        corrected_y = ray[1] / ray[2]

        return corrected_x, corrected_y


if __name__ == '__main__':
    model = OCamModel()
    print(f"Direct poly (ss): {model.ss}")
    print(f"Inverse poly (invpol): {model.invpol}")
    print(f"Center (calib): ({model.cx:.1f}, {model.cy:.1f})")
    print(f"Center (scaled): ({model.cx_scaled:.1f}, {model.cy_scaled:.1f})")
    print(f"Affine: c={model.c}, d={model.d}, e={model.e}")
    print(f"Scale: sx={model.sx}, sy={model.sy}")

    ray = model.cam2world(model.cx_scaled, model.cy_scaled)
    print(f"\nCenter ray: {ray}")

    for name, px, py in [
        ("TL", 341, 130), ("TR", 926, 126),
        ("BR", 906, 650), ("BL", 345, 630)]:
        ray = model.cam2world(px, py)
        cx, cy = model.undistort_point(px, py)
        print(f"{name} ({px},{py}): ray={ray}, undistorted=({cx:.4f},{cy:.4f})")


