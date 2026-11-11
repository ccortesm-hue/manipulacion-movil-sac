"""Geometría de medición, sin modificar el controlador ni la simulación.

Tangente espacial de NEW_MM_VISUAL._build_frenet_path_cache: cuerda en una
ventana de 0.32 m, unwrap e interpolación en arco. No filtra yaw ni errores.
"""
import numpy as np


def wrap(angle):
    return (np.asarray(angle) + np.pi) % (2*np.pi) - np.pi


class RoutePoseMetric:
    def __init__(self, path):
        self.points = np.asarray(path, dtype=float).reshape(-1, 2).copy()
        if len(self.points) < 2 or not np.isfinite(self.points).all():
            raise ValueError('La métrica necesita una ruta finita con al menos dos puntos')
        self.seg = np.diff(self.points, axis=0)
        self.length = np.linalg.norm(self.seg, axis=1)
        if np.any(self.length <= 1e-10):
            raise ValueError('Ruta con segmentos degenerados')
        self.arc = np.r_[0., np.cumsum(np.maximum(self.length, 1e-6))]
        yaw = []
        for i, s in enumerate(self.arc):
            lo = int(np.clip(np.searchsorted(self.arc, s-.16, side='left'),0,len(self.points)-1))
            hi = int(np.clip(np.searchsorted(self.arc, s+.16, side='right')-1,0,len(self.points)-1))
            if hi <= lo:
                lo, hi = max(0,i-1), min(len(self.points)-1,i+1)
            chord = self.points[hi]-self.points[lo]
            if np.linalg.norm(chord) < 1e-9:
                chord = self.seg[min(i,len(self.seg)-1)]
            yaw.append(np.arctan2(chord[1],chord[0]))
        self.yaw = np.unwrap(yaw)

    def sample(self, xy, base_yaw):
        p = np.asarray(xy, dtype=float)
        if not np.isfinite(p).all() or not np.isfinite(base_yaw):
            raise ValueError('Pose no finita')
        delta = p-self.points[:-1]
        ratio = np.clip(np.sum(delta*self.seg,axis=1)/self.length**2,0,1)
        diff = delta-ratio[:,None]*self.seg
        best = int(np.argmin(np.sum(diff**2,axis=1)))
        s = self.arc[best]+ratio[best]*(self.arc[best+1]-self.arc[best])
        path_yaw = float(wrap(np.interp(s,self.arc,self.yaw)))
        cross = self.seg[best,0]*delta[best,1]-self.seg[best,1]*delta[best,0]
        signed = float(np.linalg.norm(diff[best])*np.sign(cross))
        return dict(path_yaw_rad=path_yaw, heading_error_rad=float(wrap(base_yaw-path_yaw)),
                    cte_signed_m=signed, route_progress=float(s/self.arc[-1]))
