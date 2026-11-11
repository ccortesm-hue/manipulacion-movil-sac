"""Telemetría de P2 inferida de geometría/flags, sin cambiar el ambiente."""
import numpy as np


def p2_state(env):
    cube = env._cube_pos()
    xy = float(np.linalg.norm(cube[:2] - env._goal_xy()))
    z = abs(float(cube[2]) - float(env.Z_CUBE_REST))
    speed = float(np.linalg.norm(env._cube_vel_vec()))
    gates = (xy <= env.RELEASE_DIST_TOL_XY,
             z <= env.RELEASE_ERR_Z_MAX, speed <= env.RELEASE_V_MAX)
    # This is a diagnostic classification, not a policy or a controller FSM.
    if env._placed_success:
        stage = 'home'
    elif env._released:
        stage = 'settling'
    elif not gates[0]:
        stage = 'align_xy'
    elif not gates[1]:
        stage = 'descend'
    elif not gates[2]:
        stage = 'slow_down'
    else:
        stage = 'release_ready'
    return dict(xy_m=xy, z_m=z, speed_m_s=speed, gates=list(map(bool,gates)),
                simultaneous=bool(all(gates)), stage=stage,
                released=bool(env._released), placed=bool(env._placed_success))
