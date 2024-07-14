import numpy as np
import pybullet as p
import time
import random

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 10
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 300  # 5 minutes
DEFAULT_OUTPUT_FOLDER = "results"
DEFAULT_COLAB = False

# Initialize global variables for drone control
swarm_target_pos = np.array([0, 0, 0.1])
collision_distance = 0.2  # Minimum distance to maintain between drones
min_height = 0.2  # Minimum height above the ground


def avoid_collisions(target_pos, drone_index, positions):
    """Adjust the target position to avoid collisions with other drones and the ground."""
    adjusted_pos = target_pos.copy()
    for i, pos in enumerate(positions):
        if i != drone_index:
            distance = np.linalg.norm(adjusted_pos - pos)
            if distance < collision_distance and distance != 0:
                direction = (adjusted_pos - pos) / distance
                adjusted_pos = pos + direction * collision_distance

    # Ensure the drone stays above the minimum height
    if adjusted_pos[2] < min_height:
        adjusted_pos[2] = min_height

    return adjusted_pos


def create_sphere_obstacle(position, radius, color=[1, 0, 0, 1]):
    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_SPHERE, radius=radius, rgbaColor=color
    )
    collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=radius)
    obstacle_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_shape_id,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=position,
    )
    return obstacle_id


def run_simulation():
    #### Initialize the simulation #############################
    H = 0.1
    R = 0.3
    INIT_XYZS = np.array(
        [
            [
                R * np.cos((i / 10) * 2 * np.pi + np.pi / 2)
                + np.random.uniform(-0.01, 0.01),
                R * np.sin((i / 10) * 2 * np.pi + np.pi / 2)
                - R
                + np.random.uniform(-0.01, 0.01),
                H,
            ]
            for i in range(DEFAULT_NUM_DRONES)
        ]
    )
    INIT_RPYS = np.array(
        [
            [0, 0, i * (np.pi / 2) / DEFAULT_NUM_DRONES]
            for i in range(DEFAULT_NUM_DRONES)
        ]
    )

    #### Create the environment ################################
    env = CtrlAviary(
        drone_model=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        initial_xyzs=INIT_XYZS,
        initial_rpys=INIT_RPYS,
        physics=DEFAULT_PHYSICS,
        neighbourhood_radius=10,
        pyb_freq=DEFAULT_SIMULATION_FREQ_HZ,
        ctrl_freq=DEFAULT_CONTROL_FREQ_HZ,
        gui=DEFAULT_GUI,
        record=DEFAULT_RECORD_VISION,
        obstacles=DEFAULT_OBSTACLES,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
    )

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Create spherical obstacles ############################
    obstacle1 = create_sphere_obstacle(position=[1, 0, 0.5], radius=0.2)
    obstacle2 = create_sphere_obstacle(position=[-1, 1, 0.5], radius=0.2)

    #### Initialize the logger #################################
    logger = Logger(
        logging_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        num_drones=DEFAULT_NUM_DRONES,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB,
    )

    #### Initialize the controllers ############################
    if DEFAULT_DRONES in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = [
            DSLPIDControl(drone_model=DEFAULT_DRONES) for i in range(DEFAULT_NUM_DRONES)
        ]

    #### Run the simulation ####################################
    action = np.zeros((DEFAULT_NUM_DRONES, 4))
    START = time.time()

    while True:
        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)

        #### Compute control based on user input ###################
        keys = p.getKeyboardEvents()

        if ord("w") in keys and keys[ord("w")] & p.KEY_IS_DOWN:
            swarm_target_pos[1] += 0.02
        if ord("s") in keys and keys[ord("s")] & p.KEY_IS_DOWN:
            swarm_target_pos[1] -= 0.02
        if ord("a") in keys and keys[ord("a")] & p.KEY_IS_DOWN:
            swarm_target_pos[0] -= 0.02
        if ord("d") in keys and keys[ord("d")] & p.KEY_IS_DOWN:
            swarm_target_pos[0] += 0.02
        if ord("u") in keys and keys[ord("u")] & p.KEY_IS_DOWN:
            swarm_target_pos[2] += 0.02
        if ord("j") in keys and keys[ord("j")] & p.KEY_IS_DOWN:
            swarm_target_pos[2] -= 0.02

        positions = INIT_XYZS + swarm_target_pos

        for j in range(DEFAULT_NUM_DRONES):
            target_pos = avoid_collisions(positions[j], j, positions)

            action[j, :], _, _ = ctrl[j].computeControlFromState(
                control_timestep=env.CTRL_TIMESTEP,
                state=obs[j],
                target_pos=np.array(target_pos),
                target_rpy=INIT_RPYS[j, :],
            )

        #### Log the simulation ####################################
        for j in range(DEFAULT_NUM_DRONES):
            logger.log(
                drone=j,
                timestamp=time.time() - START,
                state=obs[j],
                control=np.hstack([target_pos, INIT_RPYS[j, :], np.zeros(6)]),
            )

        #### Printout and Camera Tracking ##########################
        env.render()

        # Update the camera position to track the swarm
        avg_position = np.mean(positions, axis=0)
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=0,
            cameraPitch=-30,
            cameraTargetPosition=avg_position,
            physicsClientId=PYB_CLIENT,
        )

        #### Sync the simulation ###################################
        sync(
            int((time.time() - START) * DEFAULT_CONTROL_FREQ_HZ),
            START,
            env.CTRL_TIMESTEP,
        )

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    logger.save()
    logger.save_as_csv("pid")  # Optional CSV save

    #### Plot the simulation results ###########################
    if DEFAULT_PLOT:
        logger.plot()


if __name__ == "__main__":
    run_simulation()
