import numpy as np
import pybullet as p
import time

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
swarm_target_rpy = np.array([0, 0, 0])
obstacle_distance = 0.5  # Distance to detect obstacles

# Define the known obstacles' positions and sizes
obstacles = [
    {"pos": np.array([0, 2, 0.5]), "radius": 0.1},  # sphere2.urdf
]


def detect_obstacle(position, obstacles, new_obstacles=[]):
    """Detect if there is an obstacle within a certain distance."""
    for obstacle in obstacles:
        if np.linalg.norm(position - obstacle["pos"]) < (
            obstacle["radius"] + obstacle_distance
        ):
            return obstacle

    for obstacle in new_obstacles:
        if np.linalg.norm(position - obstacle["pos"]) < (
            obstacle["radius"] + obstacle_distance
        ):
            return obstacle

    return None


def get_formation_positions(num_drones, center, radius):
    """Calculate positions for the drones in a circle formation."""
    positions = []
    for i in range(num_drones):
        angle = (i / num_drones) * 2 * np.pi
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        z = center[2]
        positions.append([x, y, z])
    return np.array(positions)


def avoid_obstacle(position, obstacle):
    """Adjust the drone's position to avoid the obstacle."""
    direction = position - obstacle["pos"]
    direction = direction / np.linalg.norm(direction)  # Normalize the direction
    return obstacle["pos"] + direction * (obstacle["radius"] + obstacle_distance)


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
    center = np.array([0, 0, H])
    INIT_XYZS = get_formation_positions(DEFAULT_NUM_DRONES, center, R)
    INIT_RPYS = np.array([[0, 0, 0] for _ in range(DEFAULT_NUM_DRONES)])

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

    obstacle1 = create_sphere_obstacle(position=[1, 0, 0.5], radius=0.2)
    obstacle2 = create_sphere_obstacle(position=[-1, 1, 0.5], radius=0.2)

    obstacles.append({"pos": np.array([1, 0, 0.5]), "radius": 0.1})
    obstacles.append({"pos": np.array([-1, 1, 0.5]), "radius": 0.1})

    #### Initialize the logger #################################
    logger = Logger(
        logging_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        num_drones=DEFAULT_NUM_DRONES,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB,
    )

    #### Initialize the controllers ############################
    ctrl = [
        DSLPIDControl(drone_model=DEFAULT_DRONES) for _ in range(DEFAULT_NUM_DRONES)
    ]

    #### Run the simulation ####################################
    action = np.zeros((DEFAULT_NUM_DRONES, 4))
    START = time.time()

    while True:
        #### Step the simulation ###################################
        obs, _, _, _, _ = env.step(action)

        #### Compute control based on user input ###################
        keys = p.getKeyboardEvents()

        if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
            swarm_target_pos[1] += 0.01
        if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
            swarm_target_pos[1] -= 0.01
        if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
            swarm_target_pos[0] -= 0.01
        if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
            swarm_target_pos[0] += 0.01
        if ord("u") in keys and keys[ord("u")] & p.KEY_IS_DOWN:
            swarm_target_pos[2] += 0.01
        if ord("j") in keys and keys[ord("j")] & p.KEY_IS_DOWN:
            swarm_target_pos[2] -= 0.01

        positions = INIT_XYZS + swarm_target_pos

        #### Update obstacle positions to follow the swarm ##########
        # avg_position = np.mean(positions, axis=0)
        current_pos1, _ = p.getBasePositionAndOrientation(obstacle1)
        current_pos2, _ = p.getBasePositionAndOrientation(obstacle2)

        new_pos1 = (
            current_pos1 + (positions[0] - current_pos1) * 0.01
        )  # Adjust the speed factor as needed
        new_pos2 = (
            current_pos2 + (positions[0] - current_pos2) * 0.01
        )  # Adjust the speed factor as needed

        p.resetBasePositionAndOrientation(obstacle1, new_pos1, [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(obstacle2, new_pos2, [0, 0, 0, 1])

        for j in range(DEFAULT_NUM_DRONES):
            # Check for obstacles and avoid
            obstacle = detect_obstacle(
                positions[j],
                obstacles,
                new_obstacles=[
                    {
                        "pos": np.array(new_pos1),
                        "radius": 0.1,
                    },
                    {
                        "pos": np.array(new_pos2),
                        "radius": 0.1,
                    },
                ],
            )
            if obstacle:
                positions[j] = avoid_obstacle(positions[j], obstacle)

            action[j, :], _, _ = ctrl[j].computeControlFromState(
                control_timestep=env.CTRL_TIMESTEP,
                state=obs[j],
                target_pos=positions[j],
                target_rpy=swarm_target_rpy,
            )

        #### Log the simulation ####################################
        for j in range(DEFAULT_NUM_DRONES):
            logger.log(
                drone=j,
                timestamp=time.time() - START,
                state=obs[j],
                control=np.hstack([positions[j], swarm_target_rpy, np.zeros(6)]),
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
