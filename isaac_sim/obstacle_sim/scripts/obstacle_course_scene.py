#!/usr/bin/env python3
"""
Obstacle Course Scene for Curiosity Rover in Isaac Sim.
Creates a simple plane with walls and scattered obstacles for navigation testing.
"""

import signal
import sys
import os
import random

# Isaac Sim application setup
from isaacsim import SimulationApp

kit = SimulationApp({
    "renderer": "RayTracedLighting",
    "headless": False,
})

# Enable required extensions
import omni.kit.app
manager = omni.kit.app.get_app().get_extension_manager()
manager.set_extension_enabled_immediate("omni.isaac.ros2_bridge", True)

import carb
import numpy as np
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics, PhysxSchema, Sdf, UsdShade

import omni
import omni.graph.core as og
import omni.replicator.core as rep
import omni.usd
from omni.isaac.core import World
from omni.isaac.core.prims import XFormPrim, GeometryPrim
from omni.isaac.core.utils.prims import create_prim, define_prim
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.materials import PreviewSurface
from omni.isaac.nucleus import get_assets_root_path

# Configuration
ROVER_USD_PATH = "/home/spaceros-user/curiosity_sim/models/curiosity_mars_rover/curiosity_mars_rover.usd"
ROVER_START_POS = (0.0, 0.0, 0.5)  # Start position on the ground plane
GROUND_SIZE = 50.0  # 50m x 50m arena
WALL_HEIGHT = 2.0
WALL_THICKNESS = 0.3

# Physics settings
GRAVITY = 3.71  # Mars gravity m/s^2
PHYSICS_DT = 1.0 / 100.0  # 100 Hz physics


def create_ground_plane(stage):
    """Create a large ground plane with physics collision."""
    ground_path = "/World/GroundPlane"

    # Create the ground plane geometry
    ground_prim = UsdGeom.Mesh.Define(stage, ground_path)
    half_size = GROUND_SIZE / 2.0

    # Define vertices for a simple quad
    points = [
        (-half_size, -half_size, 0),
        (half_size, -half_size, 0),
        (half_size, half_size, 0),
        (-half_size, half_size, 0),
    ]
    ground_prim.GetPointsAttr().Set(points)
    ground_prim.GetFaceVertexCountsAttr().Set([4])
    ground_prim.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
    ground_prim.GetNormalsAttr().Set([(0, 0, 1)] * 4)

    # Add collision
    UsdPhysics.CollisionAPI.Apply(ground_prim.GetPrim())

    # Add physics material for friction
    physics_material_path = "/World/Materials/GroundMaterial"
    UsdShade.Material.Define(stage, physics_material_path)
    material = UsdPhysics.MaterialAPI.Apply(stage.GetPrimAtPath(physics_material_path))
    material.CreateStaticFrictionAttr(0.8)
    material.CreateDynamicFrictionAttr(0.6)
    material.CreateRestitutionAttr(0.1)

    # Apply material to ground
    binding_api = UsdShade.MaterialBindingAPI.Apply(ground_prim.GetPrim())
    binding_api.Bind(UsdShade.Material(stage.GetPrimAtPath(physics_material_path)))

    print(f"[INFO] Created ground plane: {GROUND_SIZE}m x {GROUND_SIZE}m")
    return ground_prim


def create_box_obstacle(stage, name, position, size, color=(0.5, 0.5, 0.5)):
    """Create a box obstacle with physics collision."""
    path = f"/World/Obstacles/{name}"

    # Create cube
    cube = UsdGeom.Cube.Define(stage, path)
    cube.GetSizeAttr().Set(1.0)

    # Set transform
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    xform.AddScaleOp().Set(Gf.Vec3d(*size))

    # Add collision
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
    rigid_body = UsdPhysics.RigidBodyAPI(cube.GetPrim())
    rigid_body.CreateKinematicEnabledAttr(True)  # Static obstacle

    # Set color
    cube.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])

    return cube


def create_cylinder_obstacle(stage, name, position, radius, height, color=(0.6, 0.4, 0.2)):
    """Create a cylindrical obstacle with physics collision."""
    path = f"/World/Obstacles/{name}"

    # Create cylinder
    cylinder = UsdGeom.Cylinder.Define(stage, path)
    cylinder.GetRadiusAttr().Set(radius)
    cylinder.GetHeightAttr().Set(height)
    cylinder.GetAxisAttr().Set("Z")

    # Set transform (position is at base)
    xform = UsdGeom.Xformable(cylinder.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(position[0], position[1], position[2] + height/2))

    # Add collision
    UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(cylinder.GetPrim())
    rigid_body = UsdPhysics.RigidBodyAPI(cylinder.GetPrim())
    rigid_body.CreateKinematicEnabledAttr(True)  # Static obstacle

    # Set color
    cylinder.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])

    return cylinder


def create_wall(stage, name, start_pos, end_pos, height=WALL_HEIGHT, thickness=WALL_THICKNESS):
    """Create a wall segment between two points."""
    path = f"/World/Walls/{name}"

    # Calculate wall dimensions
    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]
    length = np.sqrt(dx*dx + dy*dy)
    angle = np.arctan2(dy, dx)

    center_x = (start_pos[0] + end_pos[0]) / 2.0
    center_y = (start_pos[1] + end_pos[1]) / 2.0
    center_z = height / 2.0

    # Create cube for wall
    cube = UsdGeom.Cube.Define(stage, path)
    cube.GetSizeAttr().Set(1.0)

    # Set transform
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(center_x, center_y, center_z))
    xform.AddRotateZOp().Set(np.degrees(angle))
    xform.AddScaleOp().Set(Gf.Vec3d(length, thickness, height))

    # Add collision
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
    rigid_body = UsdPhysics.RigidBodyAPI(cube.GetPrim())
    rigid_body.CreateKinematicEnabledAttr(True)

    # Wall color (gray concrete)
    cube.GetDisplayColorAttr().Set([Gf.Vec3f(0.6, 0.6, 0.6)])

    return cube


def create_perimeter_walls(stage):
    """Create walls around the arena perimeter."""
    half_size = GROUND_SIZE / 2.0

    # Four walls around the perimeter
    walls = [
        ("wall_north", (-half_size, half_size), (half_size, half_size)),
        ("wall_south", (-half_size, -half_size), (half_size, -half_size)),
        ("wall_east", (half_size, -half_size), (half_size, half_size)),
        ("wall_west", (-half_size, -half_size), (-half_size, half_size)),
    ]

    for name, start, end in walls:
        create_wall(stage, name, start, end)

    print(f"[INFO] Created perimeter walls")


def create_interior_walls(stage):
    """Create interior wall segments for maze-like navigation."""
    interior_walls = [
        # L-shaped barrier
        ("interior_1a", (-15, -10), (-15, 10)),
        ("interior_1b", (-15, 10), (-5, 10)),

        # Parallel walls creating corridor
        ("interior_2a", (5, -20), (5, -5)),
        ("interior_2b", (10, -20), (10, -5)),

        # Angled wall
        ("interior_3", (15, 5), (20, 15)),

        # Short barriers
        ("interior_4", (-5, -15), (5, -15)),
        ("interior_5", (0, 15), (15, 15)),
    ]

    for name, start, end in interior_walls:
        create_wall(stage, name, start, end, height=1.5)

    print(f"[INFO] Created interior walls")


def create_scattered_obstacles(stage, num_boxes=15, num_cylinders=10):
    """Create randomly scattered obstacles."""
    random.seed(42)  # For reproducibility
    half_size = GROUND_SIZE / 2.0 - 3.0  # Keep away from walls

    # Create boxes of various sizes
    for i in range(num_boxes):
        # Random position (avoid center where rover starts)
        while True:
            x = random.uniform(-half_size, half_size)
            y = random.uniform(-half_size, half_size)
            if abs(x) > 5.0 or abs(y) > 5.0:  # Clear zone around start
                break

        # Random size
        sx = random.uniform(0.5, 2.0)
        sy = random.uniform(0.5, 2.0)
        sz = random.uniform(0.3, 1.5)

        # Random earthy color
        color = (
            random.uniform(0.3, 0.7),
            random.uniform(0.2, 0.5),
            random.uniform(0.1, 0.3)
        )

        create_box_obstacle(stage, f"box_{i}", (x, y, sz/2), (sx, sy, sz), color)

    # Create cylinders (pillars)
    for i in range(num_cylinders):
        while True:
            x = random.uniform(-half_size, half_size)
            y = random.uniform(-half_size, half_size)
            if abs(x) > 5.0 or abs(y) > 5.0:
                break

        radius = random.uniform(0.3, 1.0)
        height = random.uniform(0.5, 2.5)

        color = (
            random.uniform(0.4, 0.6),
            random.uniform(0.3, 0.5),
            random.uniform(0.2, 0.4)
        )

        create_cylinder_obstacle(stage, f"cylinder_{i}", (x, y, 0), radius, height, color)

    print(f"[INFO] Created {num_boxes} boxes and {num_cylinders} cylinders")


def setup_lighting(stage):
    """Set up scene lighting."""
    # Distant light (sun)
    light_path = "/World/Lights/Sun"
    light = UsdLux.DistantLight.Define(stage, light_path)
    light.GetIntensityAttr().Set(3000)
    light.GetColorAttr().Set(Gf.Vec3f(1.0, 0.95, 0.8))

    xform = UsdGeom.Xformable(light.GetPrim())
    xform.AddRotateXYZOp().Set(Gf.Vec3f(-45, 30, 0))

    # Dome light for ambient
    dome_path = "/World/Lights/DomeLight"
    dome = UsdLux.DomeLight.Define(stage, dome_path)
    dome.GetIntensityAttr().Set(500)

    print("[INFO] Lighting configured")


def setup_physics(stage):
    """Configure physics scene settings."""
    scene_path = "/physicsScene"
    scene = UsdPhysics.Scene.Define(stage, scene_path)
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(GRAVITY)

    # PhysX settings
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    physx_scene.CreateTimeStepsPerSecondAttr().Set(int(1.0 / PHYSICS_DT))
    physx_scene.CreateEnableCCDAttr().Set(True)
    physx_scene.CreateEnableGPUDynamicsAttr().Set(False)

    print(f"[INFO] Physics configured: gravity={GRAVITY} m/s², dt={PHYSICS_DT}")


def load_curiosity_rover(stage):
    """Load the Curiosity rover model."""
    rover_prim_path = "/World/CuriosityRover"

    if not os.path.exists(ROVER_USD_PATH):
        print(f"[ERROR] Rover USD not found at: {ROVER_USD_PATH}")
        return None

    add_reference_to_stage(ROVER_USD_PATH, rover_prim_path)

    # Set initial position
    rover_prim = stage.GetPrimAtPath(rover_prim_path)
    xform = UsdGeom.Xformable(rover_prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*ROVER_START_POS))

    print(f"[INFO] Loaded Curiosity rover at position {ROVER_START_POS}")
    return rover_prim


def setup_ros2_graphs():
    """Set up ROS2 action graphs for rover control and sensors."""

    # ROS2 Joint Control Graph
    try:
        (ros_control_graph, _, _, _) = og.Controller.edit(
            {"graph_path": "/World/CuriosityRover/ROS_ControlGraph", "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnPhysicsStep", "omni.isaac.core_nodes.OnPhysicsStep"),
                    ("ROS2Context", "omni.isaac.ros2_bridge.ROS2Context"),
                    ("PublishJointState", "omni.isaac.ros2_bridge.ROS2PublishJointState"),
                    ("SubscribeJointState", "omni.isaac.ros2_bridge.ROS2SubscribeJointState"),
                    ("ArticulationController", "omni.isaac.core_nodes.IsaacArticulationController"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("ROS2Context.inputs:domain_id", 0),
                    ("PublishJointState.inputs:topicName", "/curiosity_mars_rover/joint_states"),
                    ("SubscribeJointState.inputs:topicName", "/curiosity_mars_rover/joint_command"),
                    ("ArticulationController.inputs:robotPath", "/World/CuriosityRover"),
                    ("ArticulationController.inputs:usePath", True),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnPhysicsStep.outputs:step", "PublishJointState.inputs:execIn"),
                    ("OnPhysicsStep.outputs:step", "SubscribeJointState.inputs:execIn"),
                    ("OnPhysicsStep.outputs:step", "ArticulationController.inputs:execIn"),
                    ("ROS2Context.outputs:context", "PublishJointState.inputs:context"),
                    ("ROS2Context.outputs:context", "SubscribeJointState.inputs:context"),
                    ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
                    ("SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
                    ("SubscribeJointState.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
                    ("SubscribeJointState.outputs:effortCommand", "ArticulationController.inputs:effortCommand"),
                ],
            },
        )
        print("[INFO] ROS2 Control Graph created")
    except Exception as e:
        print(f"[ERROR] Failed to create control graph: {e}")
        return False

    # ROS2 Clock Graph
    try:
        og.Controller.edit(
            {"graph_path": "/World/ROS_ClockGraph", "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("ROS2Context", "omni.isaac.ros2_bridge.ROS2Context"),
                    ("PublishClock", "omni.isaac.ros2_bridge.ROS2PublishClock"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("ROS2Context.inputs:domain_id", 0),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                    ("ROS2Context.outputs:context", "PublishClock.inputs:context"),
                ],
            },
        )
        print("[INFO] ROS2 Clock Graph created")
    except Exception as e:
        print(f"[ERROR] Failed to create clock graph: {e}")

    return True


def main():
    """Main function to set up the obstacle course scene."""
    print("\n" + "="*60)
    print("  Curiosity Rover - Obstacle Course Scene")
    print("="*60 + "\n")

    # Get stage
    stage = omni.usd.get_context().get_stage()

    # Set up root prim
    root_prim = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(root_prim)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # Create scene hierarchy
    define_prim("/World/Obstacles", "Xform")
    define_prim("/World/Walls", "Xform")
    define_prim("/World/Lights", "Xform")

    # Set up physics
    setup_physics(stage)

    # Create environment
    create_ground_plane(stage)
    create_perimeter_walls(stage)
    create_interior_walls(stage)
    create_scattered_obstacles(stage)
    setup_lighting(stage)

    # Load rover
    load_curiosity_rover(stage)

    # Set up ROS2 graphs
    setup_ros2_graphs()

    print("\n[INFO] Scene setup complete!")
    print("[INFO] Press SPACE to start simulation, ESC to exit\n")

    # Signal handler for graceful shutdown
    shutdown_requested = False

    def signal_handler(sig, frame):
        nonlocal shutdown_requested
        print("\n[INFO] Shutdown requested...")
        shutdown_requested = True

    signal.signal(signal.SIGINT, signal_handler)

    # Main simulation loop
    while not shutdown_requested:
        kit.update()

    print("[INFO] Shutting down...")
    kit.close()


if __name__ == "__main__":
    main()
