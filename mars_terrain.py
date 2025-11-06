# mars_terrain.py
"""
Terrain generation utilities for the Mars-like environment.

Provides:
 - get_height_at(x, y, heightfieldData): bilinear interpolation from the heightfield.
 - create_terrain(): constructs a procedural heightfield with Gaussian-shaped hills.
 - add_rocks(): scatter mesh rocks on the terrain (optional).
"""

import os
import random
import math
import pybullet as p

numHeightfieldRows = 512
numHeightfieldColumns = 512


def get_height_at(x_world, y_world, heightfieldData):
    """
    Return interpolated height at a world coordinate (x_world, y_world)
    via bilinear interpolation on the heightfield grid.
    """
    cell_size_x = 0.05
    cell_size_y = 0.05
    scale_z = 1.0

    # Convert world coordinates to grid indices
    x_index = (x_world + (numHeightfieldRows * cell_size_x / 2.0)) / cell_size_x
    y_index = (y_world + (numHeightfieldColumns * cell_size_y / 2.0)) / cell_size_y

    # Clamp indices to valid range
    x_index = max(0.0, min(numHeightfieldRows - 1.001, x_index))
    y_index = max(0.0, min(numHeightfieldColumns - 1.001, y_index))

    x0 = int(x_index)
    x1 = min(x0 + 1, numHeightfieldRows - 1)
    y0 = int(y_index)
    y1 = min(y0 + 1, numHeightfieldColumns - 1)

    # Fetch heights
    hx0y0 = heightfieldData[y0 * numHeightfieldRows + x0]
    hx1y0 = heightfieldData[y0 * numHeightfieldRows + x1]
    hx0y1 = heightfieldData[y1 * numHeightfieldRows + x0]
    hx1y1 = heightfieldData[y1 * numHeightfieldRows + x1]

    sx = x_index - x0
    sy = y_index - y0

    # Bilinear interpolation
    h = (hx0y0 * (1 - sx) * (1 - sy) +
         hx1y0 * sx * (1 - sy) +
         hx0y1 * (1 - sx) * sy +
         hx1y1 * sx * sy)
    return h * scale_z


def create_terrain():
    """
    Create a procedurally generated heightfield with Gaussian hills, texture, and perimeter walls.
    Returns (wall_ids, heightfieldData).
    """
    heightfieldData = [0.0] * (numHeightfieldRows * numHeightfieldColumns)

    # Random hills
    num_hills = random.randint(20, 100)
    min_hill_radius = 20
    max_hill_radius = 100

    hill_centers = []
    for _ in range(num_hills):
        radius = random.randint(min_hill_radius, max_hill_radius)
        max_height = random.uniform(0.05, 0.25)
        sigma = radius / 2.5

        # Ensure hills are within safe bounds away from edges
        cx = random.randint(radius, numHeightfieldRows - radius - 1)
        cy = random.randint(radius, numHeightfieldColumns - radius - 1)
        hill_centers.append((cx, cy, radius, max_height, sigma))

    # Apply Gaussian hills
    for cx, cy, radius, max_height, sigma in hill_centers:
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                x = cx + dx
                y = cy + dy
                if 0 <= x < numHeightfieldRows and 0 <= y < numHeightfieldColumns:
                    distance = math.sqrt(dx ** 2 + dy ** 2)
                    if distance <= radius:
                        height = max_height * math.exp(- (distance ** 2) / (2.0 * sigma ** 2))
                        index = y * numHeightfieldRows + x
                        heightfieldData[index] += height

    # Create PyBullet heightfield collision shape and multi-body
    terrainShape = p.createCollisionShape(
        shapeType=p.GEOM_HEIGHTFIELD,
        meshScale=[0.05, 0.05, 1.0],
        heightfieldTextureScaling=1,
        heightfieldData=heightfieldData,
        numHeightfieldRows=numHeightfieldRows,
        numHeightfieldColumns=numHeightfieldColumns
    )
    terrain = p.createMultiBody(0, terrainShape)
    p.resetBasePositionAndOrientation(terrain, [0, 0, 0], [0, 0, 0, 1])

    # Set friction for the terrain
    p.changeDynamics(terrain, -1, lateralFriction=1.0)

    # Add perimeter walls to contain the rover
    wall_height = 5.0
    wall_thickness = 0.1
    wall_length = numHeightfieldRows * 0.05
    wall_half = wall_length / 2.0

    wall_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_half, wall_thickness / 2.0, wall_height / 2.0])
    visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[wall_half, wall_thickness / 2.0, wall_height / 2.0],
                                       rgbaColor=[0.2, 0.2, 0.2, 1.0])

    wall_ids = []
    walls = [
        ([0, wall_half, 0.5 - wall_height / 2.0], 0.0),
        ([0, -wall_half, 0.5 - wall_height / 2.0], 0.0),
        ([wall_half, 0, 0.5 - wall_height / 2.0], 1.5708),
        ([-wall_half, 0, 0.5 - wall_height / 2.0], 1.5708)
    ]
    for pos, yaw in walls:
        orn = p.getQuaternionFromEuler([0, 0, yaw])
        wall_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=wall_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=pos,
            baseOrientation=orn
        )
        wall_ids.append(wall_id)

    # Apply texture if available
    try:
        textureId = p.loadTexture("MarsTexture.png")
        p.changeVisualShape(terrain, -1, textureUniqueId=textureId)
    except Exception:
        # texture optional
        pass

    # Set physics properties & timestep
    p.setGravity(0, 0, -9.71)
    p.setTimeStep(1.0 / 120.0)
    p.setRealTimeSimulation(1)

    return wall_ids, heightfieldData


def add_rocks(heightfieldData):
    """
    Scatter rock meshes over the terrain for additional irregularity.
    Expects a "Rocks" folder with mesh files. May raise an error if none found.
    """
    folder_path = "Rocks"
    if not os.path.isdir(folder_path):
        return []

    min_rocks = 10
    max_rocks = 35
    num_rocks = random.randint(min_rocks, max_rocks)
    max_scale = 0.7
    min_scale = 0.005

    rock_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".obj")]
    if not rock_files:
        return []

    rock_ids = []
    terrain_width = numHeightfieldRows * 0.05
    terrain_length = numHeightfieldColumns * 0.05

    # Example texture folder: adjust as needed or set to None
    texture_folder = None

    for _ in range(num_rocks):
        stl_file = random.choice(rock_files)
        full_path = os.path.join(folder_path, stl_file)
        scale = random.uniform(min_scale, max_scale)

        # Choose a random position away from center to avoid robot spawn collisions
        while True:
            x = random.uniform((-terrain_width / 2.0) + 1.5, (terrain_width / 2.0) - 1.5)
            y = random.uniform((-terrain_length / 2.0) + 1.5, (terrain_length / 2.0) - 1.5)
            if math.hypot(x, y) >= 3.0:
                break

        z = get_height_at(x, y, heightfieldData) + (scale * 0.5)

        col_shape = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=full_path,
            meshScale=[scale]*3
        )

        vis_shape = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=full_path,
            meshScale=[scale]*3
        )

        rock_id = p.createMultiBody(
            baseMass=0.00001,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=[x, y, z]
        )

        p.changeDynamics(rock_id, -1, mass=10.0*scale, lateralFriction=1.5, spinningFriction=0.5, rollingFriction=0.5, restitution=0.0)
        rock_ids.append(rock_id)

    return rock_ids
