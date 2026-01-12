import math
import numpy as np
from shapely.geometry import Polygon
from mathutils import Vector, Euler
import settings  # 引用全局配置

class Heightmap:
    def __init__(self):
        self.height = {}
        self.height_list = []
    
    def update_heightmap(self, position, size, rotation):
        """
        Update the heightmap with a block at a given position and size.
        """
        polygon = self.get_polygon(position, size, rotation)
        if position[2] != 0.75:
            pos_polygon = self.height[position[2]-size[2]/2]
            update = pos_polygon.difference(polygon)
            self.height[position[2]-size[2]/2] = update

        new_height = position[2] + size[2]/2
        if new_height not in self.height_list:
            self.height_list.append(new_height)
            self.height_list = sorted(self.height_list)
            self.height[new_height] = polygon
        else:
            current_multipolygon = self.height[new_height]
            if current_multipolygon.intersects(polygon):
                raise ValueError("polygons intersect!")
            
            new_multipoly = current_multipolygon.union(polygon)
            self.height[new_height] = new_multipoly

    def get_polygon(self, position, size, rotation):
        """
        Calculate the support area for a block on the heightmap.
        Only when the ratio is enough, the block can be placed.
        """
        l, w = size[0], size[1]
        angle = rotation[2]

        # Calculate the corners of the block in world coordinates
        corners = [
            (position[0] + l/2 * np.cos(angle) - w/2 * np.sin(angle),
             position[1] + l/2 * np.sin(angle) + w/2 * np.cos(angle)),
            (position[0] + l/2 * np.cos(angle) + w/2 * np.sin(angle),
             position[1] + l/2 * np.sin(angle) - w/2 * np.cos(angle)),
            (position[0] - l/2 * np.cos(angle) + w/2 * np.sin(angle),
             position[1] - l/2 * np.sin(angle) - w/2 * np.cos(angle)),
            (position[0] - l/2 * np.cos(angle) - w/2 * np.sin(angle),
             position[1] - l/2 * np.sin(angle) + w/2 * np.cos(angle))
        ]

        # Create a polygon from the corners
        polygon = Polygon(corners)
        return polygon

    def calculate_plane(self, degree, point=None):
        """
        根据给定倾角生成一个固定方向（朝 +x 方向抬起）的平面。
        不再区分 red/green，只保留单一斜坡方向。
        """
        degree = math.radians(degree)
        # 固定法向量，使平面朝 +x 方向抬起
        normal = (-math.sin(degree), 0, math.cos(degree))
        if not point:
            point = (-1.5, 0, 2.5)
        
        a = normal[0]
        b = 0
        c = normal[2]
        d = - a * point[0] - c * point[2]
        return a, b, c, d


    def generate_points_on_plane(self, size, degree, n_points=20, noise_level=2):
        a, b, c, d = self.calculate_plane(degree)

        x = np.random.uniform(settings.PROJECTION_X[0], settings.PROJECTION_X[1], n_points)
        y = np.random.uniform(settings.PROJECTION_Y[0], settings.PROJECTION_Y[1], n_points)
        
        z_plane = (-a*x - b*y - d) / c
        
        noise = np.random.normal(0, noise_level, n_points)
        
        X = np.column_stack([x, y, np.ones_like(x)])
        coefficients = np.linalg.lstsq(X, noise, rcond=None)[0]
        adjusted_noise = noise - X.dot(coefficients)
        
        z = z_plane + adjusted_noise

        processed_z = []
        for z_val in z:
            flag = False
            for i in range(len(self.height_list)):
                if i == 0:
                    h = self.height_list[i]
                    if z_val < h:
                        processed_z.append(self.height_list[i]+size[2]/2)
                        flag = True
                        break
                elif i<len(self.height_list):
                    h = self.height_list[i]
                    if z_val < h:
                        processed_z.append(self.height_list[i-1]+size[2]/2)
                        flag = True
                        break
            if flag == False:
                processed_z.append(self.height_list[-1]+size[2]/2)
        positions = [(float(x[i]), float(y[i]), float(processed_z[i])) for i in range(n_points)]
        return positions

    def get_valid_positions(self, size, rotation, flag):
        """Get all valid positions on the heightmap（不再区分 red/green 区域）。"""
        valid_counts = 0
        valid_positions = []
        while valid_counts < 80:
            if flag == 1:
                # 底座方块：在原点附近的更小矩形区域均匀采样，使底座更“集中”
                # 原来范围较大：x ∈ [-1.5, 1.5], y ∈ [-0.75, 0.75]
                # 这里收缩到更小的区域（如果需要再更集中，可以继续缩小这个范围）
                x = np.random.uniform(-0.8, 0.8)
                y = np.random.uniform(-0.6, 0.6)
                z = 0.75
                position = (x, y, z)
                new_polygon = self.get_polygon(position, size, rotation)
                valid_positions.append(position)
                valid_counts += 1
            else:
                positions = self.generate_points_on_plane(size, settings.DEGREE)
                for position in positions:
                    new_polygon = self.get_polygon(position, size, rotation)
                    pos_multipoly = self.height[position[2]-size[2]/2]
                    if pos_multipoly.intersection(new_polygon).area >= settings.INTERSECTION_THRESHOLD:
                    #if pos_multipoly.intersects(new_polygon):
                        valid_positions.append(position)
                        valid_counts += 1
        sorted_positions = sorted(valid_positions, key=lambda t:t[2])
        return sorted_positions

class CollisionDetector:
    def __init__(self):
        pass
    
    def get_block_vertices(self, position, size, rotation):
        """
        Get the 8 vertices of a block given its position, size, and rotation.
        """
        l, w, h = size
        half_l = l / 2
        half_w = w / 2
        half_h = h / 2
        
        # Create rotation matrix from Euler angles
        rotation_matrix = Euler(rotation, 'XYZ').to_matrix().to_4x4()
        
        # 8 vertices in local space
        local_vertices = [
            Vector(( half_l,  half_w,  half_h)),
            Vector(( half_l,  half_w, -half_h)),
            Vector(( half_l, -half_w,  half_h)),
            Vector(( half_l, -half_w, -half_h)),
            Vector((-half_l,  half_w,  half_h)),
            Vector((-half_l,  half_w, -half_h)),
            Vector((-half_l, -half_w,  half_h)),
            Vector((-half_l, -half_w, -half_h))
        ]
        
        # apply rotation and translation to get world coordinates
        world_vertices = []
        for vertex in local_vertices:
            rotated_vertex = rotation_matrix @ vertex

            world_vertex = Vector(position) + rotated_vertex
            world_vertices.append(world_vertex)
        
        return world_vertices
    
    def get_block_faces(self, vertices):
        """
        Get the faces of a block given its vertices.
        Each face is represented by a list of vertices.
        """
        faces = [
            [0, 1, 3, 2],  
            [4, 5, 7, 6],  
            [0, 4, 6, 2],  
            [1, 5, 7, 3],  
            [0, 1, 5, 4],  
            [2, 3, 7, 6]   
        ]
        
        return [[vertices[i] for i in face] for face in faces]
    
    def separating_axis_theorem(self, vertices1, vertices2):
        """
        Check for collision between two sets of vertices using the Separating Axis Theorem (SAT).
        Returns True if there is a collision, False otherwise.
        """
        # get all possible separating axes
        normals = self.get_all_separating_axes(vertices1, vertices2)
        
        # check each axis
        for normal in normals:
            min1, max1 = self.project_vertices(vertices1, normal)
            min2, max2 = self.project_vertices(vertices2, normal)
            
            if max1 <= min2 or max2 <= min1:
                return False
        # If no separating axis found, there is a collision
        return True
    
    def get_all_separating_axes(self, vertices1, vertices2):
        """
        Get all possible separating axes for two sets of vertices.
        This includes face normals and edge cross products.
        """
        faces1 = self.get_block_faces(vertices1)
        normals1 = [self.get_face_normal(face) for face in faces1]
        
        faces2 = self.get_block_faces(vertices2)
        normals2 = [self.get_face_normal(face) for face in faces2]
        
        edge_normals = []
        for i in range(len(faces1)):
            for j in range(len(faces2)):
                edges1 = self.get_face_edges(faces1[i])
                edges2 = self.get_face_edges(faces2[j])
                
                for edge1 in edges1:
                    for edge2 in edges2:
                        cross = edge1.cross(edge2)
                        if cross.length > 0.001:  # To avoid zero-length normals
                            edge_normals.append(cross.normalized())
        
        all_normals = normals1 + normals2 + edge_normals
        
        unique_normals = []
        seen = set()
        for normal in all_normals:
            # Round to avoid floating point precision issues
            key = (round(normal.x, 3), round(normal.y, 3), round(normal.z, 3))
            if key not in seen:
                seen.add(key)
                unique_normals.append(normal)
        
        return unique_normals
    
    def get_face_normal(self, face_vertices):
        """
        Calculate the normal vector of a face given its vertices.
        The face is defined by three vertices.
        """
        v0 = face_vertices[0]
        v1 = face_vertices[1]
        v2 = face_vertices[2]
        
        edge1 = v1 - v0
        edge2 = v2 - v0
        
        normal = edge1.cross(edge2).normalized()
        return normal
    
    def get_face_edges(self, face_vertices):
        """
        Get the edges of a face defined by its vertices.
        Each edge is represented as a vector from one vertex to the next.
        """
        edges = []
        n = len(face_vertices)
        for i in range(n):
            j = (i + 1) % n
            edge = face_vertices[j] - face_vertices[i]
            edges.append(edge)
        return edges
    
    def project_vertices(self, vertices, axis):
        """
        Project the vertices onto a given axis and return the min and max values.
        The axis should be a normalized vector.
        """
        min_val = float('inf')
        max_val = float('-inf')
        
        for vertex in vertices:
            projection = vertex.dot(axis)
            
            if projection < min_val:
                min_val = projection
            if projection > max_val:
                max_val = projection
        
        return min_val, max_val
    
    def check_block_collision(self, existing_blocks, new_position, new_size, new_rotation):
        """
        Check if a new block collides with existing blocks in the scene.
        """
        new_vertices = self.get_block_vertices(new_position, new_size, new_rotation)
        
        for block in existing_blocks:
                
            existing_vertices = self.get_block_vertices(
                block['position'], 
                block['size'], 
                block['rotation']
            )
            
            if self.separating_axis_theorem(new_vertices, existing_vertices):
                return True  
        
        return False