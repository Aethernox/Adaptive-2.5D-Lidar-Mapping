"""
Procedural Multi-Street City Network & Urban Driving Environment.

Features:
  - 4 interconnected city streets with frequent 90-degree corner turns every ~35m:
      Street 1: Grand Avenue (Eastbound) -> 90° Turn 1
      Street 2: South Boulevard (Southbound) -> 90° Turn 2
      Street 3: Industrial Way (Westbound) -> 90° Turn 3
      Street 4: Park Lane (Northbound) -> 90° Turn 4 (loops back into Grand Ave)
  - 4-way cross-street intersections, pedestrian crosswalks, traffic signals.
  - Distinct street aesthetics:
      Grand Ave: Commercial storefronts, street lights, parked sedans, crosswalk.
      South Blvd: Multi-story office towers, crossroad traffic, pedestrian plaza.
      Industrial Way: Construction zone with cones, crash barriers, utility poles, work truck.
      Park Lane: Urban park with dense tree canopies, bike lane cyclist, bus stops.
"""
import numpy as np
from config import RNG_SEED

CLASS_DRIVABLE, CLASS_NONDRIVABLE, CLASS_STRUCTURE, CLASS_POLE, CLASS_PED, CLASS_VEH = range(6)

# ---------------------------------------------------------------------------
# Compact City Circuit: Turns occur every ~35 meters (~7-8 seconds of driving)
# Geometry:
#   Grand Ave:     (-10, 0) -> (25, 0)          [35m, East]
#   Turn 1 (90°):  Center (25, -15), R=15m      [23.56m, East -> South]
#   South Blvd:    (40, -15) -> (40, -50)       [35m, South]
#   Turn 2 (90°):  Center (25, -50), R=15m      [23.56m, South -> West]
#   Industrial:    (25, -65) -> (-10, -65)      [35m, West]
#   Turn 3 (90°):  Center (-10, -50), R=15m     [23.56m, West -> North]
#   Park Lane:     (-25, -50) -> (-25, -15)     [35m, North]
#   Turn 4 (90°):  Center (-10, -15), R=15m     [23.56m, North -> East]
# Total Loop: 35 + 23.56 + 35 + 23.56 + 35 + 23.56 + 35 + 23.56 = 234.25m
# ---------------------------------------------------------------------------

CIRCUIT_SEG_LEN = 35.0
TURN_ARC_LEN = 23.56194
CIRCUIT_LENGTH = 4 * (CIRCUIT_SEG_LEN + TURN_ARC_LEN)  # 234.25m


def get_circuit_pose(s):
    """Compute (x, y, heading, street_name) at circuit distance s."""
    s = np.mod(s, CIRCUIT_LENGTH)
    
    # 1. Street 1: Grand Avenue (Eastbound: y=0, x from -10 to 25)
    if s <= 35.0:
        return -10.0 + s, 0.0, 0.0, "Grand Avenue"
    s -= 35.0
    
    # Turn 1: Grand Ave -> South Blvd (Center: 25, -15, R=15, angle pi/2 -> 0)
    if s <= 23.56194:
        th = (np.pi / 2.0) - (s / 15.0)
        x = 25.0 + 15.0 * np.cos(th)
        y = -15.0 + 15.0 * np.sin(th)
        heading = -(s / 15.0)  # 0 -> -pi/2
        return x, y, heading, "Turning onto South Blvd"
    s -= 23.56194
    
    # 2. Street 2: South Boulevard (Southbound: x=40, y from -15 to -50)
    if s <= 35.0:
        return 40.0, -15.0 - s, -np.pi / 2.0, "South Boulevard"
    s -= 35.0
    
    # Turn 2: South Blvd -> Industrial Way (Center: 25, -50, R=15, angle 0 -> -pi/2)
    if s <= 23.56194:
        th = 0.0 - (s / 15.0)
        x = 25.0 + 15.0 * np.cos(th)
        y = -50.0 + 15.0 * np.sin(th)
        heading = -np.pi / 2.0 - (s / 15.0)  # -pi/2 -> -pi
        return x, y, heading, "Turning onto Industrial Way"
    s -= 23.56194
    
    # 3. Street 3: Industrial Way (Westbound: y=-65, x from 25 to -10)
    if s <= 35.0:
        return 25.0 - s, -65.0, np.pi, "Industrial Way"
    s -= 35.0
    
    # Turn 3: Industrial Way -> Park Lane (Center: -10, -50, R=15, angle -pi/2 -> -pi)
    if s <= 23.56194:
        th = -np.pi / 2.0 - (s / 15.0)
        x = -10.0 + 15.0 * np.cos(th)
        y = -50.0 + 15.0 * np.sin(th)
        heading = np.pi - (s / 15.0)  # pi -> pi/2
        return x, y, heading, "Turning onto Park Lane"
    s -= 23.56194
    
    # 4. Street 4: Park Lane (Northbound: x=-25, y from -50 to -15)
    if s <= 35.0:
        return -25.0, -50.0 + s, np.pi / 2.0, "Park Lane"
    s -= 35.0
    
    # Turn 4: Park Lane -> Grand Ave (Center: -10, -15, R=15, angle -pi -> -3pi/2)
    th = -np.pi - (s / 15.0)
    x = -10.0 + 15.0 * np.cos(th)
    y = -15.0 + 15.0 * np.sin(th)
    heading = np.pi / 2.0 - (s / 15.0)  # pi/2 -> 0
    return x, y, heading, "Turning onto Grand Ave"


def _distance_to_road_network(x, y):
    """
    Vectorized shortest distance from (x, y) to the nearest street centerline
    including straight avenues, corner turns, and cross-street intersections.
    """
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    
    # 4 Main Straight Streets
    d1 = np.where((x >= -35) & (x <= 50), np.abs(y), 1e5)            # Grand Ave (y=0)
    d2 = np.where((y >= -75) & (y <= 10), np.abs(x - 40.0), 1e5)     # South Blvd (x=40)
    d3 = np.where((x >= -35) & (x <= 50), np.abs(y + 65.0), 1e5)     # Industrial Way (y=-65)
    d4 = np.where((y >= -75) & (y <= 10), np.abs(x + 25.0), 1e5)     # Park Lane (x=-25)
    
    # 4 90-degree Corner Arcs (R = 15m)
    d_c1 = np.abs(np.hypot(x - 25.0, y + 15.0) - 15.0)
    d_c1 = np.where((x >= 25) & (y >= -15), d_c1, 1e5)

    d_c2 = np.abs(np.hypot(x - 25.0, y + 50.0) - 15.0)
    d_c2 = np.where((x >= 25) & (y <= -50), d_c2, 1e5)

    d_c3 = np.abs(np.hypot(x + 10.0, y + 50.0) - 15.0)
    d_c3 = np.where((x <= -10) & (y <= -50), d_c3, 1e5)

    d_c4 = np.abs(np.hypot(x + 10.0, y + 15.0) - 15.0)
    d_c4 = np.where((x <= -10) & (y >= -15), d_c4, 1e5)

    all_d = np.stack([d1, d2, d3, d4, d_c1, d_c2, d_c3, d_c4], axis=0)
    return np.min(all_d, axis=0)


def ground_height(x, y):
    """Terrain elevation for the multi-street city grid: road, curbs, sidewalks."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    
    z = 0.02 * np.sin(x * 0.1) * np.cos(y * 0.1)
    d_road = _distance_to_road_network(x, y)
    
    # Raised curbs at 5.0m to 5.6m
    on_curb = (d_road >= 5.0) & (d_road <= 5.6)
    # Elevated sidewalks at 5.6m to 9.2m
    on_sidewalk = (d_road > 5.6) & (d_road <= 9.2)
    
    z = z + np.where(on_curb, 0.16, 0.0)
    z = z + np.where(on_sidewalk, 0.15, 0.0)
    
    # Localized potholes on Grand Ave & South Blvd
    d_pot1 = (x - 14.0) ** 2 + (y + 1.8) ** 2
    pothole1 = np.where(d_pot1 < 1.3 ** 2, -0.14 * (1.0 - d_pot1 / (1.3 ** 2)), 0.0)
    
    d_pot2 = (x - 40.0) ** 2 + (y + 35.0) ** 2
    pothole2 = np.where(d_pot2 < 1.2 ** 2, -0.12 * (1.0 - d_pot2 / (1.2 ** 2)), 0.0)
    
    return z + pothole1 + pothole2


def ground_class(x, y):
    """0 = drivable road surface, 1 = non-drivable (curbs, sidewalks, off-road, potholes)."""
    d_road = _distance_to_road_network(x, y)
    is_road = d_road <= 5.0
    
    d_pot1 = (x - 14.0) ** 2 + (y + 1.8) ** 2
    rim1 = (d_pot1 < 1.6 ** 2) & (d_pot1 >= 1.0 ** 2)
    
    d_pot2 = (x - 40.0) ** 2 + (y + 35.0) ** 2
    rim2 = (d_pot2 < 1.5 ** 2) & (d_pot2 >= 0.9 ** 2)
    
    drivable = is_road & (~rim1) & (~rim2)
    return np.where(drivable, CLASS_DRIVABLE, CLASS_NONDRIVABLE).astype(np.int32)


class StaticObject:
    """Axis-aligned box (walls, buildings, parked cars, barriers) or cylinder (poles, trees)."""

    def __init__(self, kind, cx, cy, w=0.3, d=0.3, h=1.8, base_z=None, cls=None):
        self.kind = kind
        self.cx, self.cy = float(cx), float(cy)
        self.w, self.d, self.h = float(w), float(d), float(h)
        if base_z is not None:
            self.base_z = float(base_z)
        else:
            self.base_z = float(ground_height(np.array([cx]), np.array([cy]))[0])
        
        if cls is not None:
            self.cls = cls
        elif kind in ("wall", "building", "barrier", "parked_car"):
            self.cls = CLASS_STRUCTURE
        elif kind in ("pole", "tree", "sign", "cone"):
            self.cls = CLASS_POLE
        else:
            self.cls = CLASS_STRUCTURE

    def sample_points(self, n):
        """Sample n surface points on the static object."""
        if n <= 0:
            return np.array([]), np.array([]), np.array([])
        
        if self.kind in ("pole", "tree_trunk"):
            theta = np.random.uniform(0, 2 * np.pi, n)
            z = np.random.uniform(0, self.h, n)
            r = self.w / 2.0
            x = self.cx + r * np.cos(theta)
            y = self.cy + r * np.sin(theta)
            return x, y, self.base_z + z
        
        elif self.kind == "tree":
            n_trunk = max(int(n * 0.3), 2)
            n_canopy = n - n_trunk
            th_t = np.random.uniform(0, 2 * np.pi, n_trunk)
            z_t = np.random.uniform(0, self.h * 0.45, n_trunk)
            x_t = self.cx + 0.25 * np.cos(th_t)
            y_t = self.cy + 0.25 * np.sin(th_t)
            
            phi = np.random.uniform(0, np.pi, n_canopy)
            th_c = np.random.uniform(0, 2 * np.pi, n_canopy)
            r_c = np.random.uniform(0.6 * self.w, self.w, n_canopy)
            x_c = self.cx + r_c * np.sin(phi) * np.cos(th_c)
            y_c = self.cy + r_c * np.sin(phi) * np.sin(th_c)
            z_c = self.h * 0.65 + r_c * np.cos(phi)
            
            return np.concatenate([x_t, x_c]), np.concatenate([y_t, y_c]), self.base_z + np.concatenate([z_t, z_c])
        
        elif self.kind == "cone":
            u = np.random.uniform(0, 1, n)
            z = u * self.h
            r = (1.0 - u) * (self.w / 2.0)
            theta = np.random.uniform(0, 2 * np.pi, n)
            x = self.cx + r * np.cos(theta)
            y = self.cy + r * np.sin(theta)
            return x, y, self.base_z + z
        
        else:
            # Building facade, parked car, barrier box
            faces = np.random.choice(4, size=n, p=[0.35, 0.35, 0.15, 0.15])
            x = np.zeros(n, dtype=np.float32)
            y = np.zeros(n, dtype=np.float32)
            z = np.random.uniform(0.05, self.h, n)
            
            m0 = faces == 0
            if np.any(m0):
                x[m0] = self.cx + np.random.uniform(-self.w / 2, self.w / 2, m0.sum())
                y[m0] = self.cy - self.d / 2
            
            m1 = faces == 1
            if np.any(m1):
                x[m1] = self.cx + np.random.uniform(-self.w / 2, self.w / 2, m1.sum())
                y[m1] = self.cy + self.d / 2
            
            m2 = faces == 2
            if np.any(m2):
                x[m2] = self.cx - self.w / 2
                y[m2] = self.cy + np.random.uniform(-self.d / 2, self.d / 2, m2.sum())
            
            m3 = faces == 3
            if np.any(m3):
                x[m3] = self.cx + self.w / 2
                y[m3] = self.cy + np.random.uniform(-self.d / 2, self.d / 2, m3.sum())
            
            return x, y, self.base_z + z


class DynamicActor:
    """Dynamic actor (vehicle, pedestrian, cyclist, truck, bus) navigating the city circuit."""

    def __init__(self, kind, route_s0=0.0, speed=4.0, offset_y=-1.8, w=1.95, d=4.6, h=1.6, behavior="circuit"):
        self.kind = kind
        self.route_s0 = float(route_s0)
        self.speed = float(speed)
        self.offset_y = float(offset_y)
        self.w, self.d, self.h = float(w), float(d), float(h)
        self.behavior = behavior
        self.cls = CLASS_PED if kind in ("pedestrian", "cyclist") else CLASS_VEH

    def pose(self, t):
        """Compute (cx, cy, current_vx, current_vy) at time t."""
        if self.behavior == "cross":
            s = self.route_s0
            rx, ry, rth, _ = get_circuit_pose(s)
            cross_offset = 4.2 * np.sin(t * 0.9)
            nx, ny = -np.sin(rth), np.cos(rth)
            cx = rx + nx * cross_offset
            cy = ry + ny * cross_offset
            vx = nx * 4.2 * 0.9 * np.cos(t * 0.9)
            vy = ny * 4.2 * 0.9 * np.cos(t * 0.9)
            return cx, cy, vx, vy
        else:
            s = self.route_s0 + self.speed * t
            rx, ry, rth, _ = get_circuit_pose(s)
            nx, ny = -np.sin(rth), np.cos(rth)
            cx = rx + nx * self.offset_y
            cy = ry + ny * self.offset_y
            vx = self.speed * np.cos(rth)
            vy = self.speed * np.sin(rth)
            return cx, cy, vx, vy

    def sample_points(self, t, n):
        """Sample n points on 3D volume of dynamic actor."""
        if n <= 0:
            return np.array([]), np.array([]), np.array([]), 0.0, 0.0
        
        cx, cy, _, _ = self.pose(t)
        base_z = float(ground_height(np.array([cx]), np.array([cy]))[0])
        
        if self.kind == "pedestrian":
            u = np.random.uniform(-self.w / 2, self.w / 2, n)
            v = np.random.uniform(-self.d / 2, self.d / 2, n)
            z = np.random.uniform(0.05, self.h, n)
            return cx + u, cy + v, base_z + z, cx, cy
        
        elif self.kind == "cyclist":
            u = np.random.uniform(-self.w / 2, self.w / 2, n)
            v = np.random.uniform(-self.d / 2, self.d / 2, n)
            z = np.random.uniform(0.1, self.h, n)
            return cx + u, cy + v, base_z + z, cx, cy
        
        else:
            faces = np.random.choice(5, size=n, p=[0.3, 0.3, 0.15, 0.15, 0.1])
            x = np.zeros(n, dtype=np.float32)
            y = np.zeros(n, dtype=np.float32)
            z = np.zeros(n, dtype=np.float32)
            
            m0 = faces == 0
            if np.any(m0):
                x[m0] = cx + self.d / 2
                y[m0] = cy + np.random.uniform(-self.w / 2, self.w / 2, m0.sum())
                z[m0] = np.random.uniform(0.1, self.h * 0.75, m0.sum())
            
            m1 = faces == 1
            if np.any(m1):
                x[m1] = cx - self.d / 2
                y[m1] = cy + np.random.uniform(-self.w / 2, self.w / 2, m1.sum())
                z[m1] = np.random.uniform(0.1, self.h * 0.9, m1.sum())
            
            m2 = faces == 2
            if np.any(m2):
                x[m2] = cx + np.random.uniform(-self.d / 2, self.d / 2, m2.sum())
                y[m2] = cy + self.w / 2
                z[m2] = np.random.uniform(0.1, self.h, m2.sum())
            
            m3 = faces == 3
            if np.any(m3):
                x[m3] = cx + np.random.uniform(-self.d / 2, self.d / 2, m3.sum())
                y[m3] = cy - self.w / 2
                z[m3] = np.random.uniform(0.1, self.h, m3.sum())
            
            m4 = faces == 4
            if np.any(m4):
                x[m4] = cx + np.random.uniform(-self.d * 0.35, self.d * 0.35, m4.sum())
                y[m4] = cy + np.random.uniform(-self.w * 0.45, self.w * 0.45, m4.sum())
                z[m4] = self.h
            
            return x, y, base_z + z, cx, cy


class World:
    """
    City Street Network World.
    
    Contains static architecture along all 4 city avenues, 4 corner turns,
    and active dynamic traffic circulating the network.
    """

    def __init__(self, seed=RNG_SEED):
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        self.statics = []
        self._build_city_landmarks()

        # Dynamic actors circulating the compact city circuit
        self.dynamics = [
            # Oncoming car in left lane of Grand Ave
            DynamicActor("vehicle", route_s0=24.0, speed=-7.0, offset_y=1.8, w=1.95, d=4.6, h=1.6),
            # Lead car ahead on South Blvd
            DynamicActor("vehicle", route_s0=68.0, speed=3.4, offset_y=-1.8, w=1.9, d=4.5, h=1.55),
            # Oncoming city bus on Industrial Way
            DynamicActor("truck", route_s0=130.0, speed=-6.0, offset_y=1.8, w=2.5, d=8.5, h=3.2),
            # Cyclist in bike lane on Park Lane
            DynamicActor("cyclist", route_s0=185.0, speed=3.2, offset_y=-4.0, w=0.6, d=1.8, h=1.6),
            # Crossing pedestrian at Grand Ave crosswalk
            DynamicActor("pedestrian", route_s0=16.0, speed=1.0, behavior="cross", w=0.5, d=0.5, h=1.7),
            # Crossing pedestrian at South Blvd plaza
            DynamicActor("pedestrian", route_s0=76.0, speed=1.1, behavior="cross", w=0.5, d=0.5, h=1.7),
            # Sidewalk pedestrians
            DynamicActor("pedestrian", route_s0=10.0, speed=1.2, offset_y=6.5, w=0.5, d=0.5, h=1.75),
            DynamicActor("pedestrian", route_s0=195.0, speed=-1.0, offset_y=-6.5, w=0.5, d=0.5, h=1.65),
            # Faster overtaking vehicle
            DynamicActor("vehicle", route_s0=100.0, speed=5.5, offset_y=-1.8, w=2.0, d=4.7, h=1.65),
        ]

    def _build_city_landmarks(self):
        """Place distinct buildings, light poles, trees, parked cars along each street."""
        # 1. Street 1 (Grand Avenue: y=0, x from -10 to 25)
        for x in range(-10, 26, 12):
            self.statics.append(StaticObject("wall", cx=x + 5.0, cy=12.0, w=10.0, d=5.0, h=6.5))
            self.statics.append(StaticObject("wall", cx=x + 5.0, cy=-12.0, w=10.0, d=5.0, h=5.5))
            self.statics.append(StaticObject("pole", cx=x, cy=5.5, w=0.25, h=4.2))
            self.statics.append(StaticObject("pole", cx=x + 6.0, cy=-5.5, w=0.25, h=4.2))
            self.statics.append(StaticObject("tree", cx=x + 3.0, cy=7.0, w=1.8, d=1.8, h=4.5))

        # Grand Ave parked car & construction zone (x in [16, 22]m)
        self.statics.append(StaticObject("parked_car", cx=6.0, cy=-3.8, w=1.9, d=4.4, h=1.55))
        self.statics.append(StaticObject("parked_car", cx=20.0, cy=-1.8, w=2.0, d=4.6, h=1.6))
        for ci in range(3):
            self.statics.append(StaticObject("cone", cx=16.0 + ci * 2.0, cy=-1.8, w=0.4, d=0.4, h=0.75))
        self.statics.append(StaticObject("barrier", cx=22.5, cy=-1.8, w=2.0, d=0.5, h=1.1))

        # 2. Turn 1 Landmark: Corner Glass Tower (25, -15)
        self.statics.append(StaticObject("wall", cx=42.0, cy=6.0, w=14.0, d=14.0, h=9.5))
        self.statics.append(StaticObject("pole", cx=36.0, cy=4.0, w=0.3, h=4.5))

        # 3. Street 2 (South Boulevard: x=40, y from -15 to -50)
        for y in range(-15, -55, -12):
            self.statics.append(StaticObject("wall", cx=52.0, cy=y - 5.0, w=5.0, d=10.0, h=7.5))
            self.statics.append(StaticObject("wall", cx=28.0, cy=y - 5.0, w=5.0, d=10.0, h=6.0))
            self.statics.append(StaticObject("pole", cx=45.5, cy=y, w=0.25, h=4.2))
            self.statics.append(StaticObject("pole", cx=34.5, cy=y - 6.0, w=0.25, h=4.2))
            self.statics.append(StaticObject("tree", cx=47.0, cy=y - 3.0, w=1.8, d=1.8, h=4.5))
        self.statics.append(StaticObject("parked_car", cx=44.0, cy=-30.0, w=1.9, d=4.4, h=1.55))

        # 4. Turn 2 Landmark: Warehouse Complex (25, -50)
        self.statics.append(StaticObject("wall", cx=42.0, cy=-68.0, w=14.0, d=14.0, h=8.5))

        # 5. Street 3 (Industrial Way: y=-65, x from 25 to -10)
        for x in range(-10, 26, 12):
            self.statics.append(StaticObject("wall", cx=x + 5.0, cy=-77.0, w=10.0, d=5.0, h=5.2))
            self.statics.append(StaticObject("wall", cx=x + 5.0, cy=-53.0, w=10.0, d=5.0, h=5.0))
            self.statics.append(StaticObject("pole", cx=x, cy=-70.5, w=0.25, h=4.2))
            self.statics.append(StaticObject("pole", cx=x + 6.0, cy=-59.5, w=0.25, h=4.2))
            self.statics.append(StaticObject("tree", cx=x + 3.0, cy=-72.0, w=1.8, d=1.8, h=4.5))

        # Industrial Way construction zone (x=10m)
        for ci in range(4):
            self.statics.append(StaticObject("cone", cx=4.0 + ci * 2.0, cy=-66.8, w=0.4, d=0.4, h=0.75))
        self.statics.append(StaticObject("barrier", cx=12.5, cy=-66.8, w=2.2, d=0.5, h=1.1))

        # 6. Turn 3 Landmark: Substation Building (-10, -50)
        self.statics.append(StaticObject("wall", cx=-28.0, cy=-68.0, w=14.0, d=14.0, h=7.5))

        # 7. Street 4 (Park Lane: x=-25, y from -50 to -15) - Urban Park
        for y in range(-50, -10, 10):
            # Dense park tree canopies along left
            self.statics.append(StaticObject("tree", cx=-34.0, cy=y + 2.0, w=2.4, d=2.4, h=5.5))
            self.statics.append(StaticObject("tree", cx=-36.5, cy=y + 7.0, w=2.8, d=2.8, h=6.0))
            # Right side buildings
            self.statics.append(StaticObject("wall", cx=-13.0, cy=y + 4.0, w=5.0, d=8.0, h=6.5))
            self.statics.append(StaticObject("pole", cx=-30.5, cy=y, w=0.25, h=4.2))
            self.statics.append(StaticObject("pole", cx=-19.5, cy=y + 5.0, w=0.25, h=4.2))

        # 8. Turn 4 Landmark: Metro Station (-10, -15)
        self.statics.append(StaticObject("wall", cx=-28.0, cy=4.0, w=14.0, d=14.0, h=8.0))

    def get_statics_near(self, ego_x, ego_y, radius=110.0):
        """Return all static objects within sensor radius."""
        return [obj for obj in self.statics if np.hypot(obj.cx - ego_x, obj.cy - ego_y) <= radius]

    def get_dynamics_near(self, t, ego_x, ego_y, radius=110.0):
        """Return all dynamic actors within sensor radius at time t."""
        filtered = []
        for act in self.dynamics:
            cx, cy, _, _ = act.pose(t)
            if np.hypot(cx - ego_x, cy - ego_y) <= radius:
                filtered.append(act)
        return filtered
