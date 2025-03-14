import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import matplotlib.cm as cm
import matplotlib.colors as colors
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import zoom

# ---------------------------
# Environment Definition
# ---------------------------
class UAVTerrainEnv(gym.Env):
    def __init__(self):
        super(UAVTerrainEnv, self).__init__()
        
        # Load terrain data from CSV and clip it so it’s always below the UAV's maximum altitude
        self.terrain_data = np.genfromtxt("terrain.csv", delimiter=",")
        self.grid_size = self.terrain_data.shape[0]
        self.uav_max_altitude = 20       # UAV's maximum altitude
        terrain_margin = 1.0             # Safety margin
        self.terrain_data = np.clip(self.terrain_data, None, self.uav_max_altitude - terrain_margin)
        self.terrain_min = float(np.min(self.terrain_data))
        self.terrain_max = float(np.max(self.terrain_data))
        
        # Define x-y bounds for the terrain
        self.x_bounds = (-5, 5)
        self.y_bounds = (-5, 5)
        
        # Create grid points for interpolation
        self.x_range = np.linspace(self.x_bounds[0], self.x_bounds[1], self.grid_size)
        self.y_range = np.linspace(self.y_bounds[0], self.y_bounds[1], self.grid_size)
        
        # Interpolator for continuous terrain elevation lookup
        self.terrain_function = RegularGridInterpolator((self.x_range, self.y_range), self.terrain_data, method="cubic")
        
        # Altitude bounds for the UAV (terrain_min to UAV max altitude)
        self.z_bounds = (self.terrain_min, self.uav_max_altitude)
        
        # Define observation space: [position (x,y,z), velocity (vx,vy,vz), goal (x,y,z)]
        obs_low = np.array([self.x_bounds[0], self.y_bounds[0], self.z_bounds[0],
                            -5, -5, -5,
                            self.x_bounds[0], self.y_bounds[0], self.z_bounds[0]], dtype=np.float32)
        obs_high = np.array([self.x_bounds[1], self.y_bounds[1], self.z_bounds[1],
                             5, 5, 5,
                             self.x_bounds[1], self.y_bounds[1], self.z_bounds[1]], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        
        # Define action space: accelerations in x, y, and z (continuous)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Dynamics parameter
        self.dt = 0.1
        
        # Collision margin: UAV must remain above terrain by at least this much
        self.collision_margin = 0.1
        
        # Generate goal position
        self.goal = self._generate_goal_position()
        
        # For tracking the UAV's trajectory (for visualization)
        self.trajectory = []
        
        # --- Enemy radar parameters ---
        # Place an enemy radar at a random (x,y) location
        self.enemy_radar_pos = np.array([
            np.random.uniform(*self.x_bounds),
            np.random.uniform(*self.y_bounds)
        ])
        self.enemy_radar_radius = 4.0      # Danger zone radius
        self.uav_sensor_radius = 2.0       # UAV sensor detection radius
        self.enemy_radar_detected = False
        
        # Timer to accumulate time in enemy radar danger zone (for missile hit probability)
        self.time_in_radar = 0.0  
        
        # Pygame window parameters (if rendering)
        self.window_width = 800
        self.window_height = 800
        
        self.reset()
    
    def _generate_goal_position(self):
        x_goal = np.random.uniform(*self.x_bounds)
        y_goal = np.random.uniform(*self.y_bounds)
        ground_elev = self.get_terrain_elevation(x_goal, y_goal)
        z_goal = np.random.uniform(ground_elev + 1.0, self.z_bounds[1])
        return np.array([x_goal, y_goal, z_goal], dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset timer for radar exposure
        self.time_in_radar = 0.0
        # Random starting position above terrain
        x_init = np.random.uniform(*self.x_bounds)
        y_init = np.random.uniform(*self.y_bounds)
        ground_elev = self.get_terrain_elevation(x_init, y_init)
        z_init = np.random.uniform(ground_elev + 1.0, self.z_bounds[1])
        self.position = np.array([x_init, y_init, z_init], dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.goal = self._generate_goal_position()
        self.state = np.concatenate([self.position, self.velocity, self.goal])
        self.trajectory = [self.position.copy()]
        return self.state, {}
    
    def step(self, action):
        # Clip action
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # Update dynamics
        self.velocity += action * self.dt
        self.position += self.velocity * self.dt
        self.position[0] = np.clip(self.position[0], *self.x_bounds)
        self.position[1] = np.clip(self.position[1], *self.y_bounds)
        
        ground_elev = self.get_terrain_elevation(self.position[0], self.position[1])
        min_z = max(self.z_bounds[0], ground_elev + self.collision_margin)
        self.position[2] = np.clip(self.position[2], min_z, self.z_bounds[1])
        
        # --- Radar detection and missile hit probability ---
        # Compute horizontal distance to enemy radar
        dist_to_radar = np.linalg.norm(self.position[:2] - self.enemy_radar_pos)
        if dist_to_radar < self.uav_sensor_radius:
            self.enemy_radar_detected = True
        else:
            self.enemy_radar_detected = False
        
        # Accumulate time if UAV is in enemy radar danger zone (within enemy_radar_radius)
        if dist_to_radar < self.enemy_radar_radius:
            self.time_in_radar += self.dt
        else:
            self.time_in_radar = 0.0
        
        # Compute probability of missile hit based on time in danger zone.
        # For example, if UAV is in zone for 5 seconds continuously, probability becomes 1.
        hit_probability = min(1.0, self.time_in_radar / 5.0)
        missile_hit = np.random.rand() < hit_probability
        
        # If missile hit occurs, immediately terminate the episode with a heavy penalty.
        if missile_hit and self.time_in_radar > 0:
            termination_reason = "Hit by missile due to prolonged radar exposure"
            reward = -100  # Heavy penalty for being hit
            done = True
            info = {"termination_reason": termination_reason}
            self.state = np.concatenate([self.position, self.velocity, self.goal])
            self.trajectory.append(self.position.copy())
            return self.state, float(reward), bool(done), False, info
        
        # Compute extra penalty or bonus based on radar proximity (separate from missile hit)
        extra_penalty = 0
        bonus = 0
        if dist_to_radar < self.enemy_radar_radius:
            extra_penalty = 50  # Penalty for being inside the danger zone
        else:
            bonus = 10       # Bonus for being outside the danger zone
        
        collision = self.position[2] < ground_elev + self.collision_margin
        
        self.state = np.concatenate([self.position, self.velocity, self.goal])
        self.trajectory.append(self.position.copy())
        
        distance = np.linalg.norm(self.position - self.goal)
        # Reward: negative distance to goal, minus extra penalty plus bonus.
        reward = -distance - extra_penalty + bonus
        reward = float(reward)
        
        termination_reason = None
        if distance < 0.5:
            termination_reason = "Goal reached"
        elif collision:
            termination_reason = f"Collision with terrain at ({self.position[0]:.2f}, {self.position[1]:.2f}). Altitude: {self.position[2]:.2f}, Terrain: {ground_elev:.2f}"
        
        done = bool((distance < 0.5) or collision)
        info = {"termination_reason": termination_reason}
        return self.state, reward, done, False, info
    
    def get_terrain_elevation(self, x, y):
        x = np.clip(x, self.x_bounds[0], self.x_bounds[1])
        y = np.clip(y, self.y_bounds[0], self.y_bounds[1])
        return self.terrain_function([[x, y]])[0]
    
    def create_terrain_surface(self):
        # Upscale the terrain data for a smoother visualization
        zoom_factor = self.window_width / self.terrain_data.shape[1]
        smooth_terrain = zoom(self.terrain_data, zoom_factor, order=3)
        
        # Normalize and apply the colormap
        norm = colors.Normalize(vmin=self.terrain_min, vmax=self.terrain_max)
        cmap = cm.get_cmap('terrain')
        terrain_rgba = cmap(norm(smooth_terrain))
        
        # Convert RGBA to RGB image (drop alpha) and scale to 0-255
        terrain_rgb = (terrain_rgba[:, :, :3] * 255).astype(np.uint8)
        # Transpose dimensions to match Pygame's expected (width, height, channels)
        terrain_rgb = np.transpose(terrain_rgb, (1, 0, 2))
        surface = pygame.surfarray.make_surface(terrain_rgb)
        surface = pygame.transform.scale(surface, (self.window_width, self.window_height))
        return surface
    
    def create_colorbar_surface(self):
        # Create a vertical colorbar showing altitude-to-color mapping
        bar_width = 50
        bar_height = 300
        gradient = np.linspace(self.terrain_min, self.terrain_max, bar_height)
        bar_image = np.zeros((bar_height, bar_width, 3), dtype=np.uint8)
        norm = colors.Normalize(vmin=self.terrain_min, vmax=self.terrain_max)
        cmap = cm.get_cmap('terrain')
        for i in range(bar_height):
            color = cmap(norm(gradient[i]))[:3]  # RGB color (0-1)
            color = (np.array(color) * 255).astype(np.uint8)
            bar_image[i, :] = color
        bar_image = np.transpose(bar_image, (1, 0, 2))
        surface = pygame.surfarray.make_surface(bar_image)
        return surface
    
    def env_to_screen(self, x, y):
        screen_x = int((x - self.x_bounds[0]) / (self.x_bounds[1]-self.x_bounds[0]) * self.window_width)
        screen_y = self.window_height - int((y - self.y_bounds[0]) / (self.y_bounds[1]-self.y_bounds[0]) * self.window_height)
        return screen_x, screen_y

# ---------------------------
# Pygame Runner
# ---------------------------
def run_pygame_controlled_env():
    pygame.init()
    window_width, window_height = 800, 800
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("UAV Terrain Environment with Missile Hit Condition")
    clock = pygame.time.Clock()
    
    env = UAVTerrainEnv()
    env.window_width = window_width
    env.window_height = window_height
    env.reset()
    
    terrain_surface = env.create_terrain_surface()
    colorbar_surface = env.create_colorbar_surface()
    
    running = True
    font = pygame.font.SysFont("Arial", 18)
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        action = np.zeros(3, dtype=np.float32)
        if keys[pygame.K_a]:
            action[0] = -0.5
        if keys[pygame.K_d]:
            action[0] = 0.5
        if keys[pygame.K_w]:
            action[1] = 0.5
        if keys[pygame.K_s]:
            action[1] = -0.5
        if keys[pygame.K_q]:
            action[2] = 0.5
        if keys[pygame.K_e]:
            action[2] = -0.5
        if keys[pygame.K_r]:
            env.reset()
        if keys[pygame.K_ESCAPE]:
            running = False
        
        state, reward, done, _, info = env.step(action)
        if done:
            print("Episode finished! Reason:", info.get("termination_reason"))
            env.reset()
        
        screen.blit(terrain_surface, (0, 0))
        
        radar_screen = env.env_to_screen(env.enemy_radar_pos[0], env.enemy_radar_pos[1])
        screen_radius = int((env.enemy_radar_radius / (env.x_bounds[1]-env.x_bounds[0])) * window_width)
        pygame.draw.circle(screen, (255, 0, 0), radar_screen, screen_radius, 2)
        pygame.draw.circle(screen, (255, 0, 0), radar_screen, 4)
        
        pos_screen = env.env_to_screen(env.position[0], env.position[1])
        sensor_screen_radius = int((env.uav_sensor_radius / (env.x_bounds[1]-env.x_bounds[0])) * window_width)
        pygame.draw.circle(screen, (0, 255, 0), pos_screen, sensor_screen_radius, 1)
        
        goal_screen = env.env_to_screen(env.goal[0], env.goal[1])
        pygame.draw.circle(screen, (255, 0, 0), goal_screen, 8)
        pygame.draw.circle(screen, (0, 0, 255), pos_screen, 8)
        
        if len(env.trajectory) > 1:
            traj_points = [env.env_to_screen(p[0], p[1]) for p in env.trajectory]
            pygame.draw.lines(screen, (0, 0, 0), False, traj_points, 2)
        
        info_text = f"Pos: ({env.position[0]:.2f}, {env.position[1]:.2f}, {env.position[2]:.2f})  Dist to Goal: {np.linalg.norm(env.position - env.goal):.2f}"
        text_surface = font.render(info_text, True, (0, 0, 0))
        screen.blit(text_surface, (10, 10))
        
        screen.blit(colorbar_surface, (window_width - colorbar_surface.get_width() - 10, 10))
        
        pygame.display.flip()
        clock.tick(20)
    
    pygame.quit()

if __name__ == "__main__":
    run_pygame_controlled_env()
