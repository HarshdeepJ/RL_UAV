# UAV Terrain Environment

## Repo is in progress
### Progress: Built environment, using pygame to control it.

This repository contains a custom OpenAI Gymnasium environment simulating a UAV (Unmanned Aerial Vehicle) navigating a terrain while avoiding obstacles and enemy radar zones.
## Environment Overview

- **Terrain Representation:**
  - The environment loads a terrain map from a CSV file, ensuring smooth and realistic elevation transitions.
  - The terrain is scaled and clipped to provide a meaningful navigation challenge.

- **UAV Dynamics:**
  - The UAV can move in a 3D space with continuous control over acceleration in the x, y, and z directions.
  - Velocity updates are based on the chosen actions, allowing smooth motion.

- **Enemy Radar & Missile Threats:**
  - A radar zone is placed randomly within the terrain, representing an enemy surveillance area.
  - If the UAV enters the radar zone, it has to come under 10 seconds max to max, else will be hit with the probability of 1.

- **Goal-Oriented Navigation:**
  - A randomly generated goal position is set at the start of each episode.
  - The UAV must reach the goal while minimizing risks from obstacles and enemy radar.

- **Controls**
  - W,A,S and D representing the usual control to increase/decrease acceleration.
  - Q for increasing the altitude and E for taking it down.
    
- **Rendering (Pygame-based):**
  - The environment includes a Pygame-based visualization displaying terrain, the UAV, its sensor range, radar coverage, and trajectory.
  - A color gradient represents elevation levels for better interpretability.

### To customize Terrain: Load different terrain maps for varied difficulty levels.

