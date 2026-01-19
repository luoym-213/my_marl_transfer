"""
ACO-based Multi-Target MTS Planner - MPE Continuous Space Adaptation
Adapted for Multi-Agent Particle Environment (MPE) with continuous action space.

Key differences from discrete grid version:
- Continuous 2x2 world discretized into 100x100 grid for belief representation
- UAV positions are continuous floats, not discrete grid cells
- Actions are velocity vectors with magnitude and direction
- Belief maps remain discretized for probability representation
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class Config:
    """Configuration parameters for the ACO-MTS planner (MPE continuous space)."""
    # World parameters
    world_size: float = 2.0      # MPE world: 2x2 continuous space
    grid_size: int = 100         # Discretization: 100x100 grid for belief map
    
    # Agent parameters
    num_targets: int = 2
    num_uavs: int = 2
    max_speed: float = 0.05      # Maximum UAV speed per time step (MPE default)
    dt: float = 0.1              # Time step duration
    
    # ACO parameters
    num_ants: int = 20
    num_iterations: int = 50
    alpha: float = 1.0           # Pheromone weight
    beta: float = 2.0            # Heuristic weight
    rho: float = 0.1             # Pheromone evaporation rate
    q: float = 1.0               # Pheromone reinforcement constant (Eq. 11: Δτ = 1/ET)
    tau_max: float = 1.0
    tau_min: float = 0.001
    
    # Action discretization for pheromone encoding
    num_action_directions: int = 8   # 8 directions for pheromone trails
    
    # Sensor parameters
    sensor_range: float = 0.4        # Fixed detection range (in world coordinates)
    sensor_fidelity: float = 0.95    # Detection probability within range


class SensorModel:
    """Fixed-range sensor model for target detection."""
    
    def __init__(self, sensor_range: float = 0.4, sensor_fidelity: float = 0.95):
        """
        Initialize sensor model with fixed detection range.
        
        Args:
            sensor_range: Maximum detection range (distance threshold)
            sensor_fidelity: Detection probability within range (ps)
        """
        self.sensor_range = sensor_range
        self.sensor_fidelity = sensor_fidelity
    
    def detection_probability(self, distance: float) -> float:
        """
        Calculate detection probability based on distance.
        
        Model:
            P(D|d) = ps (sensor_fidelity) if d <= r (sensor_range)
            P(D|d) = 0                     if d > r
        
        Args:
            distance: Euclidean distance from sensor to target (scalar or array)
            
        Returns:
            Detection probability in [0, 1]
        """
        # Support both scalar and array inputs
        if np.isscalar(distance):
            return self.sensor_fidelity if distance <= self.sensor_range else 0.0
        else:
            # Vectorized version for numpy arrays
            return np.where(distance <= self.sensor_range, self.sensor_fidelity, 0.0)


class BeliefMap:
    """Recursive Bayesian belief map for a single target (Eq. 6-7) - MPE Continuous Space."""
    
    def __init__(self, grid_size: int, world_size: float, initial_mean: np.ndarray = None, 
                 initial_cov_val: float = 0.2):
        """
        Initialize belief map with Gaussian prior for continuous world.
        
        Args:
            grid_size: Discretization resolution (e.g., 100x100)
            world_size: Physical world size (e.g., 2.0 for 2x2 world)
            initial_mean: Initial mean position in world coordinates [x, y]
            initial_cov_val: Initial covariance (std dev) in world units
        """
        self.grid_size = grid_size
        self.world_size = world_size
        self.cell_size = world_size / grid_size  # Size of each grid cell
        self.belief = np.zeros((grid_size, grid_size))
        
        # Sensor model will be set by environment
        self.sensor_model = None
        
        # Pre-compute grid coordinates for efficiency (avoid repeated meshgrid creation)
        # Use indexing='ij' for consistency: [i,j] -> [x,y] mapping
        i_indices = np.arange(grid_size)
        j_indices = np.arange(grid_size)
        self.grid_I, self.grid_J = np.meshgrid(i_indices, j_indices, indexing='ij')
        
        # Pre-compute world coordinates for each grid cell CENTER (not corner)
        # Add 0.5 to center the grid cell instead of using lower-left corner
        self.grid_world_x = (self.grid_I + 0.5) * self.cell_size - self.world_size / 2
        self.grid_world_y = (self.grid_J + 0.5) * self.cell_size - self.world_size / 2
        
        # Initialize with Gaussian prior
        if initial_mean is not None:
            self._initialize_gaussian(initial_mean, initial_cov_val)
        else:
            self._initialize_uniform()
    
    def _initialize_gaussian(self, mean: np.ndarray, cov: float):
        """Initialize belief map with 2D Gaussian distribution in continuous world."""
        # Convert world coordinates to grid indices
        mean_grid = self._world_to_grid(mean)
        
        # Use pre-computed grid indices (consistent with indexing='ij')
        # 2D Gaussian: N(mean, cov*I) in grid space
        dist_sq = (self.grid_I - mean_grid[0])**2 + (self.grid_J - mean_grid[1])**2
        self.belief = np.exp(-dist_sq / (2 * (cov / self.cell_size)**2))
        self.belief /= np.sum(self.belief)  # Normalize
    
    def _initialize_uniform(self):
        """Initialize belief map with uniform distribution for unknown environment."""
        self.belief = np.ones((self.grid_size, self.grid_size))
        self.belief /= np.sum(self.belief)  # Normalize so sum is 1.0
    
    def _world_to_grid(self, world_pos: np.ndarray) -> np.ndarray:
        """Convert continuous world coordinates to discrete grid indices."""
        # MPE world: [-world_size/2, world_size/2] -> Grid: [0, grid_size-1]
        # Subtract 0.5 to align continuous coordinates with integer cell centers
        grid_pos = (world_pos + self.world_size / 2) / self.cell_size - 0.5
        grid_pos = np.clip(grid_pos, 0, self.grid_size - 1)
        return grid_pos
    
    def _grid_to_world(self, grid_pos: np.ndarray) -> np.ndarray:
        """Convert discrete grid indices to continuous world coordinates."""
        world_pos = grid_pos * self.cell_size - self.world_size / 2
        return world_pos
    
    def update(self, uav_position: np.ndarray, detection: bool):
        """
        Recursive Bayesian update (Eq. 6-7) for continuous world positions.
        VECTORIZED: uses pre-computed grid coordinates and numpy broadcasting.
        
        Args:
            uav_position: Current UAV position in world coordinates [x, y]
            detection: Whether target was detected
        """
        # Use pre-computed world coordinates (no meshgrid creation!)
        # Compute distances from UAV to all cells (vectorized)
        dx = self.grid_world_x - uav_position[0]
        dy = self.grid_world_y - uav_position[1]
        distances = np.sqrt(dx**2 + dy**2)
        
        # Compute detection probabilities for all cells at once
        p_detect = self.sensor_model.detection_probability(distances)
        
        # Compute likelihood based on detection/no-detection
        if detection:
            likelihood = p_detect
        else:
            likelihood = 1.0 - p_detect
        
        # Bayesian update: belief ∝ likelihood * prior_belief
        self.belief *= likelihood
        
        # Normalize only if detection occurred
        if detection:
            self.belief /= (np.sum(self.belief) + 1e-10)
    
    def predict(self):
        """
        Belief prediction step (Algorithm 1 Line 11).
        
        For static targets: prediction is identity transformation.
        For moving targets: would apply motion model here.
        
        This method is kept for architectural completeness and future extensibility.
        """
        # Static target assumption: no prediction needed (identity)
        pass
    
    def copy(self):
        """Create a deep copy of this belief map for ant's dynamic belief tracking."""
        new_belief_map = BeliefMap.__new__(BeliefMap)
        new_belief_map.grid_size = self.grid_size
        new_belief_map.world_size = self.world_size
        new_belief_map.cell_size = self.cell_size
        new_belief_map.belief = self.belief.copy()  # Deep copy the belief array
        new_belief_map.sensor_model = self.sensor_model  # Share sensor model
        
        # Share pre-computed grid coordinates (reference, not deep copy)
        new_belief_map.grid_I = self.grid_I
        new_belief_map.grid_J = self.grid_J
        new_belief_map.grid_world_x = self.grid_world_x
        new_belief_map.grid_world_y = self.grid_world_y
        
        return new_belief_map


class Environment:
    """Continuous environment with multiple targets and discretized belief maps (MPE adaptation)."""
    
    def __init__(self, config: Config, targets: List[np.ndarray]):
        """
        Initialize continuous environment.
        
        Args:
            config: Configuration parameters
            targets: List of target positions in world coordinates [x, y]
        """
        self.config = config
        self.grid_size = config.grid_size
        self.world_size = config.world_size
        self.targets = targets
        self.num_targets = len(targets)
        
        # Belief maps: one per target (discretized for probability representation)
        self.belief_maps = []
        for i in range(self.num_targets):
            self.belief_maps.append(
                BeliefMap(config.grid_size, config.world_size, initial_mean=None)
            )
        
        # Initialize sensor model with config parameters
        self.sensor_model = SensorModel(
            sensor_range=config.sensor_range,
            sensor_fidelity=config.sensor_fidelity
        )
        
        # Set sensor model for all belief maps
        for belief_map in self.belief_maps:
            belief_map.sensor_model = self.sensor_model
    
    def detect_target(self, uav_pos: np.ndarray, target_idx: int) -> bool:
        """
        Stochastic detection based on sensor model.
        
        Args:
            uav_pos: UAV position
            target_idx: Index of target
            
        Returns:
            Boolean indicating detection
        """
        distance = np.linalg.norm(uav_pos - self.targets[target_idx])
        prob = self.sensor_model.detection_probability(distance)
        return np.random.rand() < prob


class ACOPlanner:
    """Max-Min Ant System for multi-target MTS with continuous action space (MPE adaptation)."""
    
    def __init__(self, config: Config, environment: Environment):
        """
        Initialize ACO planner with continuous action space and discretized pheromone.
        
        Args:
            config: Configuration parameters
            environment: Environment instance
        """
        self.config = config
        self.env = environment
        self.world_size = config.world_size
        self.grid_size = config.grid_size
        
        # Define action set: discretized directions for pheromone encoding
        # Actions represent DIRECTION (angle), speed is applied separately
        # 8 directions: 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW
        self.action_angles = np.linspace(0, 2*np.pi, config.num_action_directions, endpoint=False)
        self.num_actions = config.num_action_directions
        
        # Maximum planning horizon (time steps)
        self.max_time = 50  # Longer horizon for continuous space
        
        # Pheromone trails: tau[action_direction, time, uav]
        # Shape: (num_action_directions, max_time, num_uavs)
        self.pheromone = np.ones((self.num_actions, self.max_time, config.num_uavs)) * 0.1
        
        # UAV starting positions (continuous world coordinates)
        self.uav_positions = [
            np.array([-0.9, -0.9]),  # Bottom-left corner
            np.array([0.9, 0.9])     # Top-right corner
        ][:config.num_uavs]
        
        # Track convergence
        self.et_history = []
        self.best_solution = None
        self.best_et = float('inf')
    
    def _compute_expected_time(self, joint_trajectories: Dict[int, List[np.ndarray]]) -> float:
        """
        Compute Expected Time to find ALL targets (Eq. 8) - OPTIMIZED O(N) version.
        
        Equation 8: ET = Σ_{t=0}^{N} [1 - P(all targets found by time t)]
        
        OPTIMIZATION: Incremental computation instead of re-scanning history.
        Maintains cumulative "not found" probability for each target.
        
        Complexity: O(N * K * M) where N=steps, K=targets, M=UAVs
        (Previously O(N^2 * K * M) - 100x slower for N=200)
        
        Args:
            joint_trajectories: Dict mapping uav_id -> list of positions over time
            
        Returns:
            Expected time value (lower is better)
        """
        if not joint_trajectories or len(joint_trajectories) == 0:
            return float('inf')
        
        # Find maximum trajectory length
        max_length = max(len(traj) for traj in joint_trajectories.values())
        if max_length == 0:
            return float('inf')
        
        et = 0.0
        debug_mode = False  # Set to True for detailed debugging
        
        # Use belief map coordinates for vectorized calculation
        # Assuming all belief maps share the same grid structure
        grid_world_x = self.env.belief_maps[0].grid_world_x
        grid_world_y = self.env.belief_maps[0].grid_world_y
        
        # Initialize cumulative "no detection" probability map for the whole grid
        # cum_no_detect_map[i, j] = P(Target at (i,j) is NOT detected by time t)
        # Initially 1.0 everywhere
        cum_no_detect_map = np.ones_like(self.env.belief_maps[0].belief)
        
        # Incremental computation: update probabilities at each time step
        for t in range(max_length):
            # 1. Calculate detection probability map for this time step (Union of all UAVs)
            # step_no_detect_map[i, j] = P(No detection at (i,j) at time t)
            step_no_detect_map = np.ones_like(cum_no_detect_map)
            
            for uav_id, trajectory in joint_trajectories.items():
                if t < len(trajectory):
                    uav_pos = trajectory[t]
                    
                    # Vectorized distance calculation to all grid cells
                    dx = grid_world_x - uav_pos[0]
                    dy = grid_world_y - uav_pos[1]
                    dists = np.sqrt(dx**2 + dy**2)
                    
                    # P(Detect | Target at cell)
                    p_detect_grid = self.env.sensor_model.detection_probability(dists)
                    
                    # Update step no-detection map
                    step_no_detect_map *= (1.0 - p_detect_grid)
            
            # Update cumulative no-detection map
            cum_no_detect_map *= step_no_detect_map
            
            # Compute P(all targets found by time t)
            # = Π_{k=1}^{K} [1 - P(target k not found)]
            
            prob_all_found = 1.0
            for belief_map in self.env.belief_maps:
                # P(Target k NOT found) = Sum(Belief_k(cell) * P(Not Detected | cell))
                # This integrates the "not found" probability over the belief distribution
                p_target_not_found = np.sum(belief_map.belief * cum_no_detect_map)
                prob_all_found *= (1.0 - p_target_not_found)
            
            # ET contribution at time t
            et_contribution = 1.0 - prob_all_found
            et += et_contribution
            
            # Debug output
            if debug_mode and t < 5:
                print(f"  t={t}: P(all found)={prob_all_found:.6f}, "
                      f"ET contrib={et_contribution:.6f}, cumulative ET={et:.6f}")
        
        if debug_mode:
            print(f"  Final: max_length={max_length}, total ET={et:.4f}\n")
        
        return et
    
    def _compute_heuristic_info(self, current_pos: np.ndarray, action_idx: int,
                                ant_beliefs: List, ant_weights: Dict[int, float]) -> float:
        """
        Compute Multi-target MTS Heuristic for continuous action (Eq. 14-15 adapted).
        OPTIMIZED: uses pre-computed grid coordinates from belief maps.
        
        Args:
            current_pos: Current position in world coordinates [x, y]
            action_idx: Index of action direction (0 to num_actions-1)
            ant_beliefs: List of current belief maps for this ant (dynamic)
            ant_weights: Dict of fixed weights {target_idx: weight}
            
        Returns:
            Heuristic value (higher is better)
        """
        # Get action direction from angle
        angle = self.action_angles[action_idx]
        action_dir = np.array([np.cos(angle), np.sin(angle)])
        
        # Cone parameters
        cone_angle = np.pi / 3  # 60 degrees
        max_range = self.world_size * 0.5
        min_dist = 0.01
        
        total_heuristic = 0.0
        
        for target_idx, belief_map in enumerate(ant_beliefs):
            # Use pre-computed world coordinates (no meshgrid creation!)
            grid_world_x = belief_map.grid_world_x
            grid_world_y = belief_map.grid_world_y
            
            # Vectors from current position to all cells
            vec_x = grid_world_x - current_pos[0]
            vec_y = grid_world_y - current_pos[1]
            distances = np.sqrt(vec_x**2 + vec_y**2)
            
            # Filter by distance range
            valid_dist_mask = (distances >= min_dist) & (distances <= max_range)
            
            # Normalize direction vectors (avoid division by zero)
            with np.errstate(divide='ignore', invalid='ignore'):
                dir_x = vec_x / distances
                dir_y = vec_y / distances
                dir_x = np.where(valid_dist_mask, dir_x, 0)
                dir_y = np.where(valid_dist_mask, dir_y, 0)
            
            # Compute dot product with action direction
            cos_angles = dir_x * action_dir[0] + dir_y * action_dir[1]
            cos_angles = np.clip(cos_angles, -1.0, 1.0)
            
            # Filter by cone angle
            angles = np.arccos(cos_angles)
            cone_mask = (angles <= cone_angle / 2) & valid_dist_mask
            
            # Compute distance weights for cells in cone
            distance_weights = 1.0 / (1.0 + distances * 10)
            
            # Sum weighted beliefs in cone
            belief_sum = np.sum(belief_map.belief[cone_mask] * distance_weights[cone_mask])
            
            # Apply target weight
            w_k = ant_weights.get(target_idx, 1.0)
            total_heuristic += w_k * belief_sum
        
        return max(total_heuristic, 1e-6)
    
    def _construct_solution(self) -> Tuple[Dict[int, List[np.ndarray]], Dict[int, List[int]]]:
        """
        Construct joint solution for ALL UAVs with continuous actions (MPE adaptation).
        
        Key MPE adaptations:
        - UAV positions are continuous floats in world coordinates
        - Actions are velocity vectors: direction from pheromone, magnitude = max_speed
        - Positions are clipped to world bounds [-world_size/2, world_size/2]
        - Belief maps remain discretized for probability representation
        
        Returns:
            (joint_trajectories, joint_actions):
                - joint_trajectories: Dict[uav_id] -> List of continuous positions
                - joint_actions: Dict[uav_id] -> List of action direction indices
        """
        # Algorithm 1 Line 5: Generate random weights for this ant (FIXED throughout)
        ant_weights = {k: np.random.uniform(0.0, 1.0) 
                      for k in range(self.env.num_targets)}
        
        # Copy initial belief maps for this ant (will be updated dynamically)
        ant_beliefs = [belief_map.copy() for belief_map in self.env.belief_maps]
        
        joint_trajectories = {}
        joint_actions = {}
        
        # Initialize each UAV at its starting position (continuous coords)
        for uav_id in range(self.config.num_uavs):
            joint_trajectories[uav_id] = [self.uav_positions[uav_id].copy()]
            joint_actions[uav_id] = []
        
        # Synchronous time steps
        for t in range(self.max_time):
            # Algorithm 1 Line 11: Belief Prediction (before action selection)
            for target_idx in range(self.env.num_targets):
                ant_beliefs[target_idx].predict()
            
            # Each UAV selects an action at time t
            for uav_id in range(self.config.num_uavs):
                current_pos = joint_trajectories[uav_id][-1]
                
                # Compute selection probabilities for each action direction
                probabilities = []
                for action_idx in range(self.num_actions):
                    # Pheromone value: tau[action_direction, time, uav]
                    tau = self.pheromone[action_idx, min(t, self.max_time-1), uav_id]
                    
                    # Heuristic using ant's CURRENT beliefs and FIXED weights
                    eta = self._compute_heuristic_info(current_pos, action_idx, 
                                                       ant_beliefs, ant_weights)
                    
                    # ACO probability: tau^alpha * eta^beta
                    prob = (tau ** self.config.alpha) * (eta ** self.config.beta)
                    probabilities.append(prob)
                
                # Normalize probabilities
                probabilities = np.array(probabilities)
                if np.sum(probabilities) > 1e-10:
                    probabilities /= np.sum(probabilities)
                else:
                    probabilities = np.ones(self.num_actions) / self.num_actions
                
                # Select action direction
                selected_action = np.random.choice(self.num_actions, p=probabilities)
                
                # Execute action: apply velocity in selected direction
                angle = self.action_angles[selected_action]
                velocity = self.config.max_speed * np.array([np.cos(angle), np.sin(angle)])
                new_pos = current_pos + velocity
                
                # Clip to world bounds
                half_world = self.world_size / 2
                new_pos = np.clip(new_pos, -half_world, half_world)
                
                joint_trajectories[uav_id].append(new_pos)
                joint_actions[uav_id].append(selected_action)
            
            # Algorithm 1 Line 11-12: Update beliefs after all UAVs move
            for target_idx in range(self.env.num_targets):
                for uav_id in range(self.config.num_uavs):
                    uav_pos = joint_trajectories[uav_id][-1]
                    # Assume no detection at this position (Line 12)
                    ant_beliefs[target_idx].update(uav_pos, detection=False)
            
            # Early termination: check if all targets likely found
            if t > 50 and t % 10 == 0:
                et_so_far = self._compute_expected_time(joint_trajectories)
                if et_so_far < t * 0.1:  # Very good progress
                    break
        
        return joint_trajectories, joint_actions
    

    def _update_pheromone(self, all_joint_trajectories: List[Dict[int, List[np.ndarray]]], 
                          all_joint_actions: List[Dict[int, List[int]]],
                          fitness_values: List[float]):
        """
        Update pheromone trails on tau[action, time, uav] using MMAS rules (Eq. 11-12).
        
        Eq. 11: Deposit pheromone on best solution's action sequence
        Eq. 12: Evaporation on all entries
        
        Args:
            all_joint_trajectories: List of joint trajectory solutions
            all_joint_actions: List of joint action sequences
            fitness_values: ET values for each solution
        """
        # Evaporation (Eq. 12): tau = (1 - rho) * tau
        self.pheromone *= (1.0 - self.config.rho)
        
        # Find best solution (minimum ET)
        best_idx = np.argmin(fitness_values)
        best_actions = all_joint_actions[best_idx]
        best_fitness = fitness_values[best_idx]
        
        # Update global best
        if best_fitness < self.best_et:
            self.best_et = best_fitness
            self.best_solution = all_joint_trajectories[best_idx]
        
        # Deposit pheromone on best solution (Eq. 11: Δτ = 1/ET)
        # Paper's formula: Δτ = 1 / ET
        delta_tau = 1.0 / max(best_fitness, 0.01)  # Avoid division by zero
        
        # Reinforce best solution's action sequence
        for uav_id, action_sequence in best_actions.items():
            for t, action_idx in enumerate(action_sequence):
                if action_idx >= 0 and t < self.max_time:  # Valid action
                    self.pheromone[action_idx, t, uav_id] += delta_tau
        
        # Apply MMAS bounds: tau_min <= tau <= tau_max
        self.pheromone = np.clip(
            self.pheromone,
            self.config.tau_min,
            self.config.tau_max
        )
    
    def optimize(self) -> np.ndarray:
        """
        Run MMAS optimization for specified number of iterations.
        
        Paper's Algorithm 1:
        1. Each ant constructs a joint solution (all UAVs)
        2. Evaluate using ET (Eq. 8)
        3. Update pheromone on tau[action, time, uav] (Eq. 11-12)
        
        Returns:
            Array of best ET values over iterations
        """
        print("\nStarting ACO-MTS Optimization...")
        print(f"Pheromone structure: tau[{self.num_actions} actions, {self.max_time} time steps, {self.config.num_uavs} UAVs]")
        
        for iteration in range(self.config.num_iterations):
            all_joint_trajectories = []
            all_joint_actions = []
            fitness_values = []
            
            # Debug first iteration
            debug_first_ant = (iteration == 0)
            
            # Each ant constructs a joint solution for ALL UAVs
            for ant_idx in range(self.config.num_ants):
                if debug_first_ant and ant_idx == 0:
                    print(f"\n[DEBUG] First ant of first iteration:")
                
                joint_traj, joint_act = self._construct_solution()
                all_joint_trajectories.append(joint_traj)
                all_joint_actions.append(joint_act)
                
                if debug_first_ant and ant_idx == 0:
                    traj_lens = [len(joint_traj[uid]) for uid in range(self.config.num_uavs)]
                    print(f"  Trajectory lengths: {traj_lens}")
                
                # Evaluate fitness using Eq. 8
                et = self._compute_expected_time(joint_traj)
                fitness_values.append(et)
                
                if debug_first_ant and ant_idx == 0:
                    print(f"  Computed ET: {et:.6f}")
            
            # Update pheromone using MMAS rules
            self._update_pheromone(all_joint_trajectories, all_joint_actions, fitness_values)
            
            # Track best ET
            best_et = np.min(fitness_values)
            mean_et = np.mean(fitness_values)
            self.et_history.append(best_et)
            
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}: Best ET = {best_et:.4f}, Mean ET = {mean_et:.4f}")
        
        print(f"\nOptimization complete! Final best ET = {self.best_et:.4f}")
        return np.array(self.et_history)


def main():
    """Main execution: run ACO-MTS planner with MPE continuous space."""
    
    # Configuration for MPE continuous space
    config = Config(
        world_size=2.0,          # MPE: 2x2 continuous world
        grid_size=100,           # Discretization for belief maps
        num_targets=2,
        num_uavs=2,
        max_speed=0.05,          # MPE default max speed
        dt=0.1,
        num_ants=20,
        num_iterations=50,
        num_action_directions=8
    )
    
    # Define target positions in world coordinates (MPE: [-1, 1] range)
    targets = [
        np.array([0.0, 0.5]),    # Target 1 in upper-middle area
        np.array([0.5, 0.0])     # Target 2 in right-middle area
    ]
    
    # Create environment and planner
    env = Environment(config, targets)
    planner = ACOPlanner(config, env)
    
    print("=" * 60)
    print("ACO-MTS Planner - MPE Continuous Space Adaptation")
    print("=" * 60)
    print(f"World Size: {config.world_size}x{config.world_size} (continuous)")
    print(f"Grid Discretization: {config.grid_size}x{config.grid_size}")
    print(f"Cell Size: {config.world_size/config.grid_size:.4f}")
    print(f"Max UAV Speed: {config.max_speed}")
    print(f"Sensor Range: {config.sensor_range}")
    print(f"Sensor Fidelity: {config.sensor_fidelity}")
    print(f"Number of Targets: {config.num_targets}")
    print(f"Number of UAVs: {config.num_uavs}")
    print(f"Number of Iterations: {config.num_iterations}")
    print("=" * 60)
    
    # Verify sensor model
    print("\nSensor Model Verification (Fixed Range Model):")
    print(f"Detection Range: r = {config.sensor_range}")
    print(f"Detection Fidelity: ps = {config.sensor_fidelity}")
    print("Distance vs Detection Probability:")
    for d in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]:
        p = env.sensor_model.detection_probability(d)
        status = "✓ IN RANGE" if d <= config.sensor_range else "✗ OUT OF RANGE"
        print(f"  d = {d:.1f}: P(detect) = {p:.4f}  {status}")
    print()
    
    # Run optimization
    et_values = planner.optimize()
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Convergence curve
    ax = axes[0]
    ax.plot(et_values, 'b-', linewidth=2, label='Best ET')
    ax.fill_between(range(len(et_values)), et_values, alpha=0.3)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Expected Time (ET)', fontsize=12)
    ax.set_title('ACO-MTS Convergence (MPE Continuous)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Plot 2: Best solution trajectories in continuous space
    ax = axes[1]
    
    # Show combined belief map
    combined_belief = np.zeros((config.grid_size, config.grid_size))
    for belief_map in env.belief_maps:
        combined_belief += belief_map.belief / len(env.belief_maps)
    
    # Create extent for imshow to match world coordinates
    extent = [-config.world_size/2, config.world_size/2, 
              -config.world_size/2, config.world_size/2]
    im = ax.imshow(combined_belief, cmap='hot', origin='lower', 
                   alpha=0.6, extent=extent)
    
    # Plot best solution trajectories (continuous positions)
    if planner.best_solution is not None:
        colors = ['green', 'cyan', 'yellow', 'magenta']
        for uav_id, trajectory in planner.best_solution.items():
            traj_array = np.array(trajectory)
            ax.plot(traj_array[:, 0], traj_array[:, 1], 
                   marker='o', markersize=2, linewidth=1.5, 
                   color=colors[uav_id % len(colors)],
                   label=f'UAV {uav_id+1}', alpha=0.8)
    
    # Plot targets (world coordinates)
    for i, target in enumerate(targets):
        ax.plot(target[0], target[1], 'b*', markersize=20, label=f'Target {i+1}')
    
    # Plot UAV start positions (world coordinates)
    for i, uav_pos in enumerate(planner.uav_positions):
        ax.plot(uav_pos[0], uav_pos[1], 'wo', markersize=10, 
               markeredgecolor='black', markeredgewidth=2, label=f'UAV {i+1} Start')
    
    ax.set_xlabel('X (world coordinates)', fontsize=12)
    ax.set_ylabel('Y (world coordinates)', fontsize=12)
    ax.set_title('Best Solution with Continuous Trajectories', fontsize=14, fontweight='bold')
    ax.set_xlim(-config.world_size/2, config.world_size/2)
    ax.set_ylim(-config.world_size/2, config.world_size/2)
    plt.colorbar(im, ax=ax, label='Belief Probability')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/aco_mts_results_mpe.png', dpi=150)
    print("\nPlot saved to: results/aco_mts_results_mpe.png")
    
    plt.show()
    
    # Print final results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Initial ET: {et_values[0]:.4f}")
    print(f"Final ET: {et_values[-1]:.4f}")
    print(f"Improvement: {(et_values[0] - et_values[-1]) / et_values[0] * 100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    main()
