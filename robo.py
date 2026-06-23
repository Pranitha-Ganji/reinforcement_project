import streamlit as st
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

# --- 1. Enhanced Environment with Reward Shaping ---
class DroneNavigationEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # Actions: Continuous acceleration adjustments [-1.0, 1.0]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # Observations: Drone X, Drone Y, Target X, Target Y
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(4,), dtype=np.float32)
        
        self.obstacle = np.array([0.0, 0.0, 1.3], dtype=np.float32) # Centered obstacle
        self.target_pos = np.array([4.0, 4.0], dtype=np.float32)   # Destination
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Start at a predictable bottom-left quadrant to stabilize early trajectories
        self.drone_pos = np.array([-4.0, -4.0], dtype=np.float32)
        self.prev_dist = np.linalg.norm(self.drone_pos - self.target_pos)
        self.steps = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate([self.drone_pos, self.target_pos]).astype(np.float32)

    def step(self, action):
        self.steps += 1
        
        # Action scaling: Smooth out radical jumps to keep drone away from hard limits
        dt = 0.15
        self.drone_pos += np.clip(action, -1.0, 1.0) * dt
        self.drone_pos = np.clip(self.drone_pos, -5.0, 5.0)
        
        dist_to_target = np.linalg.norm(self.drone_pos - self.target_pos)
        dist_to_obstacle = np.linalg.norm(self.drone_pos - self.obstacle[:2])
        
        # --- CRITICAL: SHAPED REWARD MECHANISM ---
        # Reward the agent for getting closer to the target compared to last step
        reward = (self.prev_dist - dist_to_target) * 2.0
        self.prev_dist = dist_to_target
        
        # Minor step tax to keep actions efficient
        reward -= 0.02
        
        terminated = False
        truncated = self.steps >= 150
        
        # High value milestone rewards
        if dist_to_target < 0.6:
            reward += 20.0
            terminated = True
            
        if dist_to_obstacle < self.obstacle[2]:
            reward -= 10.0
            terminated = True  # Hard penalty forces it away from obstacle
            
        return self._get_obs(), float(reward), terminated, truncated, {}

class IntrinsicCuriosityWrapper(gym.Wrapper):
    def __init__(self, env, curiosity_weight=0.1):
        super().__init__(env)
        self.curiosity_weight = curiosity_weight
        self.visitation_counts = {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Create grid map chunks to log exploration paths
        grid_key = (round(obs[0], 1), round(obs[1], 1))
        self.visitation_counts[grid_key] = self.visitation_counts.get(grid_key, 0) + 1
        
        # Curiosity kicks in when progress stalls
        intrinsic_reward = 1.0 / np.sqrt(self.visitation_counts[grid_key])
        total_reward = reward + (self.curiosity_weight * intrinsic_reward)
        return obs, total_reward, terminated, truncated, info

# --- 2. Streamlit Dynamic UI Construction ---
st.set_page_config(page_title="Advanced Drone Reinforcement Learning", layout="wide")
st.title("🛸 Optimized Drone Navigation Dashboard")
st.markdown("Fixing border-clinging patterns via **Distance-Based Reward Shaping** and **Action Mitigation**.")

st.sidebar.header("Agent Options")
learning_rate = st.sidebar.slider("Learning Rate", 1e-4, 1e-3, 3e-4, format="%.4f")
curiosity_weight = st.sidebar.slider("Exploration Curiosity Boost", 0.0, 0.5, 0.15)
total_timesteps = st.sidebar.number_input("Total Timesteps", 10000, 100000, 30000, step=10000)

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Control Node")
    train_button = st.button("🚀 Execute Guided Training Run")
    status_text = st.empty()
    progress_bar = st.progress(0.0)
with col2:
    st.subheader("Dynamic Flight Trajectory")
    plot_spot = st.empty()

def plot_trajectory(model, env):
    obs, _ = env.reset()
    trajectory = [obs[0:2].copy()]
    
    for _ in range(150):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        trajectory.append(obs[0:2].copy())
        if terminated or truncated:
            break
            
    trajectory = np.array(trajectory)
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(trajectory[:, 0], trajectory[:, 1], '-o', label='Drone Vector', color='#1f77b4', markersize=3, linewidth=1.5)
    ax.scatter(-4.0, -4.0, color='orange', marker='o', s=100, label='Spawn point')
    ax.scatter(4.0, 4.0, color='green', marker='X', s=150, label='Target')
    
    # Obstacle mapping
    circle = plt.Circle((0, 0), 1.3, color='red', alpha=0.25, label='Danger Zone')
    ax.add_patch(circle)
    
    ax.set_xlim(-5.2, 5.2)
    ax.set_ylim(-5.2, 5.2)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right')
    
    plot_spot.pyplot(fig)
    plt.close(fig)

# --- 3. Run Executions ---
if train_button:
    status_text.info("Building Environment Systems...")
    raw_env = DroneNavigationEnv()
    env = IntrinsicCuriosityWrapper(raw_env, curiosity_weight=curiosity_weight)
    
    # PPO hyper-parameters adjusted for continuous locomotion
    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=learning_rate,
        n_steps=512, # Frequent collection updates
        batch_size=64,
        n_epochs=8,
        ent_coef=0.01, # Extra entropy forcing action variance away from edges
        verbose=0
    )
    
    steps_per_loop = 2048
    loops = int(total_timesteps / steps_per_loop)
    
    for i in range(loops):
        status_text.warning(f"Optimizing Policy Network: Generation {i+1}/{loops}...")
        model.learn(total_timesteps=steps_per_loop, reset_num_timesteps=False)
        progress_bar.progress((i + 1) / loops)
        plot_trajectory(model, env)
        
    status_text.success("🎯 Optimization Sequence Concluded!")