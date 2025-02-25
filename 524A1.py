import pygame
import numpy as np
import random
from collections import defaultdict
import datetime
import pandas as pd
import matplotlib.pyplot as plt

#maze layout
CELL_SIZE = 40
MAZE = [
    "S..##  T",  # 8 characters
    "##  #   ",   # 8 characters
    "  # ### ",   # 8 characters
    "##    # ",   # 8 characters
    "T  G#   "    # 8 characters
]

# Color definitions
COLORS = {
    "#": (0, 0, 0),        # Walls (black)
    " ": (255, 255, 255),  # Path (white)
    "S": (0, 255, 0),      # Start (green)
    "G": (255, 0, 0),      # Goal (red)
    "T": (0, 0, 255)       # Traps (blue)
}

class MazeEnv:
    def __init__(self):
        # Verify consistent maze dimensions
        row_lengths = [len(row) for row in MAZE]
        assert len(set(row_lengths)) == 1, "Error: Inconsistent maze row lengths"
        
        self.maze = np.array([list(row) for row in MAZE])
        self.start_pos = None
        self.goal_pos = None
        self.traps = []
        self._parse_maze()
        self.agent_pos = self.start_pos
        self.done = False
        
    def _parse_maze(self):
        """Parse the maze layout to locate key points"""
        for y in range(len(self.maze)):
            for x in range(len(self.maze[y])):
                cell = self.maze[y][x]
                if cell == 'S':
                    self.start_pos = (x, y)
                elif cell == 'G':
                    self.goal_pos = (x, y)
                elif cell == 'T':
                    self.traps.append((x, y))
    
    def reset(self):
        """Reset the environment to the initial state"""
        self.agent_pos = self.start_pos
        self.done = False
        return self.agent_pos
    
    def step(self, action):
        """Execute an action and return the new state, reward, and termination flag"""
        x, y = self.agent_pos
        new_x, new_y = x, y
        
        # Action mapping
        if action == 0:   # Up
            new_y = max(y - 1, 0)
        elif action == 1: # Down
            new_y = min(y + 1, len(self.maze) - 1)
        elif action == 2: # Left
            new_x = max(x - 1, 0)
        elif action == 3: # Right
            new_x = min(x + 1, len(self.maze[0]) - 1)
        
        # Collision detection
        if self.maze[new_y][new_x] == "#":
            return (x, y), -10, False  # Wall collision penalty
        
        # Update position
        self.agent_pos = (new_x, new_y)
        cell = self.maze[new_y][new_x]
        
        # Calculate reward
        if cell == "G":
            reward = 100
            self.done = True
        elif cell == "T":
            reward = -50
            self.agent_pos = self.start_pos  # Reset to start position
        else:
            reward = -1  # Ordinary move penalty
            
        return self.agent_pos, reward, self.done

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.5):
        self.q_table = defaultdict(lambda: [0.0]*4)  # 4 actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
    def get_action(self, state):
        """ε-greedy strategy"""
        if random.random() < self.epsilon:
            return random.randint(0, 3)  # Exploration
        else:
            return np.argmax(self.q_table[state])  # Exploitation
    
    def update(self, state, action, reward, next_state):
        """Update the Q-table"""
        max_next_q = np.max(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * (
            reward + self.gamma * max_next_q - self.q_table[state][action]
        )

def draw_env(screen, env):
    """Visualize the maze environment"""
    maze_width = len(MAZE[0]) * CELL_SIZE
    maze_height = len(MAZE) * CELL_SIZE
    
    # Draw the maze
    for y in range(len(env.maze)):
        for x in range(len(env.maze[y])):
            cell = env.maze[y][x]
            color = COLORS.get(cell, (255, 255, 255))
            pygame.draw.rect(screen, color,
                           (x * CELL_SIZE + 50, y * CELL_SIZE + 50, CELL_SIZE, CELL_SIZE), 0)
            
            # Draw grid lines
            pygame.draw.rect(screen, (200, 200, 200),
                           (x * CELL_SIZE + 50, y * CELL_SIZE + 50, CELL_SIZE, CELL_SIZE), 1)
    
    # Draw the agent
    x, y = env.agent_pos
    pygame.draw.circle(screen, (255, 192, 203),
                      (x * CELL_SIZE + 50 + CELL_SIZE//2,
                       y * CELL_SIZE + 50 + CELL_SIZE//2),
                      CELL_SIZE//3)

def draw_ui(screen, action_taken, reward_received, episode, total_reward, agent_pos, traps):
    """Draw the user interface"""
    # Draw the status information box
    pygame.draw.rect(screen, (220, 220, 220), (0, 0, 600, 50))
    font = pygame.font.SysFont(None, 24)
    
    # Display the current action
    action_text = font.render(f"Action: {action_taken}", True, (0, 0, 0))
    screen.blit(action_text, (10, 10))
    
    # Display the current reward
    reward_text = font.render(f"Reward: {reward_received}", True, (0, 0, 0))
    screen.blit(reward_text, (180, 10))
    
    # Display the training progress
    progress_text = font.render(f"Episode: {episode} | Total Reward: {total_reward}", True, (0, 0, 0))
    screen.blit(progress_text, (350, 10))
    
    # Display the agent's position
    agent_pos_text = font.render(f"Agent Position: ({agent_pos[0]}, {agent_pos[1]})", True, (0, 0, 0))
    screen.blit(agent_pos_text, (10, 350))
    
    # Display the remaining traps
    traps_remaining = len(traps)
    traps_text = font.render(f"Traps Remaining: {traps_remaining}", True, (0, 0, 0))
    screen.blit(traps_text, (350, 350))

def train():
    """Training main loop"""
    pygame.init()
    screen = pygame.display.set_mode((600, 700))
    pygame.display.set_caption("Q-Learning Maze with Enhanced UI")
    clock = pygame.time.Clock()
    
    env = MazeEnv()
    agent = QLearningAgent()
    
    total_episodes = 500
    rewards_history = []
    data = []  # Record data
    
    for episode in range(total_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        action_taken = None
        reward_received = 0
        step = 0
        
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            # Select action
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            
            # Update Q-table
            agent.update(state, action, reward, next_state)
            
            # Record data
            data.append({
                'Time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'State': state,
                'Action': action,
                'Reward': reward
            })
            
            # Update state
            state = next_state
            total_reward += reward
            action_taken = action
            reward_received = reward
            step += 1
            
            # Draw interface
            screen.fill((255, 255, 255))
            draw_env(screen.subsurface((0, 50, 600, 600)), env)
            draw_ui(screen, action_taken, reward_received, episode + 1, total_reward, env.agent_pos, env.traps)
            pygame.display.flip()
            clock.tick(25)  # Adjust frame rate to slow down training speed
        
        # Record training data
        rewards_history.append(total_reward)
        print(f"Episode {episode+1}/{total_episodes}, Total Reward: {total_reward}")
        
        # ε decay
        agent.epsilon = max(0.01, agent.epsilon * 0.995)
    
    # Save results
    pd.DataFrame(data).to_excel(f"training_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx", index=False)
    plt.plot(rewards_history)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    plt.show()

if __name__ == "__main__":
    train()
    pygame.quit()