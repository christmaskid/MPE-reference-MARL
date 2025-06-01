import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv('training_log_3_agent_broadcast.csv')  # Replace with your actual file name

# Plot avg_reward vs epoch
plt.figure(figsize=(15, 4))
# plt.plot(df['epoch'], df['avg_reward'])
# plt.xlabel('Episode')
# plt.ylabel('Return')
# plt.title('Average Return')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.savefig("return_curve.png")
# plt.show()


# Reward
plt.subplot(1, 3, 1)
plt.xlabel("Episode")
plt.ylabel("Return")
plt.plot(df['epoch'], df['avg_reward'], label='reward')
plt.title("Average Return")
plt.grid(True)

# Actor loss
plt.subplot(1, 3, 2)
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.plot(df['epoch'], df['actor_loss'], label='Actor Loss', color='orange')
plt.title("Actor Loss")
plt.grid(True)

# Critic loss
plt.subplot(1, 3, 3)
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.plot(df['epoch'], df['critic_loss'], label='Critic Loss', color='green')
plt.title("Critic Loss")
plt.grid(True)

plt.tight_layout()
plt.savefig("training_curves.png")  # or plt.show() if you prefer
plt.close()
