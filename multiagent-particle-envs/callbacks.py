import numpy as np

# define done_callback: done if agent collides with its goal
def done_callback(agent, world):
    # agent.goal_b is the target landmark for agent[agent.goal_a]
    if hasattr(agent, "goal_b") and agent.goal_b is not None:
        speakto_agent = agent.goal_a
        speakto_target_pos = agent.goal_b.state.p_pos
        speakto_target_pos = speakto_agent.state.p_pos if agent.goal_a is not None else (0, 0)

        dist = np.linalg.norm(speakto_pos, speakto_target_pos)
        # consider collision if distance less than sum of sizes
        if dist < (agent.size + agent.goal_b.size):
            return True
    return False