import retro
from gym import wrappers
import time

def create_env():
    # Define the path to the custom ROMs directory
    ROM_FOLDER = "/home/nos/required_files_for_docker_sonic/Roms/SonicAdvance2-GbAdvance"
    # Add the custom path to Retro's known directories
    retro.data.Integrations.add_custom_path(ROM_FOLDER)

    # Make sure that your environment is properly recognized by specifying the game and state
    env = retro.make(game='SonicAdvance2-GbAdvance', state='Act1', inttype=retro.data.Integrations.ALL)

    # Wrap the environment to record video for each episode
    env = wrappers.Monitor(env, './video', force=True, video_callable=lambda episode_id: True)
    return env

def main():
    env = create_env()
    observation = env.reset()
    done = False
    total_reward = 0

    while not done:
        env.render()  # Display the game window
        action = env.action_space.sample()  # Select a random action
        observation, reward, done, info = env.step(action)  # Apply the action
        total_reward += reward

        # Print rewards and delay each step to observe the agent's behavior
        print(f"Step reward: {reward}, Total reward: {total_reward}")
        time.sleep(0)  # Delay for 100 milliseconds to slow down the action

        if done:
            print("Episode completed. Total reward:", total_reward)
            env.reset()  # Reset the environment for a new episode

    env.close()  # Close the environment properly

if __name__ == "__main__":
    main()
