import retro
from gym import wrappers
import time
import os
from stable_baselines3 import PPO

def create_env():
    ROM_FOLDER = "/home/nos/required_files_for_docker_sonic/Roms/SonicAdvance2-GbAdvance"
    retro.data.Integrations.add_custom_path(ROM_FOLDER)
    env = retro.make(game='SonicAdvance2-GbAdvance', state='Act1', inttype=retro.data.Integrations.ALL)
    env = wrappers.Monitor(env, './video', force=True, video_callable=lambda episode_id: True)
    return env

def load_latest_model(model_directory='./models'):
    # List all files in the model directory
    files = [os.path.join(model_directory, f) for f in os.listdir(model_directory)]
    # Filter out files that are not model files
    model_files = [f for f in files if f.endswith('.zip') and os.path.isfile(f)]
    # Sort files by modification time, newest first
    latest_model = sorted(model_files, key=os.path.getmtime, reverse=True)[0]
    print(f"Loading model from {latest_model}")
    return PPO.load(latest_model)

def main():
    env = create_env()
    model = load_latest_model()  # Load the latest model
    observation = env.reset()
    done = False
    total_reward = 0

    while not done:
        env.render()  # Display the game window
        action, _states = model.predict(observation, deterministic=True)  # Use the model to predict the action
        observation, reward, done, info = env.step(action)  # Apply the action
        total_reward += reward

        print(f"Step reward: {reward}, Total reward: {total_reward}")
        time.sleep(0)  # Delay to slow down the action

        if done:
            print("Episode completed. Total reward:", total_reward)
            env.reset()  # Reset the environment for a new episode

    env.close()  # Close the environment properly

if __name__ == "__main__":
    main()
