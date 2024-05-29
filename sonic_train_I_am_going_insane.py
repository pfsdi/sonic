import os
import time
import datetime
import retro
from gym import wrappers
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

# Configuration variables
time_limit = 6000 
n_envs = 32  # Number of environments to run in parallel
batch_size = 128  # Batch size for PPO updates
n_steps = 1024  # Number of steps per environment before update
learning_rate = 3e-4  
total_timesteps = 2048
use_tensorboard = None  # Toggle for TensorBoard logging

def create_env():
    def _init():
        ROM_FOLDER = "/home/nos/required_files_for_docker_sonic/Roms/SonicAdvance2-GbAdvance"
        retro.data.Integrations.add_custom_path(ROM_FOLDER)
        env = retro.make(game='SonicAdvance2-GbAdvance', state='Act1', inttype=retro.data.Integrations.ALL)
        #env = wrappers.Monitor(env, './video', force=True, video_callable=lambda episode_id: True)
        return env
    return _init

def make_vec_env(num_envs):
    return SubprocVecEnv([create_env() for _ in range(num_envs)])

def main():
    start_time = time.time()
    env = make_vec_env(n_envs)
    
    tensorboard_log_dir = "./ppo_sonic_tensorboard/" if use_tensorboard else None
    
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log_dir,
                batch_size=batch_size, n_steps=n_steps, learning_rate=learning_rate)

    while True:
        model.learn(total_timesteps=total_timesteps)  # Learn in chunks
        elapsed_time = time.time() - start_time
        if elapsed_time > time_limit:
            print("Training stopped after reaching time limit.")
            break

    # Ensure the models directory exists
    model_directory = './models'
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    
    # Generate a timestamped filename and include the directory in the path
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_filename = os.path.join(model_directory, f"ppo_sonic_model_{timestamp}.zip")
    model.save(model_filename)
    print(f"Model saved as {model_filename}")
    env.close()

if __name__ == "__main__":
    main()
