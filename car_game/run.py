
import gymnasium as gym
import os
import argparse
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

import car_game

MODELS_DIR = "models"
LOGS_DIR = "logs"
MODEL_FILENAME = "PPO_CarGame.zip"

def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

def train(args):
    logging.info("Starting training (gym) session...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    model_path = os.path.join(MODELS_DIR, MODEL_FILENAME)
    tensorboard_log_path = os.path.join(LOGS_DIR, "ppo_cargame_tensorboard")

    logging.info(f"Creating vectorized environment with n={args.n_envs} parallel instances.")
    vec_env = make_vec_env("CarGame-v0", n_envs=args.n_envs)

    eval_callback = EvalCallback(vec_env,
                                 best_model_save_path=os.path.join(MODELS_DIR, 'best_model'),
                                 log_path=os.path.join(LOGS_DIR, 'eval_results'),
                                 eval_freq=max(args.eval_freq // args.n_envs, 1),
                                 deterministic=True,
                                 render=False)

    model = PPO("MlpPolicy",
                vec_env,
                verbose=1,
                tensorboard_log=tensorboard_log_path)

    logging.info(f"training for {args.timesteps} timesteps")
    model.learn(total_timesteps=args.timesteps,
                progress_bar=True,
                callback=eval_callback)

    logging.info(f"Saving final model to the location {model_path}")
    model.save(model_path)
    logging.info("Training complete!!!!!!")

def play(args):
    
    model_path = os.path.join(MODELS_DIR, MODEL_FILENAME)

    if not os.path.exists(model_path):
        logging.error(f"Model not found at {model_path}. Please train the model first.")
        return

    logging.info(f"Loading model from {model_path}")
    loaded_model = PPO.load(model_path)

    eval_env = gym.make("CarGame-v0", render_mode="human")

    for ep in range(args.episodes):
        obs, info = eval_env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _states = loaded_model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_reward += reward
            
            done = terminated or truncated

        logging.info(f"Episode(s))) {ep + 1} finished. Final Score: {info['score']}  , Episode Reward: {episode_reward:.2f}")

    eval_env.close()
    logging.info("Playback done!!!")

def main():
    parser = argparse.ArgumentParser(description="Train or Play a PPO agent for tha CarGame")
    parser.add_argument("mode", choices=["train", "play"], help="Mode to run: train or play")
    
    parser.add_argument("--timesteps", type=int, default=50000, help="Total timesteps for training.")
    parser.add_argument("--n_envs", type=int, default=4, help="Number of parallel environments for training.")
    parser.add_argument("--eval_freq", type=int, default=5000, help="Frequency of model evaluation plus saving.")

    parser.add_argument("--episodes", type=int, default=5, help="Number of eps to play.")

    args = parser.parse_args()
    
    setup_logging()

    if args.mode == "train":
        train(args)
    elif args.mode == "play":
        play(args)

if __name__ == "__main__":
    main()