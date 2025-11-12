import gymnasium as gym
import os
import argparse
import logging
import json

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback

import car_game

MODELS_DIR = "car_game/models"
LOGS_DIR = "logs"
MODEL_FILENAME = "PPO_CarGame.zip"

def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')



def train(args):
    logging.info("Starting training session...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    model_path = os.path.join(MODELS_DIR, MODEL_FILENAME)
    tensorboard_log_path = os.path.join(LOGS_DIR, "ppo_cargame_tensorboard")

    vec_env = make_vec_env("CarGame-v0", n_envs=args.n_envs)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    eval_env = make_vec_env("CarGame-v0", n_envs=1, seed=123)
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=True)

    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path=os.path.join(MODELS_DIR, 'best_model'),
                                 log_path=os.path.join(LOGS_DIR, 'eval_results'),
                                 eval_freq=max(args.eval_freq // args.n_envs, 1),
                                 deterministic=True,
                                 render=False)

    model_params = {
        'n_steps': args.n_steps,
        'batch_size': args.batch_size,
        'n_epochs': args.n_epochs,
        'gamma': args.gamma,
        'learning_rate': args.learning_rate,
        'clip_range_vf': args.clip_range_vf,
        'ent_coef': args.ent_coef,
        'vf_coef': args.vf_coef,
        'gae_lambda': args.gae_lambda,
        'clip_range': args.clip_range,
        'max_grad_norm': args.max_grad_norm
    }

    logging.info(f"PPO parameters: {model_params}")

    model = PPO("MlpPolicy",
                vec_env,
                verbose=1,
                tensorboard_log=tensorboard_log_path,
                **model_params)

    logging.info(f"Training for {args.timesteps} timesteps")
    model.learn(total_timesteps=args.timesteps,
                progress_bar=True,
                callback=eval_callback)

    logging.info(f"Saving final model to {model_path}")
    model.save(model_path)
    vec_stats_pickle_path = "vecnormalize_stats.pkl"
    vec_env.save(vec_stats_pickle_path)
    
    logging.info("Training complete, exporting to ")
    stats_path = "vecnormalize_stats.json" 
    
    stats = {
        "mean": vec_env.obs_rms.mean.tolist(),
        "var": vec_env.obs_rms.var.tolist(),
        "epsilon": vec_env.epsilon,
        "clip_obs": vec_env.clip_obs
    }

    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=4)


def play(args):
    model_path = os.path.join(MODELS_DIR, MODEL_FILENAME)
    stats_path = os.path.join(MODELS_DIR, "vecnormalize_stats.pkl")

    if not os.path.exists(model_path) or not os.path.exists(stats_path):
        logging.error(f"Model ({model_path}) or stats ({stats_path}) not found. Train it first.")
        return

    logging.info(f"Loading model from {model_path} and stats from {stats_path}")

    eval_env_vec = make_vec_env("CarGame-v0", n_envs=1, env_kwargs={"render_mode": "human"})

    eval_env_vec = VecNormalize.load(stats_path, eval_env_vec)

    eval_env_vec.training = False
    eval_env_vec.norm_reward = False

    loaded_model = PPO.load(model_path, env=eval_env_vec)

    for ep in range(args.episodes):
        obs = eval_env_vec.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _states = loaded_model.predict(obs, deterministic=True)
            
            obs, reward, done, info = eval_env_vec.step(action)
            
            episode_reward += reward[0]

            if done[0]:
                logging.info(f"Episode {ep + 1}: Score={info[0]['score']}, Reward={episode_reward:.2f}")

    eval_env_vec.close()
    logging.info("Playback complete")

def main():
    parser = argparse.ArgumentParser(description="Train or Play a PPO agent for CarGame")
    parser.add_argument("mode", choices=["train", "play"], help="Mode to run: train or play")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n_envs", type=int, default=2048)
    parser.add_argument("--eval_freq", type=int, default=20000)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--n_steps", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning_rate", type=float, default=0.0003)
    parser.add_argument("--clip_range_vf", type=float, default=None)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)

    args = parser.parse_args()
    setup_logging()
    if args.mode == "train":
        train(args)
    elif args.mode == "play":
        play(args)

if __name__ == "__main__":
    main()

