import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
import onnx
import onnxruntime as ort
import os
import logging

# needed to register environment
import car_game

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
MODELS_DIR = "models"
MODEL_FILENAME = "PPO_CarGame.zip"
ONNX_FILENAME = "ppo_cargame.onnx"

model_path = os.path.join(MODELS_DIR, MODEL_FILENAME)
onnx_path = ONNX_FILENAME

if not os.path.exists(model_path):
    logging.error(f"Model not found at {model_path}", " run train.py")
    exit()


class OnnxablePolicy(nn.Module):
    def __init__(self, policy: ActorCriticPolicy):
        super().__init__()
        self.features_extractor = policy.features_extractor
        self.mlp_extractor = policy.mlp_extractor
        self.action_net = policy.action_net

    def forward(self, observation: torch.Tensor):
        """
        Defines the forward pass for deterministic action prediction.
        Args:

        Returns:
        """
        features = self.features_extractor(observation)
        latent_pi, _ = self.mlp_extractor(features)
        action_logits = self.action_net(latent_pi)
        return torch.argmax(action_logits, dim=1)



logging.info("Loading the trained PPO model...")
sb3_model = PPO.load(model_path, device='cpu')

onnxable_model = OnnxablePolicy(sb3_model.policy)
onnxable_model.eval()

logging.info(f"Exporting model to {onnx_path}")
env = gym.make("CarGame-v0")
sample_obs, _ = env.reset()

sample_obs = sample_obs.astype(np.float32)
obs_tensor = torch.tensor(sample_obs).unsqueeze(0)

torch.onnx.export(
    onnxable_model,
    obs_tensor,
    onnx_path,
    opset_version=18,
    input_names=["obs"],
    output_names=["action"],
    dynamic_axes={
        "obs": {0: "batch_size"},
        "action": {0: "batch_size"},
    },
    external_data=False
)
logging.info("Export complete. - Verifying onnx model...")


onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
logging.info("ONNX model structure is valid.")

# compare model outputs to make sure we all good
ort_session = ort.InferenceSession(onnx_path)
test_obs, _ = env.reset()
test_obs_np = test_obs.astype(np.float32)

sb3_action, _ = sb3_model.predict(test_obs_np, deterministic=True)
ort_inputs = {"obs": test_obs_np.reshape(1, -1)}
ort_outputs = ort_session.run(None, ort_inputs)
onnx_action = ort_outputs[0][0]

logging.info(f"\nSample Observation: {test_obs_np}")
logging.info(f"SB3 Model Action:   {sb3_action}")
logging.info(f"ONNX Model Action:  {onnx_action}")

np.testing.assert_allclose(sb3_action, onnx_action)


env.close()
