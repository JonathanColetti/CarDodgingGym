from gymnasium.envs.registration import register

register(
     id="CarGame-v0",
     entry_point="car_game.envs:CarGameEnv",
)