import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

REWARD_SUCCESSFUL_DODGE = 15.0
REWARD_ALIVE_PER_STEP = 0.1
PENALTY_LANE_CHANGE = 0.0
PENALTY_CRASH = -100.0


class CarGameEnv(gym.Env):
    """
    A 2D car driving game environment for Gymnasium.

    The agent controls a car and avoids colliding with static vehicle.
    goal is to pass/dodge as many cars

    0 = move left, 1 = do not move, 2 = move right
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()

        pygame.init()
        try:
            self.score_font = pygame.font.Font("assets/fonts/joystix monospace.otf", 30)
        except FileNotFoundError:
            self.score_font = pygame.font.Font(None, 40)

        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 800, 660
        self.GRASS_COLOR = (60, 220, 0)
        self.DARK_ROAD_COLOR = (50, 50, 50)
        self.YELLOW_LINE_COLOR = (255, 240, 60)
        self.WHITE_LINE_COLOR = (255, 255, 255)

        self.road_w = int(self.SCREEN_WIDTH / 1.6)
        self.roadmark_w = int(self.SCREEN_WIDTH / 80)
        self.right_lane = self.SCREEN_WIDTH / 2 + self.road_w / 4
        self.left_lane = self.SCREEN_WIDTH / 2 - self.road_w / 4
        self.initial_speed = 3

        self.render_mode = render_mode

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -0.5]), high=np.array([1, 1, 1.5]), dtype=np.float32
        )

        original_car = pygame.image.load("assets/cars/car.png")
        self.car = pygame.transform.scale(
            original_car,
            (int(original_car.get_width() * 0.8), int(original_car.get_height() * 0.8)),
        )
        original_car2 = pygame.image.load("assets/cars/otherCar.png")
        self.car2 = pygame.transform.scale(
            original_car2,
            (
                int(original_car2.get_width() * 0.8),
                int(original_car2.get_height() * 0.8),
            ),
        )

        self.window = None
        self.clock = None

    def _get_obs(self):
        # normalize cars and lanes to zero or one
        player_lane_obs = 0 if self.car_loc.centerx == self.left_lane else 1
        enemy_lane_obs = 0 if self.car2_loc.centerx == self.left_lane else 1
        enemy_y_norm = self.car2_loc.center[1] / self.SCREEN_HEIGHT
        return np.array(
            [player_lane_obs, enemy_lane_obs, enemy_y_norm], dtype=np.float32
        )

    def _get_info(self):
        return {"score": self.score, "level": self.level}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.speed = self.initial_speed
        self.score = 0
        self.level = 0
        self.line_offset = 0

        if self.np_random.integers(0, 2) == 0:
            self.car_loc = self.car.get_rect(
                center=(self.left_lane, self.SCREEN_HEIGHT * 0.85)
            )
        else:
            self.car_loc = self.car.get_rect(
                center=(self.right_lane, self.SCREEN_HEIGHT * 0.85)
            )

        if self.np_random.integers(0, 2) == 0:
            self.car2_loc = self.car2.get_rect(
                center=(self.left_lane, -self.car2.get_height())
            )
        else:
            self.car2_loc = self.car2.get_rect(
                center=(self.right_lane, -self.car2.get_height())
            )

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()

    def step(self, action):
        reward = 0
        # if left and in right lane
        if action == 0 and self.car_loc.centerx == self.right_lane:
            self.car_loc.centerx = self.left_lane
            reward += PENALTY_LANE_CHANGE
        
        # if right and in left lane
        elif action == 2 and self.car_loc.centerx == self.left_lane:
            self.car_loc.centerx = self.right_lane
            reward += PENALTY_LANE_CHANGE

        # move the opponent car
        self.car2_loc.y += self.speed

        # increment speed and level
        if self.score > 0 and self.score % 5 == 0 and self.level < self.score:
            self.speed += 0.5
            self.level += 1

        if self.car2_loc.top > self.SCREEN_HEIGHT:
            reward += REWARD_SUCCESSFUL_DODGE
            # increment score ifff successful dodge
            self.score += 1
            if self.np_random.integers(0, 2) == 0:
                self.car2_loc.center = (self.right_lane, -self.car2.get_height())
            else:
                self.car2_loc.center = (self.left_lane, -self.car2.get_height())

        # collision?
        terminated = self.car_loc.colliderect(self.car2_loc)
        if terminated:
            reward += PENALTY_CRASH
        else:
            reward += REWARD_ALIVE_PER_STEP

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        return None

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
            )
            pygame.display.set_caption("Car Game Environment")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        canvas.fill(self.GRASS_COLOR)
        pygame.draw.rect(
            canvas,
            self.DARK_ROAD_COLOR,
            (
                self.SCREEN_WIDTH / 2 - self.road_w / 2,
                0,
                self.road_w,
                self.SCREEN_HEIGHT,
            ),
        )
        pygame.draw.rect(
            canvas,
            self.WHITE_LINE_COLOR,
            (
                self.SCREEN_WIDTH / 2 - self.road_w / 2 + self.roadmark_w * 2,
                0,
                self.roadmark_w,
                self.SCREEN_HEIGHT,
            ),
        )
        pygame.draw.rect(
            canvas,
            self.WHITE_LINE_COLOR,
            (
                self.SCREEN_WIDTH / 2 + self.road_w / 2 - self.roadmark_w * 3,
                0,
                self.roadmark_w,
                self.SCREEN_HEIGHT,
            ),
        )

        self.line_offset = (self.line_offset + self.speed) % (self.SCREEN_HEIGHT / 10)
        for y in range(
            -int(self.SCREEN_HEIGHT / 10),
            self.SCREEN_HEIGHT,
            int(self.SCREEN_HEIGHT / 10),
        ):
            pygame.draw.rect(
                canvas,
                self.YELLOW_LINE_COLOR,
                (
                    self.SCREEN_WIDTH / 2 - self.roadmark_w / 2,
                    y + self.line_offset,
                    self.roadmark_w,
                    self.SCREEN_HEIGHT / 20,
                ),
            )

        canvas.blit(self.car, self.car_loc)
        canvas.blit(self.car2, self.car2_loc)

        score_text = self.score_font.render(
            f"Score: {self.score}", True, (255, 255, 255)
        )
        canvas.blit(score_text, (10, 10))

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
        pygame.quit()
