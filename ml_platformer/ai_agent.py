import math
import pickle
from collections import defaultdict
import numpy as np
from . import config as C
from .player import InputState

class QAgent:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.q = defaultdict(lambda: np.zeros(len(C.ACTIONS), dtype=np.float32))
        self.alpha = 0.2
        self.gamma = 0.98
        self.epsilon = 0.25
        self.min_epsilon = 0.02
        self.decay = 0.9985  # per step (faster decay for quicker exploitation)
        self.last_state = None
        self.last_action = None
        self.total_reward = 0.0
        self.steps = 0
        self.episodes = 0

    def get_state(self, player, level):
        # Relative position to exit
        dx = (level.exit_rect.centerx - player.rect.centerx)
        dy = (level.exit_rect.centery - player.rect.centery)

        # Discretize
        def bin_val(v, size, min_b, max_b):
            b = int(math.floor(v / size))
            return max(min_b, min(max_b, b))

        sdx = bin_val(dx, 64, -30, 30)
        sdy = bin_val(dy, 48, -20, 20)
        vx = int(math.copysign(1, player.vel.x)) if abs(player.vel.x) > 40 else 0
        vy = -1 if player.vel.y < -50 else (1 if player.vel.y > 50 else 0)
        on_g = 1 if player.on_ground else 0

        # Nearby ledge hint: is there a platform under player within small drop?
        under = 0
        feet = player.rect.move(0, 8)
        for p in level.platforms:
            if feet.centerx >= p.left and feet.centerx <= p.right:
                if 0 <= p.top - feet.bottom <= 64:
                    under = 1
                    break

        return (sdx, sdy, vx, vy, on_g, under)

    def act(self, state):
        if self.rng.random() < self.epsilon:
            a = self.rng.integers(0, len(C.ACTIONS))
        else:
            qvals = self.q[state]
            a = int(np.argmax(qvals))
        return int(a)

    def reward(self, r, state, next_state, action, done):
        # Q-learning update
        qsa = self.q[state][action]
        max_next = 0.0 if done else float(np.max(self.q[next_state]))
        self.q[state][action] = qsa + self.alpha * (r - qsa + self.gamma * max_next)

        # Epsilon decay per step
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)
        self.total_reward += r
        self.steps += 1
        if done:
            self.episodes += 1

    def to_input(self, action: int) -> InputState:
        a = C.ACTIONS[action]
        return InputState(
            left=("left" in a),
            right=("right" in a),
            jump=("jump" in a) or (a == "jump")
        )

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(dict(self.q), f)

    def load(self, path: str):
        with open(path, "rb") as f:
            d = pickle.load(f)
            self.q = defaultdict(lambda: np.zeros(len(C.ACTIONS), dtype=np.float32))
            for k, v in d.items():
                self.q[k] = v
