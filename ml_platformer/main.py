import os
import time
import math
import csv
import argparse
from datetime import datetime
import pygame as pg

from . import config as C
from .level import Level
from .player import Player, InputState
from .ai_agent import QAgent
from .ui import UI

SAVE_PATH = os.path.join(os.path.dirname(__file__), "qtable.pkl")
LOG_PATH = os.path.join(os.path.dirname(__file__), "completion_times.txt")
EPISODE_LOG_PATH = os.path.join(os.path.dirname(__file__), "episode_log.csv")

def compute_reward(prev_dist, new_dist, prev_x, new_x, reached_exit, fell, dt, idle_weight: float, episode_time: float, reached_timeout: bool, furthest_x_reward: float, cur_input: InputState):
    r = 0.0
    # Penalize elapsed time per second (frame-rate independent)
    r -= C.REWARD_TIME_PENALTY_PER_SEC * dt
    # Reward progress towards exit using Euclidean distance and horizontal movement to the right
    r += C.REWARD_PROGRESS_SCALE * (prev_dist - new_dist)
    dx = new_x - prev_x
    if dx > 0:
        r += C.REWARD_PROGRESS_X_SCALE * dx
    elif dx < 0:
        r += -C.LEFT_MOVE_PENALTY_PER_PX * (-dx)
    # Idle penalty when agent stays grounded and near-zero velocity
    r -= C.IDLE_PENALTY_PER_SEC * idle_weight * dt
    # Reward for pushing furthest x this episode
    r += furthest_x_reward
    # Penalize jumping slightly to bias towards forward motion when not needed
    if cur_input.jump:
        r -= C.JUMP_PENALTY_PER_SEC * dt
    # Terminal rewards
    if reached_exit:
        # Add a large bonus for finishing quickly: bonus/time
        time_bonus = C.REWARD_TIME_BONUS / max(0.5, episode_time)
        r += C.REWARD_REACH_EXIT + time_bonus
    if fell:
        r += C.REWARD_FALL_DEATH
    if reached_timeout:
        r += C.TIMEOUT_PENALTY
    return r

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="ML Platformer - Q-learning Demo")
    g_mode = p.add_mutually_exclusive_group()
    g_mode.add_argument("--ai", dest="ai_control", action="store_true", help="Start with AI control (default)")
    g_mode.add_argument("--human", dest="ai_control", action="store_false", help="Start with human control")
    p.set_defaults(ai_control=True)
    g_train = p.add_mutually_exclusive_group()
    g_train.add_argument("--train", dest="training", action="store_true", help="Enable training (default)")
    g_train.add_argument("--no-train", dest="training", action="store_false", help="Disable Q-updates")
    p.set_defaults(training=True)
    p.add_argument("--episodes", type=int, default=0, help="Run for N episodes then exit (0 = infinite)")
    p.add_argument("--seed", type=int, default=123, help="RNG seed for the agent")
    p.add_argument("--load", action="store_true", help="Load Q-table at start if present")
    p.add_argument("--save-on-exit", action="store_true", help="Save Q-table upon exit")
    p.add_argument("--layout", type=int, default=None, help="Select layout index (0..2)")
    p.add_argument("--theme", type=int, default=None, help="Select theme index (0..N-1)")
    p.add_argument("--headless", action="store_true", help="Run without rendering (hidden window)")
    p.add_argument("--fps", type=int, default=C.FPS, help="Target FPS for the clock")
    p.add_argument("--speedup", type=float, default=1.0, help="Simulation speed multiplier (e.g., 3.0)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    # Headless mode: prefer hidden window to avoid driver issues
    if args.headless:
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    pg.init()
    flags = pg.DOUBLEBUF
    try:
        flags |= pg.SCALED  # nicer scaling on some displays
    except Exception:
        pass
    try:
        if args.headless and hasattr(pg, "HIDDEN"):
            flags |= pg.HIDDEN
        screen = pg.display.set_mode((C.WIDTH, C.HEIGHT), flags, vsync=C.VSYNC)
    except TypeError:
        # Older pygame without vsync keyword
        if args.headless and hasattr(pg, "HIDDEN"):
            flags |= pg.HIDDEN
        screen = pg.display.set_mode((C.WIDTH, C.HEIGHT), flags)
    pg.display.set_caption("ML Platformer - Optimize for Fastest Time to Exit")
    clock = pg.time.Clock()

    level = Level()
    ui = UI()
    player = Player(level.spawn_x, level.spawn_y)

    # Start a fresh log of completion times for this run
    try:
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            pass
    except Exception:
        pass

    agent = QAgent(seed=args.seed)
    training = bool(args.training)
    ai_control = bool(args.ai_control)
    if args.load and os.path.exists(SAVE_PATH):
        try:
            agent.load(SAVE_PATH)
        except Exception:
            pass
    best_time = None  # best episode time (seconds)
    last_reset_reason = None
    episode_idx = 1  # sequential episode counter for logging

    cam_x = 0.0
    t0 = time.time()
    paused = False

    episode_step = 0
    episode_time = 0.0
    last_action = 0
    action_hold = 0
    ai_frame_accum = 0
    # Keep last state for proper Q-learning update
    last_state = None
    furthest_x = 0

    # Fixed-timestep simulation for stable physics
    target_fps = max(1, int(args.fps))
    fixed_dt_base = (1.0 / target_fps) * C.TIME_SCALE
    fixed_dt_fast_default = (1.0 / target_fps) * (C.TIME_SCALE * 2.5)
    fixed_dt = fixed_dt_base
    accumulator = 0.0
    # Track spikes already awarded (across episodes) and pending boost
    # Keyed by (layout_index, spike_idx) so each physical spike awards once per layout
    awarded_spikes: set[tuple[int, int]] = set()
    pending_spike_boost: tuple[int, int] | None = None
    # Ensure episode CSV has header
    _ensure_episode_csv()

    episodes_to_run = max(0, int(args.episodes))
    episodes_completed = 0

    while True:
        frame_dt = clock.tick(target_fps) / 1000.0
        accumulator += frame_dt
        t = time.time() - t0

        # Speedup button: hold space to increase simulation speed
        keys = pg.key.get_pressed()
        # Choose fastest of: CLI speedup, Space hold, or base
        fixed_dt = fixed_dt_base
        if keys[pg.K_SPACE]:
            fixed_dt = max(fixed_dt, fixed_dt_fast_default)
        if args.speedup and args.speedup > 1.0:
            fixed_dt = max(fixed_dt, (1.0 / target_fps) * (C.TIME_SCALE * float(args.speedup)))

        # Process events (quit/toggles/save/load)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                safe_quit(agent)
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    safe_quit(agent)
                if event.key == pg.K_h:
                    ai_control = not ai_control
                if event.key == pg.K_t:
                    training = not training
                if event.key == pg.K_r:
                    reset_episode(player, level)
                # Optional: rotate layout/theme
                if event.key == pg.K_F1:
                    level.reset(rotate_layout=True, rotate_theme=False)
                    reset_episode(player, level)
                    # Clear awarded spikes when layout changes to avoid cross-layout carryover
                    awarded_spikes.clear()
                if event.key == pg.K_F2:
                    level.reset(rotate_layout=False, rotate_theme=True)
                    reset_episode(player, level)
                    episode_step = 0
                if event.key == pg.K_s:
                    agent.save(SAVE_PATH)
                if event.key == pg.K_l:
                    if os.path.exists(SAVE_PATH):
                        agent.load(SAVE_PATH)
                if event.key == pg.K_F12:
                    try:
                        save_screenshot(screen)
                    except Exception:
                        pass

        # Run fixed-step updates to catch up
        ran_updates = 0
        while accumulator >= fixed_dt and ran_updates < 4:  # clamp to avoid spiral of death
            # Determine control input at sim rate (AI/frame gate still applies)
            if ai_control:
                ai_frame_accum += 1
                if ai_frame_accum >= C.AI_UPDATE_EVERY:
                    ai_frame_accum = 0
                    if action_hold <= 0:
                        state = agent.get_state(player, level)
                        last_action = agent.act(state)
                        last_state = state
                        action_hold = C.MIN_ACTION_HOLD_FRAMES
                    else:
                        action_hold -= 1
                inp = agent.to_input(int(last_action))
            else:
                keys = pg.key.get_pressed()
                inp = InputState(
                    left=keys[pg.K_a] or keys[pg.K_LEFT],
                    right=keys[pg.K_d] or keys[pg.K_RIGHT],
                    jump=keys[pg.K_SPACE] or keys[pg.K_w] or keys[pg.K_UP],
                )
                # Human control: allow agent to learn from human actions
                state = agent.get_state(player, level)
                last_action = 2 if inp.right else (1 if inp.left else (3 if inp.jump else 0))
                last_state = state

            # Distance to exit before step
            prev_dist = dist_to_exit(player, level)
            prev_x = player.rect.centerx
            prev_hazard = any(player.rect.colliderect(h) for h in level.hazards)
            # Track previous spike index for reward
            prev_spike_idx = None
            for idx, h in enumerate(level.hazards):
                if player.rect.colliderect(h):
                    prev_spike_idx = idx
                    break

            # Step simulation
            player.update(fixed_dt, level, inp)
            level.update_clouds(fixed_dt)
            episode_time += fixed_dt

            # Check terminal conditions
            reached_exit = player.rect.colliderect(level.exit_trigger)
            # Hazard contact knocks out the player (spikes only, not ground)
            hazard_now = level.intersects_hazard(player.rect)
            died_to_hazard = False
            # Only kill player if hazard is a spike, not ground
            if hazard_now:
                # Check if colliding with any hazard that is not the ground
                for h in level.hazards:
                    if player.rect.colliderect(h):
                        player.alive = False
                        died_to_hazard = True
                        break
            fell = not player.alive
            reached_timeout = episode_time >= C.EPISODE_MAX_TIME_SEC

            # Reward and learning
            new_dist = dist_to_exit(player, level)
            new_x = player.rect.centerx
            idle_weight = 1.0 if (player.on_ground and abs(player.vel.x) < 20) else 0.0
            furthest_bonus = 0.0
            if new_x > furthest_x:
                furthest_bonus = (new_x - furthest_x) * C.REWARD_FURTHEST_X_PER_PX
                furthest_x = new_x
            r = compute_reward(
                prev_dist, new_dist, prev_x, new_x, reached_exit, fell, fixed_dt, idle_weight, episode_time, reached_timeout, furthest_bonus, inp
            )
            # If died to hazard, add extra penalty
            if died_to_hazard:
                r += C.HAZARD_DEATH_PENALTY

            # Detect if player crosses a spike from left to right in this frame.
            # We only award the bonus once per spike (per layout) and only after landing.
            cur_layout = getattr(level, "layout_index", 0)
            if pending_spike_boost is None:
                for idx, h in enumerate(level.hazards):
                    key = (cur_layout, idx)
                    if key in awarded_spikes:
                        continue
                    if prev_x < h.left and new_x >= h.right:
                        pending_spike_boost = key
                        break

            # Only give boost after landing on ground after clearing a spike
            if pending_spike_boost is not None and player.on_ground:
                # Award once and mark as awarded
                if pending_spike_boost not in awarded_spikes:
                    r += 80.0  # reward boost for passing spike and landing
                    awarded_spikes.add(pending_spike_boost)
                pending_spike_boost = None

            # Learn from both AI and human play
            if training and last_state is not None:
                next_state = agent.get_state(player, level)
                done = reached_exit or fell or (episode_time >= C.EPISODE_MAX_TIME_SEC)
                # If human passes a hazard (was not touching, now is), give a positive reward
                if not ai_control and not prev_hazard and hazard_now:
                    agent.reward(10.0, last_state, next_state, last_action, done)
                agent.reward(r, last_state, next_state, last_action, done)

            episode_step += 1
            if reached_exit or fell or (episode_time >= C.EPISODE_MAX_TIME_SEC):
                pending_spike_boost = None
                if reached_exit:
                    if best_time is None or episode_time < best_time:
                        best_time = episode_time
                    last_reset_reason = "exit"
                    # Log AI completion time
                    if ai_control:
                        try:
                            with open(LOG_PATH, "a", encoding="utf-8") as f:
                                f.write(f"{episode_idx},{episode_time:.4f}\n")
                        except Exception:
                            pass
                elif fell:
                    last_reset_reason = "fell"
                    # Set agent reward to 30% of best score to encourage survival
                    percent = 0.3
                    if best_time is not None and best_time > 0:
                        best_score = C.REWARD_TIME_BONUS / best_time
                        agent.total_reward = percent * best_score
                    else:
                        agent.total_reward = 0.0
                else:
                    last_reset_reason = "timeout"

                # Append rich episode CSV row
                try:
                    _append_episode_csv({
                        "episode": episode_idx,
                        "time": f"{episode_time:.4f}",
                        "reward": f"{agent.total_reward:.4f}",
                        "epsilon": f"{agent.epsilon:.4f}",
                        "steps": episode_step,
                        "reason": last_reset_reason,
                    })
                except Exception:
                    pass

                reset_episode(player, level)
                episode_step = 0
                episode_time = 0.0
                last_state = None
                furthest_x = 0
                episode_idx += 1

                # Respect --episodes budget
                if episodes_to_run > 0:
                    episodes_completed += 1
                    if episodes_completed >= episodes_to_run:
                        safe_quit(agent, save_on_exit=args.save_on_exit)

            # Camera follow
            target_cam = max(0, min(player.rect.centerx - C.WIDTH * 0.5, C.LEVEL_WIDTH - C.WIDTH))
            cam_x += (target_cam - cam_x) * C.CAMERA_LERP

            accumulator -= fixed_dt
            ran_updates += 1

        # Render once per frame using latest state
        if not args.headless:
            level.draw_background(screen, cam_x)
            level.draw_platforms(screen, cam_x)
            level.draw_exit(screen, cam_x, t)
            player.draw(screen, cam_x, t)

        # Prepare AI WASD indicator from the last AI input
        ai_wasd = None
        if ai_control:
            last_inp = agent.to_input(int(last_action)) if 'agent' in locals() else InputState()
            ai_wasd = {
                'w': bool(last_inp.jump),
                'a': bool(last_inp.left),
                's': False,
                'd': bool(last_inp.right),
            }

        if not args.headless:
            ui.draw(screen, {
                "training": training,
                "ai_control": ai_control,
                "episodes": agent.episodes,
                "steps": agent.steps,
                "epsilon": agent.epsilon,
                "reward": agent.total_reward,
                "time": episode_time,
                "best_time": best_time,
                "reason": last_reset_reason,
                "ai_wasd": ai_wasd,
            })

            pg.display.flip()

def dist_to_exit(player: Player, level: Level) -> float:
    dx = level.exit_rect.centerx - player.rect.centerx
    dy = level.exit_rect.centery - player.rect.centery
    return math.hypot(dx, dy)

def reset_episode(player: Player, level: Level):
    player.reset(level.spawn_x, level.spawn_y)

def save_screenshot(screen: pg.Surface, out_dir: str | None = None):
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "images")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(out_dir, f"screenshot-{ts}.png")
    pg.image.save(screen, path)
    return path


def _ensure_episode_csv():
    if not os.path.exists(EPISODE_LOG_PATH):
        try:
            with open(EPISODE_LOG_PATH, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["episode", "time", "reward", "epsilon", "steps", "reason"]) 
        except Exception:
            pass


def _append_episode_csv(row: dict):
    try:
        with open(EPISODE_LOG_PATH, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                row.get("episode"), row.get("time"), row.get("reward"),
                row.get("epsilon"), row.get("steps"), row.get("reason")
            ])
    except Exception:
        pass


def safe_quit(agent: QAgent, save_on_exit: bool = True):
    try:
        if save_on_exit:
            agent.save(SAVE_PATH)
    except Exception:
        pass
    pg.quit()
    raise SystemExit

if __name__ == "__main__":
    main()
