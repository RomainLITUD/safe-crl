"""Rollout helpers for visualization scripts."""

from __future__ import annotations

import jax
from jax import numpy as jp

def resolve_visual_env_id(env_id: str, headless: bool | None = None) -> str:
    """Maps an environment ID to its headless or renderable variant."""
    if headless is None:
        return env_id
    is_headless = env_id.endswith("_headless")
    if headless and not is_headless:
        return f"{env_id}_headless"
    if not headless and is_headless:
        return env_id[: -len("_headless")]
    return env_id


def force_square_html(html_string: str, size: int) -> str:
    """Injects CSS that makes Brax HTML render in a fixed square viewport."""
    square_css = f"""
<style>
html, body {{
  margin: 0;
  padding: 0;
  width: {size}px;
  height: {size}px;
  overflow: hidden;
  background: #ffffff;
}}
body > div:first-child {{
  width: {size}px !important;
  height: {size}px !important;
}}
canvas {{
  width: {size}px !important;
  height: {size}px !important;
  display: block;
}}
</style>
"""
    if "</head>" in html_string:
        return html_string.replace("</head>", f"{square_css}</head>", 1)
    return square_css + html_string




def sync_mocap_pipeline_state_for_render(env, pipeline_state):
    """Mirrors mocap body positions into link positions for Brax HTML renders.

    Some environments move task bodies through ``mocap_pos``. The Brax HTML
    renderer reads link transforms from ``x`` and can otherwise display those
    task bodies at their XML default positions.
    """
    mocap_pos = getattr(pipeline_state, "mocap_pos", None)
    if mocap_pos is None:
        return pipeline_state
    link_names = list(getattr(getattr(env, "sys", None), "link_names", ()))
    if not link_names:
        return pipeline_state

    pos = pipeline_state.x.pos

    def set_if_present(body_name: str, mocap_id):
        nonlocal pos
        if mocap_id is None:
            return
        try:
            mocap_index = int(mocap_id)
        except (TypeError, ValueError):
            return
        if mocap_index < 0 or body_name not in link_names:
            return
        pos = pos.at[link_names.index(body_name)].set(mocap_pos[mocap_index])

    set_if_present("goal1", getattr(env, "_target_mocap_id", None))
    set_if_present("goal1", getattr(env, "_goal_mocap_id", None))
    for index, mocap_id in enumerate(getattr(env, "_hazard_mocap_ids", ()), start=1):
        set_if_present(f"hazard{index}", mocap_id)

    return pipeline_state.replace(x=pipeline_state.x.replace(pos=pos))

def _step_and_maybe_reset(env, state, action, auto_reset_on_done: bool):
    """Steps once, optionally replacing terminal states with a fresh reset."""
    next_state = env.step(state, action)
    if not auto_reset_on_done:
        return next_state
    done = next_state.done.astype(bool)
    return jax.lax.cond(
        done,
        lambda terminal_state: env.reset(terminal_state.info["rng"]),
        lambda terminal_state: terminal_state,
        next_state,
    )


def _action_for_mode(env, action, action_mode: str, rng: jax.Array):
    """Returns the action for one rollout step."""
    if action_mode == "zero":
        return action
    if action_mode == "random":
        return jax.random.uniform(rng, (env.action_size,), minval=-1.0, maxval=1.0)
    raise ValueError(f"Unknown action_mode={action_mode!r}; expected 'zero' or 'random'.")


def selected_frame_steps(rollout_steps: int, frame_stride: int, max_render_frames: int) -> list[int]:
    """Returns rollout step numbers to render, including 0 and rollout_steps."""
    if rollout_steps < 1:
        raise ValueError("rollout_steps must be at least 1.")
    if frame_stride < 1:
        raise ValueError("frame_stride must be at least 1.")
    if max_render_frames == 1:
        raise ValueError("max_render_frames must be 0 or at least 2.")

    steps = [0]
    steps.extend(range(frame_stride, rollout_steps + 1, frame_stride))
    if steps[-1] != rollout_steps:
        steps.append(rollout_steps)

    if max_render_frames > 0 and len(steps) > max_render_frames:
        last = len(steps) - 1
        indexes = sorted(
            {
                round(i * last / (max_render_frames - 1))
                for i in range(max_render_frames)
            }
        )
        steps = [steps[i] for i in indexes]
    return steps


def python_rollout_states(
    env,
    state,
    action,
    frame_steps: list[int],
    auto_reset_on_done: bool = True,
    action_mode: str = "zero",
    rng: jax.Array | None = None,
):
    """Rolls out with a Python loop and returns selected states."""
    if rng is None:
        rng = jax.random.PRNGKey(0)
    selected = {step: idx for idx, step in enumerate(frame_steps)}
    states = [None] * len(frame_steps)
    states[selected[0]] = state
    for step in range(1, frame_steps[-1] + 1):
        rng, action_rng = jax.random.split(rng)
        step_action = _action_for_mode(env, action, action_mode, action_rng)
        state = _step_and_maybe_reset(env, state, step_action, auto_reset_on_done)
        if step in selected:
            states[selected[step]] = state
    return states


def jitted_rollout_states(
    env,
    state,
    action,
    frame_steps: list[int],
    auto_reset_on_done: bool = True,
    action_mode: str = "zero",
    rng: jax.Array | None = None,
):
    """Rolls out with a jitted scan/loop and returns selected states."""
    if rng is None:
        rng = jax.random.PRNGKey(0)
    durations = jp.array(
        [end - start for start, end in zip(frame_steps[:-1], frame_steps[1:])],
        dtype=jp.int32,
    )

    def rollout(state, rng):
        def advance(carry, duration):
            current_state, current_rng = carry

            def one_step(_, loop_carry):
                loop_state, loop_rng = loop_carry
                loop_rng, action_rng = jax.random.split(loop_rng)
                step_action = _action_for_mode(env, action, action_mode, action_rng)
                next_state = _step_and_maybe_reset(
                    env,
                    loop_state,
                    step_action,
                    auto_reset_on_done,
                )
                return next_state, loop_rng

            next_state, next_rng = jax.lax.fori_loop(
                0,
                duration,
                one_step,
                (current_state, current_rng),
            )
            return (next_state, next_rng), next_state

        return jax.lax.scan(advance, (state, rng), durations)[1]

    stacked_states = jax.jit(rollout)(state, rng)
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        stacked_states,
    )
    states = [state]
    states.extend(
        jax.tree_util.tree_map(lambda x, i=i: x[i], stacked_states)
        for i in range(len(frame_steps) - 1)
    )
    return states
