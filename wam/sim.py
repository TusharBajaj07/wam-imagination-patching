"""Closed-loop LIBERO runner (sim image only). Mirrors experiments/libero/eval_libero_single.py run_single_episode()
and _predict_action_chunk(), reusing FastWAM's own LIBERO helpers, with two optional interventions per decision:
  imagine_prompt  imagine the future with a DIFFERENT instruction, then put that future (cache tokens 98-293, chosen
                  layers) into the real run; the current-frame tokens stay the real run's
  act_prompt      run the action steps with a different instruction (the cache stays the real run's)

Official settings reproduced: render 256x256 (LIBERO_ENV_RESOLUTION), wait 30 no-op steps, replan every 10 steps,
max steps 400 (Spatial/Object/Goal) / 700 (LIBERO-10), gripper: x*2-1, sign flip, binarize. Seed 42 (train.yaml) is
used for env.seed() and for the model's noise at every decision, exactly as the official eval does.
"""
from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, "/root/FastWAM")
import wamlib as W  # noqa: E402
from experiments.libero.libero_utils import (  # noqa: E402
    get_libero_dummy_action, get_libero_env, get_libero_image, invert_gripper_action, quat2axisangle)

MAX_STEPS = {"libero_spatial": 400, "libero_object": 400, "libero_goal": 400, "libero_10": 700}
WAIT, REPLAN, RES, SEED = 30, 10, 256, 42
FUT = slice(98, 294)


def suite_task(suite: str, task_id: int):
    """Task + its 50 fixed starting states. Loaded like eval_libero_single.py:733-738 (weights_only=False: the files
    are pickled numpy arrays from the pinned LIBERO repo, which PyTorch >= 2.6 refuses by default)."""
    from pathlib import Path
    from libero.libero import benchmark, get_libero_path
    s = benchmark.get_benchmark_dict()[suite]()
    task = s.get_task(task_id)
    path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    return task, torch.load(path, weights_only=False)


def obs_inputs(obs, model, proc):
    imgs = get_libero_image(obs)                                   # rotated 180 deg to match training data
    image = W.to_model_image(imgs["image"], imgs["wrist_image"], model.device, model.torch_dtype)
    state = np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"])).astype(np.float32)
    return imgs, image, state, W.normalize_state(state, proc)


def to_env_actions(a_norm: torch.Tensor, proc) -> np.ndarray:
    key = proc.shape_meta["action"][0]["key"]
    a = proc.normalizer.normalizers["action"][key].backward(a_norm.float().cpu().unsqueeze(0)).numpy()[0]
    a[..., -1] = a[..., -1] * 2 - 1
    a = invert_gripper_action(a)
    a[..., -1] = np.sign(a[..., -1])
    return a


def plan(model, proc, obs, prompt, mode="idm", imagine_prompt=None, act_prompt=None, swap_layers=range(30),
         alpha=1.0, seed=SEED):
    """One planning decision. Returns env actions [32,7], the normalized chunk, and the stored inputs.
    With imagine_prompt: future tokens 98-293 of the chosen layers become (1-alpha)*own + alpha*other (alpha=1: swap)."""
    imgs, image, state, prop = obs_inputs(obs, model, proc)
    if mode == "first_frame":
        a_norm = model.infer_action(prompt=prompt, input_image=image, action_horizon=W.ACTION_HORIZON, proprio=prop,
                                    num_inference_steps=W.NUM_STEPS, sigma_shift=W.SIGMA_SHIFT, seed=seed,
                                    rand_device="cpu", action_infer_mode="first_frame",
                                    num_video_frames=W.NUM_VIDEO_FRAMES)["action"].float()
    else:
        p = W.prepare(model, image, prop, prompt, seed=seed)
        K, V, amask, vseq = W.prefill(model, p, W.imagine(model, p))
        if imagine_prompt is not None:
            pb = W.prepare(model, image, prop, imagine_prompt, seed=seed)
            Kb, Vb, _, _ = W.prefill(model, pb, W.imagine(model, pb))
            if alpha == 1.0:
                K, V = W.edit_slice(K, swap_layers, FUT, Kb), W.edit_slice(V, swap_layers, FUT, Vb)
            else:
                K = W.edit_slice(K, swap_layers, FUT, lambda i: torch.lerp(K[i][:, FUT], Kb[i][:, FUT], alpha))
                V = W.edit_slice(V, swap_layers, FUT, lambda i: torch.lerp(V[i][:, FUT], Vb[i][:, FUT], alpha))
        pa = p if act_prompt is None else W.prepare(model, image, prop, act_prompt, seed=seed)
        a_norm = W.act(model, pa, K, V, amask)
    return to_env_actions(a_norm, proc), a_norm.cpu(), {"main": imgs["image"], "wrist": imgs["wrist_image"],
                                                        "state": state}


def run_episode(model, proc, suite, task_id, init_idx, mode="idm", imagine_task=None, act_task=None,
                check_task=None, keep_states=False, swap_layers=range(30), alpha=1.0, seed=SEED, video_every=0, stop_on_check=False):
    """Run task `task_id` from starting state `init_idx`. Optionally imagine / act with another task's instruction,
    and check another task's goal (`check_task`) on the simulator state at every decision. `seed` drives env.seed()
    and the model noise (official eval: 42). stop_on_check: end the episode at the first decision where the other
    task's goal is seen (saves GPU; in 272 earlier episodes A never succeeded after B's goal was reached). video_every=k>0 also puts the main camera every k steps (128 px) in result["_frames"]."""
    task, inits = suite_task(suite, task_id)
    env, desc = get_libero_env(task, RES, seed)
    prompt = W.PROMPT.format(task=desc)
    other = {}
    for k, tid in (("imagine", imagine_task), ("act", act_task), ("check", check_task)):
        if tid is not None:
            other[k] = suite_task(suite, tid)[0]
    imagine_prompt = W.PROMPT.format(task=other["imagine"].language) if "imagine" in other else None
    act_prompt = W.PROMPT.format(task=other["act"].language) if "act" in other else None
    env_check = get_libero_env(other["check"], RES, seed)[0] if "check" in other else None
    if env_check is not None:
        env_check.reset()

    env.reset()
    obs = env.set_init_state(inits[init_idx])
    pending, done, t, decisions, plan_s, states, frames = [], False, 0, 0, [], [], []
    check_first, check_error = None, None

    def check_other():
        nonlocal check_first, check_error
        if env_check is None or check_first is not None or check_error is not None:
            return
        try:
            env_check.set_init_state(env.get_sim_state())
            if env_check.check_success():
                check_first = t - WAIT
        except Exception as exc:  # e.g. the two tasks' simulator states differ in size
            check_error = f"{type(exc).__name__}: {exc}"

    while t < MAX_STEPS[suite] + WAIT:
        if t < WAIT:
            obs, _, done, _ = env.step(get_libero_dummy_action())
            t += 1
            continue
        if not pending:
            check_other()
            if stop_on_check and check_first is not None:
                break
            t0 = W.cuda_sync_time()
            acts, a_norm, stored = plan(model, proc, obs, prompt, mode, imagine_prompt, act_prompt, swap_layers,
                                        alpha, seed)
            plan_s.append(W.cuda_sync_time() - t0)
            pending = acts[:REPLAN].tolist()
            decisions += 1
            if keep_states:
                stored.update(t=t, a_norm=a_norm.numpy())
                states.append(stored)
        obs, _, done, _ = env.step(pending.pop(0))
        t += 1
        if video_every and (t - WAIT) % video_every == 0:
            from PIL import Image
            frames.append(np.asarray(Image.fromarray(get_libero_image(obs)["image"]).resize((128, 128))))
        if done:
            break
    check_other()
    env.close()
    if env_check is not None:
        env_check.close()
    r = {"suite": suite, "task_id": task_id, "task": desc, "init": init_idx, "mode": mode,
            "imagine_task": imagine_task, "act_task": act_task, "check_task": check_task,
            "success": bool(done), "check_success_step": check_first, "check_error": check_error,
            "steps": t - WAIT, "decisions": decisions, "plan_s_mean": float(np.mean(plan_s)) if plan_s else None,
            "swap_layers": [i + 1 for i in swap_layers] if imagine_task is not None else None,
            "alpha": alpha if imagine_task is not None else None, "seed": seed, "stopped_on_check": stop_on_check}
    if video_every:  # popped by the caller, never written to the jsonl
        r["_frames"] = np.stack(frames) if frames else np.zeros((0, 128, 128, 3), np.uint8)
    return r, states
