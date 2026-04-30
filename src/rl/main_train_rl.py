# src/rl/main_train_rl.py
from __future__ import annotations

from pathlib import Path
import argparse
import time
import csv
import json
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yaml
import torch

from src.rl.replay_buffer import ReplayBuffer
from src.rl.dqn_agent import DQNAgent, DQNConfig
from src.rl.env_pick_rl import PickRLRunner


class _DummyPolicyAgent:
    """Política determinista barata: prioriza urgent si puede."""
    def act(self, state, greedy: bool = False) -> int:
        # state: [q_u, q_n, ...]
        q_u = state[0]
        q_n = state[1]
        if q_u > 0 and q_n > 0:
            return 0
        if q_u > 0:
            return 0
        return 1


def _weighted_choice(items: List[dict], probs: List[float], rng: random.Random) -> dict:
    r = rng.random()
    acc = 0.0
    for it, p in zip(items, probs):
        acc += float(p)
        if r <= acc:
            return it
    return items[-1]


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _make_resources(base_resources: Dict, workers: Tuple[int, int, int]) -> Dict:
    r = dict(base_resources)
    # Estos nombres deben coincidir con tu sim_multistage.yaml
    r["picking_workers"] = int(workers[0])
    r["packing_workers"] = int(workers[1])
    r["dispatch_workers"] = int(workers[2])
    return r


def _estimate_dec(
    sim_cfg: dict,
    base_resources: dict,
    service_cfg: dict,
    reward_cfg: Optional[dict],
    orders_ep: pd.DataFrame,
    workers: Tuple[int, int, int],
    seed: int,
) -> int:
    """Ejecuta un episodio greedy con agente dummy para estimar decision_points (dec)."""
    r_cfg = _make_resources(base_resources, workers)
    runner = PickRLRunner(
        sim_cfg=sim_cfg,
        resources_cfg=r_cfg,
        service_cfg=service_cfg,
        seed=seed,
        reward_cfg=reward_cfg,
    )
    dummy_buf = ReplayBuffer(capacity=1)
    m = runner.run_episode(
        orders=orders_ep,
        agent=_DummyPolicyAgent(),
        buffer=dummy_buf,
        episode_seed=seed,
        greedy=True,
    )
    return int(m.get("decision_points", 0) or 0)


def _auto_pick_eval_windows(
    rl_cfg: dict,
    orders: pd.DataFrame,
    episode_orders: int,
    sim_cfg: dict,
    base_resources: dict,
    service_cfg: dict,
    cache_path: Path,
) -> dict:
    """
    Busca (una vez) ventanas fijas para evaluación por buckets de dificultad (dec) y las cachea.
    """
    cached = _load_json(cache_path)
    if cached and cached.get("version") == 1 and cached.get("episode_orders") == episode_orders:
        return cached

    eval_cfg = rl_cfg.get("eval_fixed", {})
    search_cfg = eval_cfg.get("search", {})
    rng = random.Random(int(search_cfg.get("random_seed", 123)))
    max_trials = int(search_cfg.get("max_trials_per_bucket", 60))

    max_start = max(0, len(orders) - episode_orders - 1)
    picked = {"version": 1, "episode_orders": episode_orders, "picked": {}}

    for b in eval_cfg.get("buckets", []):
        name = b["name"]
        dec_min = int(b["dec_min"])
        dec_max = int(b["dec_max"])
        workers = tuple(int(x) for x in b["workers"])

        best = None
        best_dist = 10**18
        found = None

        for t in range(max_trials):
            start_idx = rng.randint(0, max_start) if max_start > 0 else 0
            orders_ep = orders.iloc[start_idx:start_idx + episode_orders].copy()

            dec = _estimate_dec(
                sim_cfg=sim_cfg,
                base_resources=base_resources,
                service_cfg=service_cfg,
                reward_cfg=rl_cfg.get("reward"),
                orders_ep=orders_ep,
                workers=workers,
                seed=10_000 + t,
            )

            if dec_min <= dec <= dec_max:
                found = {"start_idx": start_idx, "workers": list(workers), "dec": dec}
                break

            if dec < dec_min:
                dist = dec_min - dec
            elif dec > dec_max:
                dist = dec - dec_max
            else:
                dist = 0
            if dist < best_dist:
                best_dist = dist
                best = {"start_idx": start_idx, "workers": list(workers), "dec": dec}

        picked["picked"][name] = found if found is not None else best

    _save_json(cache_path, picked)
    return picked


def _eval_fixed(
    agent: DQNAgent,
    rl_cfg: dict,
    orders: pd.DataFrame,
    episode_orders: int,
    sim_cfg: dict,
    base_resources: dict,
    service_cfg: dict,
    cache: dict,
) -> List[dict]:
    eval_cfg = rl_cfg.get("eval_fixed", {})
    n_seeds = int(eval_cfg.get("seeds", 3))
    out = []

    for bucket_name, info in cache.get("picked", {}).items():
        start_idx = int(info["start_idx"])
        workers = tuple(int(x) for x in info["workers"])
        dec_ref = int(info.get("dec", -1))

        r_cfg = _make_resources(base_resources, workers)
        runner = PickRLRunner(
            sim_cfg=sim_cfg,
            resources_cfg=r_cfg,
            service_cfg=service_cfg,
            seed=123,
            reward_cfg=rl_cfg.get("reward"),
        )

        sla_list, u_list, n_list = [], [], []
        for s in range(n_seeds):
            orders_ep = orders.iloc[start_idx:start_idx + episode_orders].copy()
            buf_dummy = ReplayBuffer(capacity=1)
            m = runner.run_episode(
                orders=orders_ep,
                agent=agent,
                buffer=buf_dummy,
                episode_seed=1_000_000 + s,
                greedy=True,
            )
            sla_list.append(float(m["sla_rate"]))
            u_list.append(float(m["sla_urgent"]))
            n_list.append(float(m["sla_normal"]))

        out.append({
            "bucket": bucket_name,
            "workers": f"{workers[0]}-{workers[1]}-{workers[2]}",
            "dec_ref": dec_ref,
            "sla": float(np.mean(sla_list)),
            "sla_u": float(np.mean(u_list)),
            "sla_n": float(np.mean(n_list)),
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DQN agent")
    parser.add_argument("--config", default="configs/rl.yaml", help="RL config file (relative to repo root)")
    parser.add_argument("--run-name", default="", help="Tag for output files; empty = legacy names")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]

    sim_cfg_path = root / "configs" / "sim_multistage.yaml"
    rl_cfg_path = root / args.config

    # Ajusta estas rutas si en tu proyecto se llaman distinto
    orders_path = root / "data" / "orders_base.csv"
    out_dir = root / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name.strip()

    print("🚀 Entrenamiento DQN — Global (mezcla escenarios + eval fijo)")
    if run_name:
        print(f"   run-name : {run_name}  |  config: {rl_cfg_path.name}")
    print("🔁 Comenzando entrenamiento...\n")

    with open(sim_cfg_path, "r", encoding="utf-8") as f:
        sim_cfg_full = yaml.safe_load(f)
    with open(rl_cfg_path, "r", encoding="utf-8") as f:
        rl_cfg = yaml.safe_load(f)

    simulation = sim_cfg_full["simulation"]
    base_resources = sim_cfg_full["resources"]
    service_time = sim_cfg_full["service_time"]

    base_seed = int(simulation.get("random_seed", 123))
    rng_np = np.random.default_rng(base_seed)
    rng_py = random.Random(2026)

    orders = pd.read_csv(orders_path, parse_dates=["arrival_time"])
    orders = orders.sort_values("arrival_time").reset_index(drop=True)

    # training params
    episode_orders = int(rl_cfg["training"]["episode_orders"])
    episodes = int(rl_cfg["training"]["episodes"])
    updates_per_episode = int(rl_cfg["training"].get("updates_per_episode", 500))
    train_start_size = int(rl_cfg["training"]["train_start_size"])
    train_every_steps = int(rl_cfg["training"]["train_every_steps"])
    target_update_steps = int(rl_cfg["training"]["target_update_steps"])
    ckpt_every = int(rl_cfg["training"].get("ckpt_every", 10))

    # agent
    agent_cfg = DQNConfig(
        input_dim=int(rl_cfg["network"].get("input_dim", 8)),
        hidden_dim=int(rl_cfg["network"]["hidden_dim"]),
        lr=float(rl_cfg["training"]["lr"]),
        gamma=float(rl_cfg["training"]["gamma"]),
        batch_size=int(rl_cfg["training"]["batch_size"]),
        target_update_steps=target_update_steps,
        train_start_size=train_start_size,
        train_every_steps=train_every_steps,
        eps_start=float(rl_cfg["epsilon"]["start"]),
        eps_end=float(rl_cfg["epsilon"]["end"]),
        eps_decay_steps=int(rl_cfg["epsilon"]["decay_steps"]),
    )
    agent = DQNAgent(agent_cfg, seed=base_seed)

    # replay buffer (tu implementación)
    cap = int(rl_cfg["buffer"]["capacity"])
    buffer = ReplayBuffer(capacity=cap)

    # scenario mix
    mix = rl_cfg.get("scenario_mix", {})
    scenarios = mix.get("scenarios", []) if bool(mix.get("enabled", True)) else []
    if not scenarios:
        scenarios = [{"name": "default", "prob": 1.0, "workers": [1, 1, 1], "dec_cap": int(rl_cfg.get("curriculum", {}).get("dec_cap", 3000))}]
    probs = [float(s["prob"]) for s in scenarios]
    ssum = sum(probs) if sum(probs) > 0 else 1.0
    probs = [p / ssum for p in probs]

    # fixed eval windows cache
    eval_cfg = rl_cfg.get("eval_fixed", {})
    eval_enabled = bool(eval_cfg.get("enabled", True))
    eval_every = int(eval_cfg.get("every_episodes", 10))
    cache_rel = str(eval_cfg.get("cache_file", "data/eval_scenarios.json"))
    cache_path = (root / cache_rel).resolve()
    eval_cache = None
    if eval_enabled and bool(eval_cfg.get("auto_pick", True)):
        eval_cache = _auto_pick_eval_windows(
            rl_cfg=rl_cfg,
            orders=orders,
            episode_orders=episode_orders,
            sim_cfg=simulation,
            base_resources=base_resources,
            service_cfg=service_time,
            cache_path=cache_path,
        )

    # history csv
    hist_path = out_dir / (f"{run_name}_train_history.csv" if run_name else "rl_train_history.csv")
    with open(hist_path, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow([
            "episode","scenario","workers","start_idx","dec","dec_cap",
            "sla_rate","sla_urgent","sla_normal","epsilon","loss_mean",
            "buffer_size","updates_done","grad_step_total","p_urgent_decisions",
            "forced_urgent","forced_normal"
        ])

    n_orders = len(orders)
    grad_step_total = 0
    last_slas: List[float] = []
    ma_window = 5

    for ep in range(1, episodes + 1):
        t0 = time.time()
        ep_seed = int(rng_np.integers(0, 1_000_000_000))

        scen = _weighted_choice(scenarios, probs, rng_py)
        workers = tuple(int(x) for x in scen["workers"])
        dec_cap = int(scen.get("dec_cap", int(rl_cfg.get("curriculum", {}).get("dec_cap", 3000))))

        # sample window
        if n_orders <= episode_orders:
            start_idx = 0
            orders_ep = orders
        else:
            start_idx = int(rng_np.integers(0, n_orders - episode_orders))
            orders_ep = orders.iloc[start_idx:start_idx + episode_orders].copy()

        # runner per scenario
        resources_ep = _make_resources(base_resources, workers)
        runner = PickRLRunner(
            sim_cfg=simulation,
            resources_cfg=resources_ep,
            service_cfg=service_time,
            seed=base_seed,
            reward_cfg=rl_cfg.get("reward", None),
        )

        metrics = runner.run_episode(
            orders=orders_ep,
            agent=agent,
            buffer=buffer,
            episode_seed=ep_seed,
            greedy=False,
        )

        dec = int(metrics.get("decision_points", 0) or 0)
        sla = float(metrics["sla_rate"])
        sla_u = float(metrics["sla_urgent"])
        sla_n = float(metrics["sla_normal"])
        pct_u = float(metrics.get("p_urgent_decisions", 0.0))
        forced_u = int(metrics.get("forced_urgent", 0) or 0)
        forced_n = int(metrics.get("forced_normal", 0) or 0)

        # curriculum skip-train (solo afecta a los updates; el buffer ya está lleno con transiciones del runner)
        curriculum_mode = str(rl_cfg.get("curriculum", {}).get("mode", "skip_train"))
        do_train = not (curriculum_mode == "skip_train" and dec > dec_cap)

        updates_done = 0
        losses: List[float] = []
        if do_train and len(buffer) >= train_start_size:
            for _ in range(updates_per_episode):
                if grad_step_total % train_every_steps == 0:
                    batch = buffer.sample(agent.cfg.batch_size, agent.rng)
                    loss = agent.train_step(batch, grad_step_total)
                    if loss is not None:
                        losses.append(float(loss))
                    updates_done += 1
                grad_step_total += 1
                if grad_step_total % target_update_steps == 0:
                    agent.update_target()

        loss_mean = (sum(losses) / len(losses)) if losses else None
        eps = float(agent.epsilon())

        last_slas.append(sla)
        last_slas = last_slas[-ma_window:]
        ma5 = float(sum(last_slas) / len(last_slas))

        loss_str = "NA" if loss_mean is None else f"{loss_mean:.4f}"

        print(
            f"[EP {ep:03d}] scen={scen.get('name','s')} W={workers[0]}-{workers[1]}-{workers[2]} "
            f"SLA={sla:.4f} | U={sla_u:.4f} | N={sla_n:.4f} | ma5={ma5:.4f} | "
            f"eps={eps:.3f} | loss={loss_str} | buf={len(buffer)} | upd={updates_done} | grad={grad_step_total} | "
            f"dec={dec} | %U={pct_u:.2f} | time={time.time()-t0:.1f}s"
        )

        with open(hist_path, "a", newline="", encoding="utf-8") as f:
            wcsv = csv.writer(f)
            wcsv.writerow([
                ep, scen.get("name", "s"), f"{workers[0]}-{workers[1]}-{workers[2]}", start_idx, dec, dec_cap,
                sla, sla_u, sla_n, eps, (loss_mean if loss_mean is not None else ""),
                len(buffer), updates_done, grad_step_total, pct_u, forced_u, forced_n
            ])

        if eval_enabled and eval_cache and (ep % eval_every == 0):
            eval_res = _eval_fixed(
                agent=agent,
                rl_cfg=rl_cfg,
                orders=orders,
                episode_orders=episode_orders,
                sim_cfg=simulation,
                base_resources=base_resources,
                service_cfg=service_time,
                cache=eval_cache,
            )
            for r in eval_res:
                print(
                    f"   └─ eval[{r['bucket']}] W={r['workers']} dec≈{r['dec_ref']} : "
                    f"SLA={r['sla']:.4f} | U={r['sla_u']:.4f} | N={r['sla_n']:.4f}"
                )

        if ep % ckpt_every == 0:
            ckpt_name = f"dqn_{run_name}_ckpt_ep{ep:03d}.pt" if run_name else f"dqn_ckpt_ep{ep:03d}.pt"
            ckpt_path = out_dir / ckpt_name
            torch.save(agent.q.state_dict(), ckpt_path)
            print(f"💾 guardado: {ckpt_path}")

    final_path = out_dir / (f"dqn_{run_name}_final.pt" if run_name else "dqn_final.pt")
    torch.save(agent.q.state_dict(), final_path)
    print(f"\n✅ Entrenamiento finalizado. Modelo: {final_path}")
    print(f"📝 Histórico: {hist_path}")


if __name__ == "__main__":
    main()