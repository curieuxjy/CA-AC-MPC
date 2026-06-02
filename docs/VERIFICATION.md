# Brax 포트 검증 매뉴얼 (다른 PC용)

다른(CUDA) PC에서 위에서 아래로 그대로 따라 하는 **단계별 검증 체크리스트**.
각 단계: **무엇을 검증** / **명령** / **기대 출력(통과 기준)** / **실패 시**.

> 전제: 이 레포의 Brax 포트는 전부 코드만 작성됐고 **실행 검증 0** 상태다. 이 문서의 목적은
> 실제 환경에서 한 단계씩 통과시키며 실측을 얻는 것.
> 환경 셋업 상세는 [`SETUP_CUDA.md`](SETUP_CUDA.md), 전체 개요는 [`../brax_import.md`](../brax_import.md).

**원칙: 의존성 적은 것부터.** 앞 단계가 통과해야 다음이 의미 있다. 실패하면 거기서 멈추고 원인부터 해결.

---

## 0. 사전 점검

```bash
nvidia-smi                 # 드라이버 + CUDA 버전(우상단) 확인
python3.10 --version       # 3.10 필요 (gym 0.21 제약)
git rev-parse HEAD         # 검증 대상 커밋 기록
```
- [ ] GPU 인식됨 (없으면 CPU로도 대부분 가능, MJX/학습만 느림)
- [ ] Python 3.10 사용 가능

---

## 1. 환경 셋업

```bash
cd /path/to/CA-AC-MPC
python3.10 -m venv .venv && source .venv/bin/activate
pip install --upgrade "pip<24.1" "setuptools<66" wheel

# CUDA 12 기준 (11이면 cu118 / jax[cuda11])
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -U "jax[cuda12]"
pip install numpy pyyaml matplotlib pandas tqdm rich "gym==0.21.0" "brax>=0.9.0"
pip install mujoco            # MJX(복잡 로봇) 단계용

export JAX_PLATFORMS=cpu      # 시작은 Brax CPU + Torch GPU (메모리 충돌 회피)
```
확인:
```bash
python -c "import torch,jax,brax,gym; print('torch',torch.__version__,'cuda',torch.cuda.is_available()); print('jax',jax.__version__); print('brax ok, gym',gym.__version__)"
```
- [ ] import 에러 없음
- [ ] `gym 0.21.0` 확인 (다른 버전이면 fork SB3와 충돌 가능)

> ⚠️ `pip install stable-baselines3` 하지 말 것 — 벤더링 포크를 `sys.path`로 쓴다.

---

## 2. 검증 사다리

### Step 1 — 템플릿 동역학 (torch만)
**검증**: Phase 2 이중적분기 모델의 Jacobian이 유한차분과 일치, 선형 정확성.
```bash
python tests/test_template_dx.py
```
**기대 출력 (통과 기준)**:
```
[ok] shapes + analytic Jacobian matches finite-difference
[ok] linear-exactness: forward == base + R·dx + S·du
[ok] DiffMPC wrapper interface
[ok] custom non-square B_a
All Phase-2 template-dynamics tests passed.
```
- [ ] 통과
**실패 시**: `AssertionError`의 max diff 출력 확인 → `dynamics/template_dx.py`의 R/S 부호·dt. torch 미설치면 환경 문제.

---

### Step 2 — 템플릿 ↔ 미분가능 iLQR ↔ cost (torch + DifferentialMPC)
**검증**: 가장 위험한 신규 결합. MPC solve가 유한·bound 준수, **gradient가 action→MPC→cost로 흐름**.
```bash
python tests/test_template_mpc_solve.py
```
**기대 출력**:
```
[ok] solve: X(4, 4, 12) U(4, 3, 6) finite, bounds respected
[ok] gradient flows: action -> MPC -> cost params (d u0 / d c != 0)
All Phase-5 template-MPC integration tests passed.
```
- [ ] 통과
**실패 시**:
- solve가 NaN/발산 → iLQR 컨디셔닝. 테스트의 cost 스케일이 문제면 `range_q` 영향. 실제 학습 전 신호.
- grad가 0 → `DifferentialMPC` 미분 경로(`ILQRSolve`) 문제. import 경로(`mpc.pytorch`) 확인.

---

### Step 3 — jax2torch 경계 (jax + torch) · Phase 7
**검증**: JAX 함수의 gradient가 PyTorch autograd로 정확히 흘러옴(`jax.grad`와 일치).
```bash
python tests/test_jax_torch_bridge.py
```
**기대 출력**:
```
[ok] jax2torch scalar grad matches jax.grad
[ok] jax2torch multi-output + partial-use backward
All Phase-7 jax2torch bridge tests passed.
```
- [ ] 통과
**실패 시**: DLPack 변환(`jnp.from_dlpack`/`torch.utils.dlpack`) API 버전 차이 → `envs/jax_torch_bridge.py`의 `_t2j/_j2t` fallback 확인. (APG를 안 쓸 거면 이 단계는 건너뛰어도 됨)

---

### Step 4 — Brax VecEnv 어댑터 (jax + brax) · Phase 1
**검증**: Brax→SB3 어댑터의 obs 레이아웃·autoreset·VecNormalize 래퍼 체인.
```bash
python tests/test_brax_vec_env.py
```
**기대 출력**:
```
[ok] basic step: obs_dim=<N> mpc_state_dim=<M>
[ok] auto-reset, episode info, terminal_observation
[ok] VecNormalize + ResetWithRawStateWrapper contract
All Phase-1 smoke tests passed.
```
- [ ] 통과
**실패 시**:
- `envs.create() unexpected keyword` / API 에러 → brax 버전 드리프트. `pip install "brax>=0.9,<0.11"`로 핀.
- `pipeline_state has no attribute 'q'` → 어댑터의 q/qd↔qpos/qvel 폴백 확인.

---

### Step 5 — MJCF/MJX env (jax + brax + mujoco) · 복잡 로봇
**검증**: 임의 MJCF(기본: Brax 번들 humanoid)를 MJX로 로드·step, 작동관절 레이아웃.
```bash
python tests/test_mjcf_env.py
```
**기대 출력**:
```
[ok] reset: obs_dim=<N> n_act=<K>
[ok] step: reward(2,) done(2,) finite obs
[ok] actuated MPC-state dim = <2K> (= 2*n_act)
MJCF/MJX env smoke test passed.
```
- [ ] 통과 (humanoid면 보통 `n_act≈17`, `2*n_act≈34`)
**실패 시**:
- `mujoco` 미설치 → `pip install mujoco`.
- MJX 첫 컴파일이 오래 걸림(수십 초~분) — 정상.
- 커스텀 MJCF면 freejoint/floor 필요(§ SETUP_CUDA 7b).

---

## 3. Smoke 학습 (전체 스택)

### Step 6 — PPO baseline (mlp_only) — 배관/PPO 루프 검증
```bash
python train_acmpc.py --config configs/train_mlp_brax_halfcheetah.yaml \
  --override ppo.total_timesteps=20000 --override vec_env.n_envs=8
```
**통과 기준**: 크래시 없이 끝까지 진행, PPO 로그(`rollout/ep_rew_mean` 등) 출력, NaN 없음, `[Train] done` 출력.
- [ ] 통과
**실패 시**: 여기서 깨지면 MPC가 아니라 **환경/배관** 문제 → Step 4로 복귀.

### Step 7 — acmpc_brax (template MPC) — MPC 경로 검증
```bash
python train_acmpc.py --config configs/train_acmpc_brax_halfcheetah.yaml \
  --override ppo.total_timesteps=20000 --override vec_env.n_envs=8
```
**통과 기준**: `[MPC] horizon=...`, `[Brax] Overriding mpc_state_dim ... -> 12`, 학습 진행, action/return NaN 없음.
- [ ] 통과
**실패 시**:
- action 포화/발산·NaN → `mpc_range_q`(↓), `mpc_range_p`(↓), `mpc_horizon`(2로↓), `mpc_ctrl_scale` 조정.
- Step 6은 되는데 7이 깨지면 문제는 **MPC에 국한** → Step 2 재확인.

### Step 8 — 평가
```bash
python utils/evaluate_brax.py \
  --model-path runs/acmpc_brax_halfcheetah/halfcheetah/model.zip \
  --robot halfcheetah --policy-type acmpc_brax --mpc-horizon 3 --episodes 10
```
**기대 출력**: `=== Brax evaluation ===` + `return: mean=... std=...` + `step: ... ms/vec-step (... env-fps)`.
- [ ] 통과

---

## 4. (선택) 복잡 로봇 / 실험적

### Step 9 — MJX humanoid smoke
```bash
python train_acmpc.py --config configs/train_acmpc_mjx_humanoid.yaml \
  --override ppo.total_timesteps=20000 --override vec_env.n_envs=4
```
- [ ] 크래시 없이 진행 (MJX 컴파일로 시작이 느림)

### Step 10 — APG (Phase 7, EXPERIMENTAL)
```bash
python utils/train_apg.py --config configs/train_apg_brax_halfcheetah.yaml \
  --override apg.total_timesteps=20000 --override vec_env.n_envs=8
```
- [ ] `[APG] window k/N return=... grad_norm=...` 로그가 NaN 없이 진행
**실패 시**: jax2torch 경계(Step 3)부터 재확인. 불안정하면 `apg.window`↓, `apg.dreplay_interval`↓.

---

## 5. 검증 결과 기록표

| Step | 내용 | 의존성 | 통과 | 비고(수치/에러) |
|---|---|---|---|---|
| 1 | template_dx | torch | ☐ | |
| 2 | template_mpc_solve | torch+DiffMPC | ☐ | |
| 3 | jax_torch_bridge | jax+torch | ☐ | |
| 4 | brax_vec_env | jax+brax | ☐ | |
| 5 | mjcf_env | +mujoco | ☐ | |
| 6 | PPO mlp_only smoke | full | ☐ | |
| 7 | PPO acmpc_brax smoke | full | ☐ | |
| 8 | evaluate_brax | full | ☐ | |
| 9 | MJX humanoid smoke | +mujoco | ☐ | |
| 10 | APG smoke | full | ☐ | |

---

## 6. 검증 후 본 실험 (Phase 6)

Smoke(6–8)가 통과하면 실제 비교 실험:
```bash
python train_acmpc.py --config configs/train_mlp_brax_halfcheetah.yaml      # baseline
python train_acmpc.py --config configs/train_acmpc_brax_halfcheetah.yaml    # actor-MPC
tensorboard --logdir runs/
```
비교 지표: 샘플효율(`rollout/ep_rew_mean` vs steps), 최종 return, `evaluate_brax`의 `ms/vec-step`(MPC 오버헤드).

---

## 7. 막힐 때 보고할 정보

다음을 캡처해 공유하면 바로 디버깅 가능:
1. 어느 **Step**에서 실패했는지 + 위 결과표.
2. **전체 에러 트레이스백**(마지막 30줄 이상).
3. 버전: `pip show torch jax jaxlib brax mujoco gym | grep -E "Name|Version"`.
4. (학습 실패 시) NaN/발산 직전 로그 + 사용한 `--override` 값.
5. (MJCF 커스텀 시) 사용한 `mjcf_path`와 freejoint/floor 유무.

## 알려진 위험(예상 실패 지점)
- brax 버전 드리프트 (`envs.create` 시그니처, `pipeline_state` 필드) — Step 4/5.
- iLQR 컨디셔닝 (`range_q/p`) — Step 2/7.
- 모델 불일치 하 PPO 수렴 — Step 7 (먼저 Step 6로 분리 진단).
- jax2torch 경계 — Step 3/10 (불안정 시 JAX 네이티브 재고).
- 커스텀 MJCF의 freejoint/관절-액추에이터 수 가정 — Step 5/9.
