# Brax 환경 이식 (CA-AC-MPC → Brax) 정리

이 레포의 핵심 아이디어(**actor가 미분가능 MPC의 cost를 예측 → MPC가 action 생성 → end-to-end 학습**)는
유지하고, **실험 환경만** quadrotor(custom Gym) → **Brax 다자유도 로봇**으로 교체한 작업의 전체 요약.
DiffMimic은 "differentiable simulator(Brax) 위에서 다자유도 로봇을 다룬다"는 환경/문제 세팅을 참고.

> 상세 단계별 계획: [`docs/brax_port_plan.md`](docs/brax_port_plan.md)
> CUDA 머신 셋업: [`docs/SETUP_CUDA.md`](docs/SETUP_CUDA.md)
> **다른 PC 검증 매뉴얼(단계별 체크리스트)**: [`docs/VERIFICATION.md`](docs/VERIFICATION.md)

---

## 1. 설계 결정 (확정)

| 축 | 선택 | 의미 |
|---|---|---|
| Framework | **PyTorch 유지 + Brax 브리지** | 기존 미분가능 MPC·CUDA 자산 재사용, Brax(JAX)는 환경으로 연결 |
| Training | **PPO 유지** (Phase 1–6) | model-free. Brax는 빠른 벡터화 env. APG는 Phase 7(옵션) |
| MPC dynamics | **축소/템플릿 모델** | `DroneDx` 대체: per-joint 이중적분기 |
| Robot/Task | **단순 로코모션부터** | HalfCheetah → Walker2d → Ant |

### 핵심 전제 2가지
1. **현 PPO 경로는 Brax 미분가능성을 사용하지 않는다.** 미분은 `action → MPC → actor` 경로에서만 일어나고,
   Brax는 "빠른 다자유도 시뮬레이터"일 뿐. DiffMimic의 핵심(APG)은 Phase 7로 분리.
2. **모델 불일치가 이번 이식의 핵심 난점.** 원본 drone은 env 시뮬레이터가 곧 MPC 모델(`DroneDx`)이라 불일치 0.
   Brax로 가면 `true sim(Brax) ≠ MPC model(template)` 불일치가 처음 생기고, **actor가 cost를 적응시켜 흡수**해야 함
   (AC-MPC 전제). → 첫 검증을 평면·contact 단순한 HalfCheetah로 시작하는 이유.

---

## 2. 아키텍처 매핑 (drone → Brax)

| 항목 | drone (원본) | Brax 템플릿 (HalfCheetah 예) |
|---|---|---|
| true sim | `DroneDx` (== MPC 모델) | Brax `halfcheetah` (MPC와 **불일치**) |
| MPC state `x` | `[p,q,v]` = 10 | `[q_act, qd_act]` = 2·n_act (6→12) |
| MPC ctrl `u` | CTBR thrust+rates = 4 | joint 토크/가속 = n_act (6) |
| 템플릿 동역학 | 비행 동역학 | per-joint 이중적분기 (선형, 상수 Jacobian) |
| actor 출력 | 28·T | 2·(nx+nu)·T |
| action map | u → physical_u | MPC `u_0 / u_max` (identity dist mean) |
| obs tail | `[v,R,o_track]` | Brax 네이티브 task obs |

**관측 레이아웃**: env가 `[mpc_state(prefix) | task_obs(tail)]` 단일 벡터를 emit.
`AcmpcPolicyObsExtractor`가 prefix를 잘라 tail만 cost-net에 전달, MPC는 `ResetWithRawStateWrapper`의
`(obs, raw_state)` 계약으로 un-normalized prefix를 받음. 이 계약은 backend-무관 → brax도 그대로 동작.

---

## 3. 단계별 진행 내역

| Phase | 내용 | 상태 |
|---|---|---|
| **1. VecEnv 어댑터** | Brax(batched JAX) → SB3 `VecEnv` 브리지 | ✅ 코드 완료 |
| **2. 템플릿 동역학** | `DroneDx` 대체 이중적분기 모델 + Jacobian | ✅ 코드 완료 |
| **3. 정책 일반화** | actor-MPC를 `nx,nu`로 일반화 (drone 비파괴) | ✅ 코드 완료 |
| **4. obs/reward/config** | Brax 네이티브 obs·reward 사용 (배관으로 충족) | ✅ |
| **5. 학습 통합·sanity** | signature·sanity 일반화 + MPC 통합 테스트 | ✅ 코드 완료 |
| **6. 베이스라인·평가** | Brax 평가기 + CUDA 셋업 가이드 | ✅ 코드 완료 |
| **7. DiffMimic 정렬 (APG)** | 미분가능 시뮬 통한 analytic gradient + DReplay | ⚠️ 코드 완료 (EXPERIMENTAL) |

### Phase 1 — Brax↔PyTorch VecEnv 어댑터
- `envs/brax_vec_env.py`: `BraxVecEnvAdapter` (표준 SB3 `VecEnv`), `make_actuated_joint_state_fn`.
- 단일 프로세스 vmap 배치(**SubprocVecEnv 금지**), JAX↔host는 **numpy**(DLPack zero-copy 불필요).
- **수동 autoreset**(jit step + masked reset)로 올바른 `terminal_observation`/`TimeLimit.truncated` 보존.
- `make_vec_env`가 붙이던 `Monitor` 부재 보완: `info["episode"]` 직접 emit.

### Phase 2 — 템플릿 동역학
- `dynamics/template_dx.py`: `TemplateDoubleIntegratorDx`(+`TemplateDxDiffMPCWrapper`).
- `x=[q,qd]`, `n_state=2·n_act`, `n_ctrl=n_act`. forward-Euler: `q'=q+dt·qd`, `qd'=qd+dt·ctrl_scale·B_a·u`.
- Jacobian 상수 `R=[[I,dt·I],[0,I]]`, `S=[[0],[dt·ctrl_scale·B_a]]` → forward와 grad_input이 **완전 일관**(선형계).
- 작동관절 = brax `q/qd`의 마지막 `n_act`개(MuJoCo root-first 규약, halfcheetah/walker2d/ant 성립).

### Phase 3 — 정책 일반화
- `acmpc_public-master/training_modules/mlp_mpc_policy_brax.py`: `MlpMpcPolicyBrax`/`CustomNetworkBrax`.
- drone의 14-블록 cost 제거 → `n_tau=nx+nu` 대각 Q(>0)+선형 c 일반식. `diffmpc` 백엔드 전용(fast는 drone 커널).
- action = MPC `u_0 / u_max`. n_act은 `action_space`에서 자동 취득. drone 경로는 **별도 파일로 비파괴**.
- 포크 SB3 호환 확인: `mlp_extractor(features,states)`/`forward_actor`/`forward_critic` + `latent_dim_pi/vf`.

### Phase 4 — obs/reward/config
- obs tail = Brax 네이티브 `state.obs`, reward = Brax 네이티브 로코모션 reward → PPO 그대로. 신규 코드 불필요.

### Phase 5 — 학습 통합·sanity
- `utils/train_support.py`: sanity에 `mpc_state_dim` 정합성 경고, signature에 `env_backend` 추가
  (구버전 drone 체크포인트 resume 안 깨지게 누락 키 미강제).
- `tests/test_template_mpc_solve.py`: 템플릿↔미분가능 iLQR↔cost 통합 + **gradient가 action→MPC→cost로 흐름** 검증.

### Phase 6 — 베이스라인·평가
- `utils/evaluate_brax.py`: return(=`evaluate_policy_vec` 재사용) + 스텝당 MPC 오버헤드(ms/step) + 선택적 HTML 렌더.
- `train_acmpc.py` post_eval에 brax 분기. drone `evaluate_acmpc2.py`(gate 전용)는 비파괴.

### Phase 7 — DiffMimic 정렬 (APG + DReplay) ⚠️ EXPERIMENTAL
- `envs/jax_torch_bridge.py`: `jax2torch` (Brax step을 `torch.autograd.Function`으로, backward=`jax.vjp`).
- `envs/diff_brax_env.py`: `DifferentiableBraxEnv` (flatten된 State를 torch로 carry → 다단계 grad 보존).
- `utils/train_apg.py`: short-horizon windowed return 미분 + DReplay 리셋 (PPO 아님).
- 로코모션 우선이라 imitation loss 대신 **Brax 네이티브 reward를 미분가능 시뮬 통해 직접 미분**(SHAC식 APG).
  `reference_fn` 훅은 향후 mocap imitation용. gradient가 **sim+MPC 양쪽** 통과.
- **한계**: JAX↔PyTorch 경계(`jax2torch`)가 fragile. 불안정하면 JAX 네이티브 재작성이 정도. humanoid mocap 미착수.

---

## 4. 산출물 (파일 목록)

**신규**
```
envs/brax_vec_env.py                  # P1 VecEnv 어댑터 (+mjcf_path 라우팅)
envs/mjcf_env.py                      # 복잡로봇: 범용 MJCF/MJX locomotion env
envs/jax_torch_bridge.py              # P7 jax2torch
envs/diff_brax_env.py                 # P7 미분가능 Brax env (+mjcf_path)
dynamics/__init__.py, template_dx.py  # P2 템플릿 동역학
acmpc_public-master/training_modules/mlp_mpc_policy_brax.py  # P3 정책
utils/evaluate_brax.py                # P6 평가기
utils/train_apg.py                    # P7 APG 트레이너
configs/train_mlp_brax_halfcheetah.yaml     # P1 실행 타깃 (mlp_only)
configs/train_acmpc_brax_halfcheetah.yaml   # P3 실행 타깃 (acmpc)
configs/train_apg_brax_halfcheetah.yaml     # P7 APG
configs/train_acmpc_mjx_humanoid.yaml       # 복잡로봇: MJX humanoid
tests/test_template_dx.py             # P2 (torch만)
tests/test_template_mpc_solve.py      # P5 (torch+DiffMPC)
tests/test_brax_vec_env.py            # P1 (jax+brax)
tests/test_jax_torch_bridge.py        # P7 (jax+torch)
tests/test_mjcf_env.py                # 복잡로봇 (jax+brax+mujoco)
requirements-brax.txt
docs/brax_port_plan.md, docs/SETUP_CUDA.md
.gitignore
```
**수정** (drone 경로 비파괴, 분기로 추가)
```
utils/train_support.py   # brax 분기·정책 등록·signature·sanity
train_acmpc.py           # env.backend 배선·ACMPC_BRAX_* env var·post_eval 분기
```

---

## 5. 실행 방법

상세는 [`docs/SETUP_CUDA.md`](docs/SETUP_CUDA.md). 요약:
```bash
# Python 3.10 venv (gym 0.21 제약), CUDA 12 예시
python3.10 -m venv .venv && source .venv/bin/activate
pip install --upgrade "pip<24.1" "setuptools<66" wheel
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -U "jax[cuda12]"
pip install numpy pyyaml matplotlib pandas tqdm rich "gym==0.21.0" "brax>=0.9.0"
export JAX_PLATFORMS=cpu     # 시작은 Brax CPU + Torch GPU (메모리 충돌 회피)

# 검증 사다리 (의존성 적은 순서)
python tests/test_template_dx.py            # torch
python tests/test_template_mpc_solve.py     # torch + DifferentialMPC
python tests/test_jax_torch_bridge.py       # jax + torch
python tests/test_brax_vec_env.py           # jax + brax

# 학습 (PPO)
python train_acmpc.py --config configs/train_mlp_brax_halfcheetah.yaml     # 베이스라인
python train_acmpc.py --config configs/train_acmpc_brax_halfcheetah.yaml   # actor-MPC

# 평가
python utils/evaluate_brax.py --model-path runs/acmpc_brax_halfcheetah/halfcheetah/model.zip \
  --robot halfcheetah --policy-type acmpc_brax --mpc-horizon 3 --episodes 30

# APG (Phase 7, 실험적)
python utils/train_apg.py --config configs/train_apg_brax_halfcheetah.yaml
```

---

## 6. 검증 상태 (솔직하게)

- ✅ **모든 파일 `py_compile` 통과**. 인터페이스 정합성은 소스로 교차 확인
  (포크 SB3 시그니처, 컨트롤러의 grad 라우팅 `ILQRSolve.apply`, forward/grad_input shape 계약).
- ❌ **실행 검증 전무**. 개발 머신에 torch/jax/brax 미설치 + `gym==0.21`이 Python 3.12와 비호환.
  CUDA도 없었음. 모든 검증은 CUDA + Python 3.10 환경 셋업 후 위 사다리로 수행해야 함.

### 실행 시 점검·보정 대상
- `mpc_range_q`/`mpc_range_p` — iLQR 컨디셔닝(발산·action 포화 시 축소).
- `mpc_horizon`(2–5), `mpc_ctrl_scale` — 모델 불일치 영향·가속 스케일.
- 모델 불일치 하 PPO 수렴 — 먼저 `mlp_only`가 학습되는지로 환경/배관 분리 진단.
- brax 버전 드리프트(`envs.create` 시그니처, `pipeline_state.q/qd` vs `qpos/qvel`) — 방어 코드 있으나 실측 필요.
- Phase 7: `jax2torch` 경계 그래디언트 정확성(`test_jax_torch_bridge.py`부터) — 불안정 시 JAX 네이티브 재고.

---

## 6b. 복잡 로봇 확장 (MJCF + MJX) ⚠️ 코드 완료 (실행 검증 안 됨)

Brax 내장(halfcheetah 등) 외에 **임의 MJCF 파일 로봇**(humanoid 등)을 MJX 백엔드로 실험.
선택: **Humanoid 로코모션 + 공개 MJCF 예제 + MJX**.

- `envs/mjcf_env.py`: **`MjcfLocomotionEnv`** — 임의 MJCF를 로드하는 범용 `PipelineEnv`
  (forward velocity + healthy - ctrl cost 보상, MJX Newton 솔버 설정). `mjcf_locomotion`으로 등록.
  `mjcf_path: brax_humanoid`면 Brax 번들 humanoid.xml(MJX-ready) 사용 → 추가 파일 없이 바로 실행.
- 어댑터/Phase7 env 확장: `BraxVecEnvAdapter`·`DifferentiableBraxEnv`에 `mjcf_path` 추가 →
  주어지면 `mjcf_locomotion`(MJX)로 라우팅. 기존 simple-robot 경로 비파괴.
- `utils/evaluate_brax.py`: `--mjcf-path` 추가. `configs/train_acmpc_mjx_humanoid.yaml`, `tests/test_mjcf_env.py` 신규.
- **재사용성**: 템플릿 모델은 로봇-agnostic(n_act만 필요), 작동관절 = `q/qd`의 마지막 n_act개(root-first).
  humanoid(17 act)→ mpc_state_dim 34. 어댑터가 자동 동기화.
- **주의**: 커스텀 MJCF는 floating base(freejoint)+floor 필요, 관절수=액추에이터수 가정, MJX 첫 컴파일 느림.
  의존성 `pip install mujoco`. 커스텀 로봇은 MuJoCo Menagerie 권장(URDF는 MJCF 변환). 상세: `docs/SETUP_CUDA.md` §7b.

## 7. 향후
- Phase 6 실측: `mlp_only` vs `acmpc_brax` 샘플효율/최종 return/MPC 오버헤드 비교.
- 로봇 확장: HalfCheetah → Walker2d → Ant.
- Phase 7 확장: humanoid mocap imitation(SMPL/AMP 로더 + state-matching loss via `reference_fn`),
  또는 경계가 fragile하면 actor-MPC의 JAX 네이티브 재작성.
