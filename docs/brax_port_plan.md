# CA-AC-MPC → Brax 이식 계획 (locomotion 검증)

> 목표: 이 레포의 **핵심 아이디어**(actor가 미분가능 MPC의 cost를 예측 → MPC가 action 생성 → PPO로 end-to-end 학습)를
> 그대로 유지하고, **실험 환경만** quadrotor(custom Gym) → **Brax 다자유도 로봇**으로 교체해 검증한다.
> DiffMimic은 "differentiable simulator(Brax) 위에서 다자유도 캐릭터를 다룬다"는 환경/문제 세팅을 참고한다.

## 확정된 설계 결정 (사용자 선택)

| 축 | 선택 | 의미 |
|---|---|---|
| Framework | **PyTorch 유지 + Brax 브리지** | 기존 미분가능 MPC·CUDA 커널 그대로 재사용. Brax(JAX)는 환경으로만 연결 |
| Training | **PPO 유지** | model-free. Brax는 빠른 벡터화 env. (APG는 향후 옵션) |
| MPC dynamics | **축소/템플릿 모델** | `DroneDx`를 대체하는 단순 동역학 모델 + Jacobian |
| Robot/Task | **단순 로코모션부터** | HalfCheetah / Walker2d / Ant → 단계적 |

### 솔직한 전제 (중요)
- 이 조합(PyTorch+브리지+PPO)에서는 **Brax의 미분가능성을 실제로 사용하지 않는다.** PPO는 env에 대해 model-free이고,
  미분은 오직 *action → MPC → actor cost-net* 경로에서만 일어난다. 즉 현 단계의 Brax는 "빠른 다자유도 시뮬레이터"일 뿐이다.
  DiffMimic의 핵심(APG, analytic gradient through sim)은 **Phase 7(옵션)** 으로 분리한다.
- 이 레포에서 drone env의 시뮬레이터는 곧 MPC가 쓰는 `DroneDx`와 **동일 모델**이다(`gate_racing_env.py:287`이 `self.dx.forward` 호출).
  따라서 model mismatch가 0이다. Brax로 가면 **true sim(Brax) ≠ MPC model(template)** 이 되어 **모델 불일치**가 처음 생긴다.
  → 이것이 이번 이식의 핵심 난점이자 연구 포인트다. AC-MPC 철학상 actor가 cost를 적응시켜 이 불일치를 흡수해야 한다.

---

## 현재 구조에서 바꿔야 할 seam (코드 근거)

1. **Env** — `envs/gate_racing_env.py`
   - `action_space = Box(4,)`, `observation_space = Box(mpc_state_dim + tail,)`, `self.state`(10-dim)이 MPC raw state.
   - `step`: action → physical_u → `self.dx.forward` 로 내부 적분 (사실상 DroneDx가 시뮬레이터).
   - `_get_obs`: `[mpc_state(prefix) | paper_obs tail([v,R,o_track])]` 결합.
2. **Obs 분리** — `utils/acmpc_obs_extractor.py`
   - `AcmpcPolicyObsExtractor`가 앞 `mpc_state_dim` 잘라내고 tail만 cost-net에 전달. (이 계약 유지)
3. **VecEnv/state 계약** — `utils/train_support.py`
   - `ResetWithRawStateWrapper`: 포크된 SB3가 `reset() -> (obs, raw_state)` 기대.
   - `_make_vec_env_from_cfg`: `DummyVecEnv/SubprocVecEnv` + `VecNormalize`(obs 정규화하되 MPC엔 un-normalized prefix 전달).
4. **Policy/MPC** — `acmpc_public-master/training_modules/mlp_mpc_policy_diffmpc.py`
   - `self.dx = drone.DroneDx(...)` (nx=10, nu=4) → `DifferentiableMPCController`/`FastMPC` 구성.
   - actor 출력 `n_o = 28 = 2*(nx+nu) = 2*14` (대각 Q 14 + 선형 c 14), per-step, Sigmoid.
   - `_build_cost_tensors`가 sigmoid → (C diag, c) 로 스케일. `u_min/u_max`는 drone 전용.
5. **동역학 모델** — `acmpc_public-master/diff_mpc_drones/drone.py`
   - `DroneDx`: `n_state=10, n_ctrl=4, dt=0.02, forward(x,u), grad_input(x,u)->(R,S)` 이산 Jacobian.
   - 인터페이스 계약은 `DroneDxDiffMPCWrapper.f_dyn / f_dyn_jac` (policy가 이걸 MPC에 주입).
6. **설정 → env var** — `train_acmpc.py`가 `ACMPC_T / ACMPC_MPC_MAX_ITER / ACMPC_MPC_BACKEND` 를 `os.environ`에 쓰면 policy가 읽음.

---

## 매핑: drone → Brax 로코모션

| 항목 | drone (현재) | Brax 템플릿 (목표, 예: HalfCheetah) |
|---|---|---|
| true sim | `DroneDx` (analytic, == MPC model) | Brax `halfcheetah` (true), MPC와 **불일치** |
| MPC state `x` | `[p(3),q(4),v(3)]`=10 | `[q, qd]` of 작동 joints (예: 6+6=12) |
| MPC ctrl `u` | CTBR thrust+rates=4 | joint 토크/가속 명령 (예: 6) |
| 템플릿 동역학 | 비행 동역학 | **per-joint double integrator** (선형, Jacobian 상수) |
| actor 출력 | `28*T` | `2*(nx+nu)*T` (HalfCheetah: 36*T) |
| action map | u → physical_u | MPC `u_0` → Brax 토크(클립/스케일) |
| obs tail | `[v,R,o_track]` | task feature (목표 속도, 자세, phase 등) |

추천 템플릿 동역학 = **작동 자유도별 이중적분기**: `x=[q,qd]`, `x_next=[q+dt·qd, qd+dt·a]`, `a=B·u`.
- Jacobian이 상수 → `grad_input`이 자명, iLQR이 사실상 시간가변 LQR. 안정적이고 디버깅 쉬움.
- 불일치는 actor가 cost로 보정 (AC-MPC 철학). 첫 검증에 최적.
- (향후) centroidal/learned dynamics로 교체 가능 — 같은 `f_dyn/f_dyn_jac` 인터페이스 유지.

---

## 단계별 계획

### Phase 0 — 환경/의존성
- JAX + Brax 설치. **버전 충돌 주의**: 레포는 `gym==0.21` 핀. Brax는 최신 gym/gymnasium 가정 → Brax의 gym wrapper에 의존하지 말고 **Brax `State` pytree를 직접** 다루는 어댑터를 작성.
- GPU 메모리: JAX(XLA)가 기본 90% 선점 → `XLA_PYTHON_CLIENT_PREALLOCATE=false` (또는 `XLA_PYTHON_CLIENT_MEM_FRACTION`) 설정해 PyTorch와 공존.
- 산출물: `requirements-brax.txt`, README의 setup 노트 보강.

### Phase 1 — Brax↔PyTorch VecEnv 어댑터 (핵심 신규 인프라) ✅ 코드 완료 (실행 검증은 환경 셋업 후)
구현됨: `envs/brax_vec_env.py`(`BraxVecEnvAdapter`), `utils/train_support.py`(brax 분기),
`train_acmpc.py`(`env.backend` 전달 + mpc_state_dim 동기화), `tests/test_brax_vec_env.py`,
`configs/train_mlp_brax_halfcheetah.yaml`, `requirements-brax.txt`.
설계 메모: numpy 경계(DLPack 불필요), 수동 autoreset로 terminal_obs 보존, Monitor 대체용 episode info emit.

- 신규 `envs/brax_vec_env.py`: `BraxVecEnvAdapter`
  - 내부에 **jit·vmap된 Brax batched env**(batch = `n_envs`)를 단일 프로세스로 보유. **SubprocVecEnv 사용 금지** (Brax는 이미 GPU 벡터화).
  - SB3 VecEnv API 제공: `reset()->(obs, raw_state)`, `step(actions)->(obs, rewards, dones, infos)`.
  - JAX↔torch 변환은 **DLPack**(zero-copy, 같은 디바이스) 사용.
  - obs를 `[mpc_state prefix | policy obs tail]`로 구성, `raw_state`(un-normalized prefix) 별도 반환 → `ResetWithRawStateWrapper` 계약 충족.
  - 자동 reset(done 시 Brax state 리셋) 처리.
- `utils/train_support.py::_make_vec_env_from_cfg`에 `backend: brax` 분기 추가 (기존 gym 경로 보존).
- `VecNormalize`는 obs에만 적용하고 MPC엔 raw prefix 전달하는 기존 계약 유지.

### Phase 2 — 템플릿 동역학 모듈 (DroneDx 대체) ✅ 코드 완료 (실행 검증은 환경 셋업 후)
구현됨: `dynamics/template_dx.py`(`TemplateDoubleIntegratorDx` + `TemplateDxDiffMPCWrapper`),
`envs/brax_vec_env.py`(`make_actuated_joint_state_fn` + `mpc_state_mode="actuated"`),
`utils/train_support.py`(`mpc_state_mode` 전달), `tests/test_template_dx.py`(유한차분/선형정확성 검증).
설계 메모: `x=[q,qd]`, `n_state=2·n_act`, `n_ctrl=n_act`(B_a로 비정방도 지원), forward(Euler)와 grad_input이
선형계라 **완전 일관**(linearization error 0). 작동관절 = brax `q/qd`의 마지막 `n_act`개(root-first 규약).
**주의(Phase 3 선결)**: acmpc 정책은 아직 `DroneDx`를 생성 → 정책이 `TemplateDx`를 쓰고 cost 차원을
`nx,nu`로 일반화해야 Brax에서 end-to-end로 돈다. acmpc용 brax config는 `kwargs.mpc_state_mode: actuated` 필요.

- 신규 `differentialMPCPerformance/dynamics/` 또는 별도 `dynamics/template_dx.py`: `TemplateDx(nn.Module)`
  - `n_state, n_ctrl, dt, forward(x,u), grad_input(x,u)->(R,S)` — DroneDx와 동일 시그니처.
  - 이중적분기: 해석적 상수 Jacobian. (배치 지원, torch)
  - 로봇별 `(nx, nu, B, 토크 한계)` 설정화.
- 검증: 단위테스트로 `grad_input` vs `torch.autograd`(finite-diff) 일치 확인 (`DifferentialMPC/utils.py`에 finite-diff 헬퍼 존재).

### Phase 3 — Policy 일반화 ✅ 코드 완료 (실행 검증은 환경 셋업 후)
구현됨: `acmpc_public-master/training_modules/mlp_mpc_policy_brax.py`(`MlpMpcPolicyBrax`/`CustomNetworkBrax`),
`utils/train_support.py`(`acmpc_brax` 등록 + eval backend), `train_acmpc.py`(backend 강제 diffmpc + `ACMPC_BRAX_*` env var),
`configs/train_acmpc_brax_halfcheetah.yaml`.
설계 메모: drone 경로 비파괴(별도 파일). `TemplateDx`(nx=2·n_act, nu=n_act) 사용, cost를 `n_tau=nx+nu` 대각 Q(>0)+선형 c로
일반화(drone의 p/q/v/w/t 분리 제거), backend=diffmpc 전용(fast는 drone 커널). action = MPC `u_0 / u_max`(identity dist mean).
포크 SB3 호환 확인: `mlp_extractor(features,states)`/`forward_actor`/`forward_critic` + `latent_dim_pi/vf` 시그니처 일치(policies.py:588/674/685).
n_act는 `action_space`에서 자동 취득. acmpc용 brax config는 `mpc_state_mode: actuated` 필수.
- `MlpMpcPolicyDiffMPC`를 drone-하드코딩에서 분리:
  - `self.dx = TemplateDx(...)` 주입 (env var/config로 로봇 선택).
  - `n_o`, `_build_cost_tensors`를 `nx,nu`로 일반화 (현재 14-블록(p/q/v/w/t) 가정 제거 → 대각 Q(nx+nu)+선형 c(nx+nu) 일반식).
  - `u_min/u_max`를 로봇 토크 한계로.
  - MPC `u_0` → Brax action 매핑 함수 (스케일/클립). 필요시 `mlp_mpc_policy_diffmpc.py` 복제본 `mlp_mpc_policy_brax.py`로 분리해 drone 코드 비파괴.
- `resolve_policy_class`에 신규 policy 등록.

### Phase 4 — 관측/보상/설정 ✅ (Phase 1–3 배관으로 충족)
- obs tail = Brax 네이티브 task obs(`state.obs`), reward = Brax 네이티브 로코모션 reward → PPO 그대로. config 작성 완료.
- VecNormalize는 full obs 정규화하되 MPC엔 un-normalized prefix 전달(`get_original_obs`) — backend-무관이라 brax도 그대로 동작.
- 별도 신규 코드 불필요(과장 회피). 향후 goal-conditioned 태스크 시 task_obs_fn 확장 지점만 열려 있음.

### Phase 5 — 학습 통합 & sanity check ✅ 코드 완료 (실행 검증은 환경 셋업 후)
구현됨: `tests/test_template_mpc_solve.py`(템플릿↔iLQR↔cost 통합·미분 검증, torch만), `utils/train_support.py`
(sanity check에 mpc_state_dim 정합성 경고 추가 + signature에 `env_backend` 추가, 구버전 체크포인트 호환되게 누락 키 skip),
`train_acmpc.py`(signature에 env_backend 전달).
설계 메모: signature 확장이 drone resume를 깨지 않도록 `_check_signature_compatibility`가 observed에 없는 키는 미강제.
smoke run = `--override ppo.total_timesteps=20000`이 곧 full-stack 통합 검증.

### Phase 4 — 관측/보상/설정 (원본 계획, 참고)
- Brax env의 task feature를 obs tail로 노출 (로코모션: 전진속도, 몸통 자세/높이, 목표속도, 필요시 phase).
- 보상: Brax 기본 로코모션 reward 사용 (forward velocity − ctrl cost 등) → PPO 그대로.
- 신규 config `configs/train_acmpc_brax_halfcheetah.yaml`: `policy_type`, `mpc_backend`, `mpc_horizon(T=2~5 권장)`, `mpc_state_dim`(=nx), `env.backend: brax`, `env.robot: halfcheetah`, `vec_env.type: brax`.

### Phase 5 — 학습 통합 & sanity check
- `_run_input_sanity_checks`를 새 차원에 맞게 일반화 (`mpc_state_dim == nx` 확인).
- training signature에 `robot`, `template_dynamics` 추가 (resume 호환성).
- smoke run: `total_timesteps` 소량으로 그래프/그래디언트 흐름·NaN·MPC 수렴 확인.

### Phase 6 — 베이스라인 & 평가 ✅ 코드 완료 (실행 검증은 환경 셋업 후)
구현됨: `utils/evaluate_brax.py`(Brax 평가기 — return 지표는 `evaluate_policy_vec` 재사용, 스텝당 MPC 오버헤드 측정,
선택적 HTML 렌더), `train_acmpc.py`(post_eval에 brax 분기), `docs/SETUP_CUDA.md`(CUDA 머신 셋업 가이드).
비교: 동일 로봇에서 `mlp_only` vs `acmpc_brax`(+CUDA 시 속도). 지표 = 샘플효율/최종 return/ms·step/안정성.
drone `evaluate_acmpc2.py`는 gate 전용이라 비파괴, Brax는 별도 평가기로 분리.

### Phase 6 — 베이스라인 & 평가 (원본 계획, 참고)
- 동일 Brax 로봇에서 비교:
  - `mlp_only` (MPC 없음, 순수 PPO) — 대조군.
  - `acmpc_diffmpc` + `diffmpc` 백엔드 — 본 방법.
  - (CUDA 가능시) `fast` 백엔드 — 속도 이득 재현(레포의 ~10x 주장 검증).
- 지표: 샘플효율(return vs steps), 최종 성능, MPC solve 시간/스텝, 학습 안정성.
- `utils/evaluate_acmpc2.py`를 Brax 평가 경로로 확장(롤아웃/플롯; 영상은 Brax HTML 렌더 활용 가능).

### Phase 7 — DiffMimic 정렬 (APG + DReplay) ⚠️ 코드 완료 (EXPERIMENTAL, 실행 검증 안 됨)
구현됨: `envs/jax_torch_bridge.py`(`jax2torch`: Brax step을 torch.autograd.Function으로 감싸 backward=`jax.vjp`),
`envs/diff_brax_env.py`(`DifferentiableBraxEnv`: flatten된 state를 torch로 carry해 다단계 grad 보존),
`utils/train_apg.py`(short-horizon windowed return 미분 + DReplay 리셋, PPO 아님), `configs/train_apg_brax_halfcheetah.yaml`,
`tests/test_jax_torch_bridge.py`(jax2torch grad vs jax.grad).
설계 메모: 로코모션 우선이라 DiffMimic의 imitation loss 대신 **Brax 네이티브 reward를 미분가능 시뮬 통해 직접 미분**(SHAC식 short-horizon APG).
`reference_fn` 훅은 향후 mocap imitation용으로 열어둠. gradient가 sim+MPC 양쪽을 통과(action→MPC→actor params, reward→sim→action→...).
**한계(명시)**: JAX↔PyTorch 경계(`jax2torch`)가 fragile. 불안정하면 JAX 네이티브 actor-MPC 재작성이 정도. humanoid mocap은
데이터 자산(SMPL/AMP) 필요해 미착수. 실행 검증 전무(의존성 미설치) — 검증 사다리 1순위는 `tests/test_jax_torch_bridge.py`.

### Phase 7 — (옵션/향후) DiffMimic 정렬 확장 (원본 계획, 참고)
- **APG 모드**: Brax 미분가능성을 켜서 imitation loss를 sim·MPC 통해 actor로 직접 역전파.
  - PyTorch 경계에선 `jax2torch`로 Brax 그래디언트를 autograd로 노출 필요(브리지 한계 → 이때 JAX 네이티브 재고).
  - **DReplay/RSI**: 주기적으로 sim state를 reference 모션으로 리셋해 그래디언트 horizon 단축·발산 억제.
- **Humanoid mocap imitation**: DiffMimic과 직접 비교 가능한 태스크로 확장.
- 이 단계는 별도 의사결정(프레임워크 재검토) 후 진행.

---

## 주요 리스크 / 체크포인트
- **모델 불일치**: template ≠ Brax. 완화 = 단순/준선형 로코모션부터, 짧은 horizon(T=2~5), 보수적 토크 한계, actor가 cost로 보정. 불일치가 크면 학습 실패 가능 → Ant(3D, contact-rich)보다 **HalfCheetah/Walker2d(평면)** 우선.
- **JAX/PyTorch 한 GPU 공존**: 메모리 선점·DLPack 디바이스/동기화 이슈. 초기엔 작은 batch로.
- **gym 0.21 vs Brax**: API 불일치 → Brax wrapper 미사용, State 직접 처리.
- **VecNormalize 계약**: 정규화 obs와 raw MPC prefix 분리 유지 (현 구조 보존).
- **action 매핑 스케일**: MPC u(가속/토크) → Brax 토크 단위/범위 보정이 성능에 민감.

## 첫 타깃 권장
**HalfCheetah** (평면, contact 단순, PPO 강한 베이스라인 존재) → 파이프라인 안정화 후 Walker2d → Ant.

## 변경/신규 파일 요약
- 신규: `envs/brax_vec_env.py`, `dynamics/template_dx.py`(또는 `differentialMPCPerformance/dynamics/`),
  `mlp_mpc_policy_brax.py`, `configs/train_acmpc_brax_*.yaml`, `requirements-brax.txt`, 단위테스트.
- 수정: `utils/train_support.py`(`_make_vec_env_from_cfg`, `resolve_policy_class`, signature, sanity check),
  `train_acmpc.py`(robot/backend 선택 전달), `utils/evaluate_acmpc2.py`(Brax 평가).
- 비파괴: drone 경로(`gate_racing_env.py`, `drone.py`, `mlp_mpc_policy_diffmpc.py`)는 그대로 두고 분기로 추가.
