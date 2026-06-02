# CUDA 머신 환경 셋업 가이드 (Brax 포트)

이 레포의 Brax 포트(`docs/brax_port_plan.md`, Phase 1–6)를 **CUDA가 있는 다른 컴퓨터**에서
실행하기 위한 가이드. 두 가지 런타임 스택이 공존한다는 점이 핵심이다.

## 0. 어떤 부분이 CUDA를 쓰나

| 컴포넌트 | 프레임워크 | GPU 필요성 |
|---|---|---|
| 정책 MLP + 미분가능 MPC(`diffmpc`) | PyTorch | 선택(권장). CPU도 동작 |
| Brax 환경(시뮬레이터) | JAX | 선택. CPU도 동작하나 GPU가 훨씬 빠름 |
| drone `fast` 백엔드(원본 실험) | C++/CUDA 커널(nvcc) | **필수**. Brax acmpc는 사용 안 함 |

> Brax acmpc 경로(`policy_type: acmpc_brax`)는 `diffmpc`만 쓰므로 **nvcc 없이도** 돌아간다.
> nvcc는 원본 drone `fast` 백엔드를 재현할 때만 필요하다(아래 §7).

---

## 1. 사전 확인

```bash
nvidia-smi                 # 드라이버 + CUDA 런타임 버전(우상단 "CUDA Version") 확인
nvcc --version 2>/dev/null # 있으면 toolkit 버전 (drone fast 백엔드용; 없어도 brax는 OK)
```
- `nvidia-smi`의 CUDA 버전이 **12.x면 cu12 휠**, **11.x면 cu11 휠**을 설치한다(아래에서 torch/jax 모두 같은 메이저로 맞춘다).

---

## 2. Python 3.10 가상환경 (gym 0.21 제약)

이 레포는 `gym==0.21`을 핀한다. gym 0.21은 **Python ≥3.11 / 최신 setuptools에서 설치 실패**하므로
반드시 **Python 3.10** + 구버전 setuptools를 쓴다.

```bash
# Python 3.10이 없으면 (Ubuntu 예시)
sudo apt-get install -y python3.10 python3.10-venv

cd /path/to/CA-AC-MPC
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade "pip<24.1" "setuptools<66" wheel    # gym 0.21 빌드에 필요
```

---

## 3. 의존성 설치 (CUDA)

`nvidia-smi`의 CUDA 메이저에 맞춰 둘 중 하나를 고른다.

### CUDA 12.x
```bash
# 1) PyTorch (cu121)
pip install torch --index-url https://download.pytorch.org/whl/cu121
# 2) JAX (CUDA 12, 로컬 CUDA 불필요한 plugin 휠)
pip install -U "jax[cuda12]"
# 3) 나머지 (gym 0.21 포함)
pip install numpy pyyaml matplotlib pandas tqdm rich "gym==0.21.0" "brax>=0.9.0"
```

### CUDA 11.x
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -U "jax[cuda11]"
pip install numpy pyyaml matplotlib pandas tqdm rich "gym==0.21.0" "brax>=0.9.0"
```

> CPU만으로 먼저 검증하려면 `requirements-brax.txt`(루트)를 쓰면 된다.
> `pip install -r requirements-brax.txt` (torch/jax CPU 휠).

설치 확인:
```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import jax; print('jax', jax.__version__, jax.devices())"
python -c "import brax, gym, stable_baselines3" 2>/dev/null || \
  echo "참고: stable_baselines3는 벤더링본을 sys.path로 쓰므로 전역 설치 불필요"
```
(이 레포는 SB3 포크를 `acmpc_public-master/...`에서 `sys.path`로 로드하므로 `pip install stable-baselines3` 하지 말 것 — 충돌 위험.)

---

## 4. JAX ↔ PyTorch GPU 공존 (중요)

JAX(XLA)는 기본적으로 GPU 메모리의 ~75–90%를 **선점**한다. 같은 GPU에서 PyTorch(정책+MPC)가
메모리를 못 잡아 OOM이 난다. 두 가지 레이아웃 중 하나를 택한다.

### 레이아웃 A — Brax CPU + PyTorch GPU (권장 시작점, 가장 견고)
시뮬레이터는 CPU, 학습/MPC는 GPU. 메모리 충돌 없음. locomotion 64 envs면 충분히 실용적.
```bash
export JAX_PLATFORMS=cpu              # Brax를 CPU로 고정
# config: device: auto (torch가 GPU 사용), env.kwargs.brax_backend: generalized
```

### 레이아웃 B — Brax GPU + PyTorch GPU (최고 속도, 튜닝 필요)
둘 다 GPU. XLA 메모리 선점을 꺼야 한다.
```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# 또는 상한 고정:
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
```
> 주의: 현재 브리지는 JAX↔PyTorch 경계를 **numpy(host)** 로 넘긴다(스텝마다 host 왕복).
> 정확하지만 GPU↔GPU zero-copy는 아니다. DLPack 최적화는 향후 과제(plan Phase 7 메모).

공통:
```bash
export CUDA_HOME=/usr/local/cuda      # drone fast 백엔드 빌드 시 필요
export CUDA_VISIBLE_DEVICES=0         # 단일 GPU 지정(선택)
```

---

## 5. 검증 사다리 (순서대로)

의존성이 적은 것부터 올라간다. 앞 단계가 통과해야 다음이 의미 있다.

```bash
source .venv/bin/activate

# (1) 템플릿 동역학 — torch만. Jacobian/선형정확성
python tests/test_template_dx.py

# (2) 템플릿 ↔ 미분가능 iLQR ↔ cost — torch + DifferentialMPC. solve·미분 경로
python tests/test_template_mpc_solve.py

# (3) Brax env 어댑터 — jax + brax. obs 레이아웃·autoreset·래퍼 체인
python tests/test_brax_vec_env.py

# (4) full-stack smoke — 전체. 20k step 학습이 도는지(수렴 아님)
python train_acmpc.py --config configs/train_mlp_brax_halfcheetah.yaml --override ppo.total_timesteps=20000
python train_acmpc.py --config configs/train_acmpc_brax_halfcheetah.yaml --override ppo.total_timesteps=20000 --override vec_env.n_envs=8
```

---

## 6. 본 실험 (Phase 6 베이스라인 비교)

같은 로봇에서 `mlp_only`(MPC 없음) vs `acmpc_brax`(템플릿 MPC)를 학습·비교한다.

```bash
# 베이스라인
python train_acmpc.py --config configs/train_mlp_brax_halfcheetah.yaml
# 본 방법
python train_acmpc.py --config configs/train_acmpc_brax_halfcheetah.yaml

# 평가 (return + 스텝당 MPC 오버헤드 측정, 선택적 HTML 렌더)
python utils/evaluate_brax.py \
  --model-path runs/acmpc_brax_halfcheetah/halfcheetah/model.zip \
  --robot halfcheetah --policy-type acmpc_brax --mpc-horizon 3 \
  --episodes 30 --render rollout.html
```

비교 지표: 샘플효율(return vs steps, TensorBoard `rollout/ep_rew_mean`), 최종 return,
`evaluate_brax.py`의 `ms/vec-step`(acmpc vs mlp의 MPC 비용 차이), 학습 안정성.
TensorBoard: `tensorboard --logdir runs/`.

실측 시 보정 포인트(plan에 명시):
- `mpc_range_q` / `mpc_range_p` (config) — iLQR 컨디셔닝. solve가 발산하거나 action이 포화되면 줄인다.
- `mpc_horizon` — 2~5에서 시작. 길수록 비용↑, 모델 불일치 영향↑.
- `mpc_ctrl_scale` — 템플릿 가속↔실제 토크 스케일.
- 안 되면 먼저 `mlp_only`가 학습되는지로 환경/배관을 분리 진단.

---

## 7. (선택) 원본 drone `fast` CUDA 백엔드 재현

Brax 포트엔 불필요. 원본 quadrotor 실험의 CUDA 커널을 쓸 때만:
```bash
export CUDA_HOME=/usr/local/cuda      # nvcc가 여기 있어야 함
nvcc --version                        # 동작 확인
python train_acmpc.py --config configs/train_acmpc_fixed_map.yaml  # 첫 실행 시 커널 JIT 컴파일
```
빌드 산출물은 `differentialMPCPerformance/build/`. 깨끗한 재빌드는 이 디렉터리 삭제.

---

## 7b. 복잡 로봇 (MJCF + MJX)

Brax 내장 외에 **임의 MJCF 로봇**(humanoid 등)을 MJX 백엔드로 돌리는 경로.

### 의존성
```bash
pip install mujoco            # mujoco.mjx 포함 (brax의 backend='mjx'가 사용)
```

### 바로 실행 (Brax 번들 humanoid, 추가 파일 불필요)
`mjcf_path: brax_humanoid`는 Brax에 내장된 MJX-ready `humanoid.xml`을 쓴다.
```bash
python tests/test_mjcf_env.py        # MJCF/MJX env 로드·step 확인 (jax+brax+mujoco)
python train_acmpc.py --config configs/train_acmpc_mjx_humanoid.yaml \
  --override ppo.total_timesteps=20000 --override vec_env.n_envs=8
python utils/evaluate_brax.py --model-path runs/acmpc_mjx_humanoid/humanoid/model.zip \
  --robot humanoid --policy-type acmpc_brax --mjcf-path brax_humanoid --brax-backend mjx --mpc-horizon 3
```

### 커스텀 MJCF 로봇 (예: MuJoCo Menagerie)
```bash
git clone https://github.com/google-deepmind/mujoco_menagerie
# config의 env.kwargs.mjcf_path를 해당 .xml로 교체, 예:
#   mjcf_path: mujoco_menagerie/unitree_h1/scene.xml
```
주의:
- **floating base 필요**: 로코모션은 base에 `freejoint`가 있어야 한다(첫 7 qpos / 6 qvel = free base).
  Menagerie 로봇 xml은 보통 고정 베이스이거나 floor가 없으니, `scene.xml`(floor 포함)을 쓰거나
  freejoint+floor를 넣은 wrapper MJCF를 만든다.
- **작동관절 추출 가정**: 어댑터는 `q/qd`의 마지막 `n_act`개를 작동관절로 본다(root-first 규약).
  관절 수 ≠ 액추에이터 수(mimic/unactuated joint)면 어긋난다 → `tests/test_mjcf_env.py` 출력으로 확인.
- **MJX 컴파일**: 첫 step에서 XLA 컴파일이 오래 걸린다(수십 초~분). `n_envs`는 작게 시작.
- **URDF**: Brax 네이티브 로더는 제한적. MuJoCo로 MJCF 변환 후 사용
  (`<mujoco><include>`/`mj_loadXML`, 또는 `mjcf` 도구). 변환본을 `mjcf_path`로.

### Phase 7(APG)도 동일 적용
`configs/train_apg_brax_halfcheetah.yaml`의 `env.kwargs`에 `mjcf_path`/`brax_backend: mjx`를
넣으면 `DifferentiableBraxEnv`도 MJCF 로봇으로 APG를 돌린다(단, MJX 접촉 그래디언트는 노이즈가
있어 APG 안정성은 더 불확실 — DReplay·짧은 window 필수).

## 8. 트러블슈팅

| 증상 | 원인 / 해결 |
|---|---|
| `gym==0.21` 설치 실패 (`error in gym setup command`) | Python ≥3.11 또는 새 setuptools. → Python 3.10 + `setuptools<66`, `pip<24.1` |
| OOM (둘 다 GPU) | XLA 선점. → `XLA_PYTHON_CLIENT_PREALLOCATE=false` 또는 레이아웃 A(`JAX_PLATFORMS=cpu`) |
| `jaxlib`/CUDA 버전 불일치, `jax.devices()`에 GPU 없음 | torch와 JAX의 CUDA 메이저 불일치. → 둘 다 cu12 또는 둘 다 cu11 |
| `AttributeError: ... pipeline_state has no attribute 'q'` | brax 버전 차이(MJX 백엔드 등). → `brax_backend: generalized` 사용. 어댑터는 q/qd↔qpos/qvel 폴백 있음 |
| `envs.create() unexpected keyword` | brax API 드리프트. → `pip install "brax>=0.9,<0.11"` 범위로 핀, 또는 `envs/brax_vec_env.py`의 create 호출부 조정 |
| acmpc 학습은 도는데 발산/포화 | `mpc_range_q/p` 과대. → 100/10에서 점차 축소. `mpc_max_iter`도 조정 |
| `stable_baselines3` import가 PyPI본을 잡음 | 전역 설치 제거. 벤더링 포크가 `sys.path` 우선이어야 함 |

---

## 빠른 시작 요약
```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install --upgrade "pip<24.1" "setuptools<66" wheel
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -U "jax[cuda12]"
pip install numpy pyyaml matplotlib pandas tqdm rich "gym==0.21.0" "brax>=0.9.0"
export JAX_PLATFORMS=cpu            # 시작은 레이아웃 A로 안전하게
python tests/test_template_dx.py && python tests/test_template_mpc_solve.py && python tests/test_brax_vec_env.py
python train_acmpc.py --config configs/train_acmpc_brax_halfcheetah.yaml --override ppo.total_timesteps=20000 --override vec_env.n_envs=8
```
