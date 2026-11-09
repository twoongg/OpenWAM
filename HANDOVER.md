# HANDOVER — video-expert TTT (agent-driven evolving WAM)

- 레포: `https://github.com/twoongg/OpenWAM` (private), 작업 브랜치 **`feat/video-ttt`**
- 베이스: upstream `OpenWAM-Official/OpenWAM` @ `a00c4ba` (2026-11-02)
- 작업 환경: 개인 Mac(M4, 16GB, CUDA 없음) — **CPU 테스트·구조 검증까지만. 실가중치·GPU로는 아직 한 번도 안 돌림.**
- 설계·근거·결정 이유: [`docs/video-ttt.md`](docs/video-ttt.md) (이 문서는 "무엇을 했고 어디서 이어갈지"만)

## 0. 실험 계획 (내가 정한 것)

1. **1번 — 성능 유지 확인**: RoboDojo 트레이닝셋만으로 TTT 레이어(만) 학습 → 에이전트 없이 평가했을 때 릴리즈 체크포인트와 비슷한 점수인지.
   - 1-a: **Generalization 12개 task**로 학습·평가 (릴리즈 점수: GenStd SR 25.56 / GenRand SR 4.11)
   - 1-b: 학습 데이터가 있는 **35개 task 전체**로 학습 → 42개 task 평가 (Open 8개는 demo가 없어서 학습 불가)
2. **2번 — evolving**: task마다 TC 순서대로 진행, 실패 시 에이전트가 실시간으로 도와 성공 → 그 성공을 TTT에 쌓으면서 점수가 오르는 곡선.

주의: TTT를 붙이기만 한(학습 전) 모델은 릴리즈와 출력이 **완전히 동일** (θ_O zero-init, 테스트로 확인). 1번에서 실제로 보는 건 "TTT 학습 후에도 안 망가졌나". 그리고 1번 통과 ≠ 2번 성공 — TTT가 "아무것도 안 하는 층"이 돼도 1번은 통과하므로 학습 중 `train/ttt_context_gain`(아래)을 꼭 볼 것.

---

## 1. 여기서 한 것

| 항목 | 상태 |
|---|---|
| TTT 메모리 `openwam/model/ttt/video_ttt.py` (SwiGLU fast weight, KV 재구성 쓰기, 누적/리셋, weight norm, zero-init 출력) | ✅ |
| joint loop(`mot_driver.py::step`) + video-only 경로 hook (checkpoint 구간 바깥) | ✅ |
| config `model.architecture.ttt.*` (`configs/model/dual_system.yaml`, 기본 off) | ✅ |
| `arch.restrict_training_to_ttt()` (TTT만 학습. `freeze:` 쓰면 gradient 끊김) | ✅ |
| `video_ttt.*` 없는 체크포인트 strict load 허용 | ✅ |
| 체크포인트 호환성 체크 `scripts/ttt/check_checkpoint_compat.py` | ✅ RoboDojo ckpt 통과 |
| **[1번용] task 선택**: `MultiTaskRoboDojoDataset(tasks=[...])` / dataloader `tasks:` | ✅ |
| **[1번용] 같은 task 쌍 dataset** `openwam/dataloader/ttt_pairs.py` (`type: robodojo_ttt_pairs`) | ✅ |
| **[1번용] meta-training step** `openwam/train/utils/ttt_meta.py` + trainer 연결 (`training.ttt_meta`) | ✅ |
| **[1번용] config** `configs/dataloader/robodojo_ttt_gen.yaml` (Generalization 12 task) | ✅ Hydra compose 확인 |
| 테스트 (TTT 54개: `tests/test_video_ttt*.py`, `test_robodojo_ttt_pairs.py`, `test_ttt_meta*.py`) | ✅ 전체 1990 passed / 40 skipped(기존 GPU 테스트) |

**RoboDojo 사실 확인** (공식 데이터셋·RoboDojo 레포·XPolicyLab에서 직접 확인):
- 평가: 42 task × seed {0,1,2}. Generalization 12 / Precision 8 / Long-Horizon 8 / Memory 6 / Open 8. 일반 task는 seed당 앞 50 layout, Generalization은 표준 25 + `<task>_random` 25.
- `_random` = 같은 task + 방해 물체 ~11개 + 방 색·테이블 재질 랜덤.
- 학습 데이터: 35 task × 100 demo (깨끗한 장면). **`_random` demo 없음, Open demo 없음.**
- 공개 `OpenWAM-Alpha-Sim-RoboDojo` ckpt의 `normalization_stats.npy` metadata.tasks = 정확히 그 35개 → **GenRand·Open은 학습 안 됨.**
- train/val 분리 없음 (split을 줘도 전체 episode 사용) → ckpt는 task당 100 demo 전부 학습한 상태.

**안 한 것**: 실가중치 로딩/forward, 실제 meta-training 실행, RoboDojo 시뮬 평가, evolving 루프(2번), deploy에서 TTT 켜기(meta-trained ckpt는 자동으로 켜짐 — 아래).

---

## 2. 사내 시작 순서

### Step 0 — 받기 + 환경

```bash
git clone https://github.com/twoongg/OpenWAM.git && cd OpenWAM && git checkout feat/video-ttt
git remote add upstream https://github.com/OpenWAM-Official/OpenWAM.git
# README 정식 설치 (torch 2.7.1 + cu128, pip install -e .)
pytest tests/test_video_ttt.py tests/test_video_ttt_architecture.py tests/test_video_ttt_real_wan.py \
       tests/test_robodojo_ttt_pairs.py tests/test_ttt_meta.py tests/test_ttt_meta_trainer.py -q
# 기대: 54 passed
```

### Step 1 — 체크포인트 + 데이터 받기

```bash
python scripts/download_assets/download_openwam_checkpoints.py      # OpenWAM_Alpha → *-RoboDojo
python scripts/ttt/check_checkpoint_compat.py assets/openwam_ckpt/openwam_alpha/<ckpt> \
    --filename checkpoint_step_60000.safetensors                     # 기대: RESULT: OK, TTT missing 240
```

데이터는 **Generalization 12 task만 584GB** (전체 35 task는 TB 단위). 공식 다운로더(`download_benchmark_data.py`)는 전체를 받으므로 1-a는 골라 받기:

```python
from huggingface_hub import snapshot_download
GEN = ["arrange_largest_number","fold_clothes","hang_mugs","make_toast","pack_objects_into_box",
       "pour_liquid_into_cup","push_T","sort_nesting_dolls_by_size","stack_blocks","stack_bowls",
       "store_laptop_and_headphones","sweep_blocks"]
snapshot_download("RoboDojo-Benchmark/RoboDojo", repo_type="dataset", local_dir="/data/robodojo_raw",
                  allow_patterns=[f"data/RoboDojo/{t}/arx_x5/data/*.hdf5" for t in GEN])
# → dataset_dir = /data/robodojo_raw/data/RoboDojo   (layout: <task>/arx_x5/data/episode_*.hdf5)
```

### Step 2 — 1-a 학습 (Generalization 12 task, TTT만)

```bash
CKPT=assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-RoboDojo
NPROC_PER_NODE=8 bash scripts/train.sh \
  dataloader=robodojo_ttt_gen \
  dataloader.dataset_dir=/data/robodojo_raw/data/RoboDojo \
  dataloader.normalization_stats_path=$CKPT/normalization_stats.npy \
  model.architecture.ttt.enabled=true \
  training.ttt_meta.enabled=true \
  training.finetune_ckpt_path=$CKPT \
  training.num_epochs=null training.max_steps=<N> training.batch_size=<B> \
  training.output_path=outputs/ttt_gen \
  project.wandb.project=null project.tensorboard.enabled=true
```

**학습 곡선 (TensorBoard — 사내에서 wandb 불가)**: `project.wandb.project=null`로 wandb를 끄고 `project.tensorboard.enabled=true`. 로그는 `<training.output_path>/<시각>/tensorboard/`에 쌓임 (wandb와 같은 태그: `train/loss`, `train/loss_video`, `train/loss_action`, `train/grad_norm`, `train/lr`, `train/ttt_context_gain`, `train/ttt_loss_no_context`).

```bash
tensorboard --logdir outputs/ttt_gen --port 6006          # 여러 run을 한 번에 비교
ssh -L 6006:localhost:6006 <학습서버>                      # 서버에서 돌리면 로컬 브라우저로 http://localhost:6006
```

- **`normalization_stats_path`는 반드시 ckpt 것** (안 주면 디스크에 있는 12 task로 통계를 새로 만들어 action 스케일이 어긋남. placeholder 그대로면 바로 에러).
- 한 step = context 4 window 쓰기(backbone `no_grad`) + target forward/backward (+ `gain_every`마다 context 없는 target forward 1번). 일반 finetune step보다 forward가 ~5배.
- 볼 것: `train/loss` (target loss), **`train/ttt_context_gain`** = (context 없을 때 loss) − (context 쓴 후 loss), 같은 noise로 비교. **> 0 이고 학습하며 커져야** 2번이 의미 있음. ~0이면 TTT가 써넣은 걸 안 씀.
- 처음엔 작은 `max_steps`로 메모리/속도부터 확인 (batch, `ttt.layers`, `ttt_context_windows` 조절).
- 저장된 ckpt의 `config.yaml`엔 `model.architecture.ttt`가 들어가므로 deploy가 TTT를 자동으로 빌드하고 TTT 가중치를 로드함.

### Step 3 — 1-a 평가

- 릴리즈 ckpt vs 학습된 ckpt, XPolicyLab으로 Generalization 12 task × seed 0/1/2 (표준 25 + `_random` 25). 에이전트 없음, TTT 메모리는 비어 있는 상태(쓰기 없음).
- 통과 기준(내 제안): GenStd/GenRand SR이 릴리즈(25.56 / 4.11)와 오차 범위 안.
- TTT가 켜지면 `torch.compile` MoT 루프를 끔 → 추론 속도 확인.

### Step 4 — 1-b (35 task 전체)

`robodojo_ttt_gen.yaml` 복사 → `tasks:` 줄 삭제(=전체) 또는 35개 나열. 데이터 전체 다운로드 필요(용량 확인).

### Step 5 — 2번 evolving (미구현)

- `openwam/deploy/policy.py::WAMPolicy.reset()`은 episode마다 불림 → 여기서 TTT 리셋 금지. task 시작용 `reset_memory()`, TC 종료 후 `write_episode(...)`를 policy/server 프로토콜에 추가.
- 쓰기: 성공 episode를 33프레임 window로 순서대로 `with torch.no_grad(), arch.video_ttt.writing(lr=...)` 안에서 forward.
- 곡선: TTT 없음 / TC마다 리셋 / 누적. GenRand는 seed당 공식 25 TC(layout은 30개 있음), 나머지 차원은 50 TC.
- deploy에서 meta-training 안 한 ckpt로 TTT를 켜려면 `load_from_checkpoint_dir`에 model override 인자 추가 필요 (지금은 ckpt `config.yaml`만 봄).

---

## 3. 주의 (함정)

- **`video_backbone`/`action_backbone`을 `model.freeze:`에 넣지 말 것** — `freeze_modules`가 하위 모듈까지 `no_grad`로 감싸서 Wan `GateModule` residual이 끊김. `training.ttt_meta.enabled=true`면 trainer가 `restrict_training_to_ttt()`를 자동 호출함.
- `training.ttt_meta.enabled`와 `type: robodojo_ttt_pairs`는 **같이** 써야 함 (하나만 쓰면 trainer가 에러).
- context pass는 기본으로 backbone `no_grad` (메모리 절약). meta-gradient는 θ_K/θ_V/W_init까지 가지만, context 쪽 레이어 간 상호작용은 미분 안 함. 정확한 버전은 `training.ttt_meta.context_backbone_grad=true` (메모리 훨씬 큼).
- color jitter(train)가 context window마다 따로 걸림 — 같은 context episode 안에서도 색이 다를 수 있음 (PoC에선 그대로 둠).
- fast-weight state는 `state_dict`에 없음 (`VideoTTTMemory._state`) → 저장/`.to(device)` 대상 아님.
- deploy loader는 component를 초기화 안 된 CPU 텐서로 만든 뒤 복사 → 로딩 시 CPU RAM ~30GB.
- 테스트가 못 잡는 회귀: driver에서 TTT 입력 `z_in`을 detach하는 변경.
- CFG(batch 2배) 상황의 fast weight broadcast는 `generate()` 경로로 검증 안 함.

## 4. 아직 내 결정이 필요한 것

1. 1-a 학습 길이(`max_steps`)·batch·`ttt.layers`(기본 30블록 전부, 194.6M params).
2. `ttt_context_episodes`(기본 1) / `ttt_context_windows`(기본 4) — 2번 evolving은 여러 episode가 쌓이므로 1-b에서 늘려볼지.
3. 쓰기 loss: A) KV 재구성만(현재) / B) test time에 video loss 추가 / C) 논문대로.
4. 2번에서 WAM 단독 성공도 써넣을지, 쓰기 입력의 noise level·action 포함 여부.
5. **inner LR** (`training.ttt_meta.inner_lr`, 기본 0.1): WAM-TTT Table B.1 원문 값(meta 0.1 / test 0.01)을 가져왔지만, 그쪽은 inner loss가 video loss + 0.04×KVM이고 우리는 KVM만이라 그대로 대응 안 됨 → 300~500 step 짧은 run으로 0.01 / 0.1 / 1.0 비교, `ttt_context_gain` 제일 잘 오르는 값.
6. **batch / steps**: OpenWAM Table 10의 RoboDojo SFT는 global batch 256 · 60k step (64 GPU). H100 8장이면 `training.batch_size=16 training.gradient_accumulation_steps=2`(=256), OOM이면 8×4. 60k step은 8장 기준 원래의 ~20배 시간이라 PoC는 2~3k step부터.

---

## 5. FlowDAgger 역계산 측정 (브랜치 `inversion-probe`, main에 안 합침)

질문: **"얼린 체크포인트가 libero-plus demo의 action을 낼 수 있나?"** — 낼 수 있으면 FlowDAgger(모델은 그대로, 시작 노이즈만 골라주는 작은 네트워크 학습)가 통하고, 못 내면 LoRA처럼 가중치를 바꿔야 함. 가중치는 하나도 안 바꿈.

창 하나마다 하는 일 (`openwam/flowdagger/inversion.py`):
1. 서버와 똑같이 생성 (같은 schedule·seed 42·조건) → `base_mse` = 지금 모델 출력 vs demo
2. 생성 과정을 step마다 거꾸로 되감아(논문 Eq.3, step당 고정점 반복 `--fp-iters`, 기본 5) demo action이 나오게 하는 시작 노이즈를 찾음. 비디오+action을 같이 되감음 (논문 App. A.2: action만 되감으면 오차가 바닥에서 안 내려감). 목표 비디오는 모델 자신의 생성 결과, action만 demo로 교체 (§4.2).
3. 찾은 노이즈로 다시 생성 → `recon_mse` = 그 결과 vs demo

```bash
git fetch && git checkout inversion-probe
# libero-plus demo (체크포인트 reader 형식 = LeRobot v3 LIBERO, meta/conversion.json 있는 것)
python scripts/flowdagger/inversion_probe.py --ckpt-dir <OpenWAM-Alpha-Sim-LIBERO> \
  --dataset-dir <libero_plus_lerobot> --num-samples 64 --label plus --out results/inv_plus.json
# 비교 기준: 원래 LIBERO 학습 demo로 한 번 더
python scripts/flowdagger/inversion_probe.py --ckpt-dir <같은 ckpt> \
  --dataset-dir <libero_lerobot> --num-samples 64 --label libero --out results/inv_libero.json
```

읽는 법 (값은 모델의 정규화 action 공간, 끝에 표로 출력 + JSON에 창별 기록):
| 지표 | 뜻 |
|---|---|
| `base_mse` | 지금 모델이 demo와 얼마나 다른가 |
| `seed_best_mse` | 노이즈만 랜덤으로 4번 더 바꿔봤을 때 제일 가까운 것 |
| `recon_ratio` = recon/base | **≪1 (예: <0.1)이면 도달 가능**, 1 근처면 못 감 |
| `noise_action_std`, `…frac_abs_gt3` | 찾은 노이즈가 정상 가우시안처럼 생겼나 (std≈1, 0.0027). 도달했어도 노이즈가 이상하게 크면 "억지로" 간 것 |
| `self_recon_mse` | 모델 자기 출력을 되감았다 다시 생성한 오차 = 이 방법의 바닥 (bf16 포함) |
| `fp_action_update_last` | 마지막 step 고정점 반복이 수렴했나. 안 줄어들면 `--fp-iters 20`으로 다시 |
| per active dim | dim 0-2 xyz, 3-8 rot6d, 9 gripper (LIBERO EEF10) |

- 비용(실측 안 함): 창당 forward 약 110~170번 (10 step 기준: 기본 생성 10 + 추가 seed 40 + 되감기 50 + 재생성 10 + self-check 60). H100 한 장에서 돌리는 걸 가정.
- demo는 체크포인트의 `normalization_stats.npy`로 정규화함 (새 demo로 통계를 다시 만들지 않음). min-max라 범위 밖 값은 [-1,1]로 잘림.
- reader가 데이터 형식을 거부하면(conversion.json 없음 등) 형식이 다른 것 — 이 경우 말해줄 것.
- 테스트: `tests/test_flowdagger_*.py` (작은 실제 Wan+ActionDiT, CPU). 생성 루프가 `generate()`와 비트 단위로 같은지, 되감기가 자기 노이즈를 복원하는지, 고정 첫 프레임·비활성 action dim 유지, 스크립트 전체 배선까지.

**결과 (2026-11-09, 회사):** open-loop라 판정에 못 씀. LoRA closed-loop 결과(교란 view 6/30 → 20/30)와 반대로 나왔음. demo와의 거리는 성공 여부를 예측하지 못함. 이 측정은 회귀 감시용으로만 남김.

## 6. FlowDAgger 상한 closed-loop 측정 — 노이즈 정답지 서버 (브랜치 `inversion-probe`)

질문: **"노이즈를 완벽하게 고르면 교란 view에서 얼마나 성공하나?"** 이 값이 FlowDAgger의 사실상 상한임. 학습은 하지 않고 가중치도 그대로 둠.

seed sweep(5개)에서는 평범한 노이즈로 9/30까지 감. 그런데 거꾸로 계산해서 찾은 노이즈는 평범하지 않았음(std 2.6). 그래서 그 영역의 노이즈로 실제 로봇이 움직이는지 직접 확인함.

매 replan마다 서버가 하는 일 (`openwam/flowdagger/oracle.py`):
1. **선생**: 같은 얼린 모델에 **원래 view**를 보여주고 action chunk를 만듦. stock 성공률은 97~100%.
2. **학생**: 같은 모델에 **교란 view**를 보여줌. 선생 action이 나오도록 시작 노이즈를 거꾸로 계산함(§5와 같은 방식). 목표는 학생 자신의 비디오에 action만 선생 것으로 바꾼 것.
3. 그 노이즈로 학생을 돌려서 나온 **학생의 출력**을 로봇에 보냄. 학생이 못 따라한 부분은 그대로 rollout에 남음.

### driver 쪽에 추가할 것 (회사 Claude 담당)

매 step obs에 원래 카메라 위치에서 렌더한 이미지를 같이 보내면 됨.

```json
{"type": "obs",
 "images":         {"head_camera": <교란 view>, "left_wrist_camera": <wrist>},
 "teacher_images": {"head_camera": <같은 상태, 원래 view>},
 "prompt": "...", "state": [...]}
```
- `teacher_images`에서 빠지거나 null인 카메라는 `images`의 것을 그대로 씀. 카메라 교란이 3인칭 카메라만 옮긴다면 `head_camera` 하나만 보내면 됨.
- 원래 view 렌더는 baseview650 대조군을 만들 때 다시 렌더했던 방식을 그대로 쓰면 됨. 같은 MuJoCo 상태에 원래 카메라 pose를 쓰면 됨.
- PONG에 `oracle_mode`가 실려 오므로 driver에서 모드를 확인할 수 있음.

### 실행 — 같은 시작 위치 30개(env seed 42)에서 모드 3개를 각각 돌림

```bash
git fetch && git checkout inversion-probe
CK=<OpenWAM-Alpha-Sim-LIBERO>
python scripts/flowdagger/oracle_server.py --ckpt-dir $CK --mode teacher --port 8877 --log results/oracle/teacher.jsonl
python scripts/flowdagger/oracle_server.py --ckpt-dir $CK --mode student --port 8877 --log results/oracle/student.jsonl
python scripts/flowdagger/oracle_server.py --ckpt-dir $CK --mode oracle  --port 8877 --log results/oracle/oracle.jsonl
```
나머지 플래그(`--denoise-steps`, `--inference-horizon`, ...)는 stock 서버와 같음. compile과 DiT velocity cache는 항상 꺼짐: 거꾸로 계산하려면 캐시 없는 같은 sampler가 필요하고, 세 모드가 같은 경로를 써야 비교가 됨. `--seed`는 시작 노이즈 seed이고 기본값 42임.

### 읽는 법

| 모드 | 의미 | 예상 |
|---|---|---|
| `teacher` | 배선 확인. 원래 view로 로봇을 움직임 | stock 수준(29~30/30). 낮으면 렌더/배선 문제이므로 oracle 결과를 믿지 말 것 |
| `student` | 같은 경로(eager, 캐시 없음)에서의 기준선 | zeroed eager 6/30 근처 |
| `oracle` | **노이즈를 완벽하게 골랐을 때 = FlowDAgger 상한** | 높으면(예: 20/30 이상) FlowDAgger가 해볼 만함. student 수준이면 이 문제는 노이즈로 못 고침 → LoRA |

`oracle.jsonl`에는 chunk마다 한 줄씩 기록됨. 키는 `episode`, `chunk`, `base_mse`(학생이 그냥 돌렸을 때 선생과의 차이), `recon_mse`(노이즈를 바꾼 뒤 차이), `base/recon_per_dim`, `noise_action`(std, |w|>3 비율), `fp_action_update_last`, `video_drift`, `seconds`.
- oracle 결과가 낮을 때: `recon_mse`가 `base_mse`보다 크게 줄었으면 "선생을 따라하긴 했는데도 실패"한 것임. 이상한 노이즈가 closed-loop에서 안 통한다는 뜻. 안 줄었으면 "따라하지도 못함"이고, 이때는 `--fp-iters 20`으로 다시 확인함.
- 비용(실측 안 함): replan당 forward가 oracle 80번(선생 10 + 학생 10 + 되감기 50 + 재생성 10), teacher 10번, student 10번. oracle은 느리므로 10 trial부터 돌려보기를 권함.
- 테스트: `tests/test_flowdagger_oracle.py` (CPU 작은 모델). 다음을 확인함:
  - student/teacher 모드가 `generate()`와 비트 단위로 같음
  - oracle이 선생 action에 도달함
  - 학생 비디오가 유지됨
  - wrist fallback, reset 시 episode 증가, 로그
  - eager·캐시 꺼짐 강제, async 거부, CLI 배선
