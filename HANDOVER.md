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
  project.output_dir=<out>
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
5. WAM-TTT Appendix B 하이퍼파라미터(N, η, λ, d, f_h) PDF로 확인.
