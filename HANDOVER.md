# HANDOVER — video-expert TTT (agent-driven evolving WAM)

- 레포: `https://github.com/twoongg/OpenWAM` (private), 작업 브랜치 **`feat/video-ttt`**
- 베이스: upstream `OpenWAM-Official/OpenWAM` @ `a00c4ba` (2026-11-02)
- 1차 작업: 2026-11-04, 개인 Mac(M4, 16GB, CUDA 없음) — **CPU 테스트와 구조 검증까지만 함. 실가중치·GPU로는 아직 한 번도 안 돌림.**
- 설계·근거·결정 이유 전체: [`docs/video-ttt.md`](docs/video-ttt.md) (이 문서는 "어디서부터 이어갈지"만)

---

## 1. 여기서 한 것

| 항목 | 상태 |
|---|---|
| TTT 메모리 모듈 `openwam/model/ttt/video_ttt.py` (SwiGLU fast weight, KV 재구성 쓰기, 누적/리셋, weight norm, zero-init 출력) | ✅ 구현·테스트 |
| joint loop(`mot_driver.py::step`)와 video-only 경로에 TTT residual hook (checkpoint 구간 바깥) | ✅ |
| config `model.architecture.ttt.*` (`configs/model/dual_system.yaml`, 기본 off) | ✅ Hydra → 아키텍처 전달 확인 |
| TTT만 학습: `arch.restrict_training_to_ttt()` (`freeze:` 쓰면 gradient 끊김 — 아래 주의) | ✅ |
| `video_ttt.*` 없는 체크포인트도 strict load 통과 (`_optional_checkpoint_prefixes`) | ✅ |
| 체크포인트 호환성 체크 스크립트 `scripts/ttt/check_checkpoint_compat.py` | ✅ RoboDojo ckpt로 확인 |
| 테스트 `tests/test_video_ttt.py`(14), `tests/test_video_ttt_architecture.py`(13), `tests/test_video_ttt_real_wan.py`(4) | ✅ 전체 suite 1967 passed / 40 skipped(기존 GPU 테스트) |

**RoboDojo 체크포인트 확인 결과** (`OpenWAM/OpenWAM-Alpha-Sim-RoboDojo`, `checkpoint_step_60000.safetensors`, 헤더만 읽음):
TTT off → missing/unexpected/shape mismatch 모두 0. TTT 30레이어 → missing은 `video_ttt.*` 240개뿐(허용), 나머지 0. TTT 파라미터 194.6M.
※ 내가 가진 체크포인트가 Sim인지 Real(ARX-X5/Piper)인지 확인 안 함 — Real이면 같은 체크 한 번 더.

**안 한 것**: 실가중치 로딩/forward, meta-training(데이터·trainer step), evolving 평가 루프, deploy에서 TTT 켜기.

---

## 2. 사내에서 시작 순서

### Step 0 — 받기 + 환경 (30분)

```bash
git clone https://github.com/twoongg/OpenWAM.git && cd OpenWAM
git checkout feat/video-ttt
git remote add upstream https://github.com/OpenWAM-Official/OpenWAM.git   # upstream 추적용
# README의 정식 설치 (torch 2.7.1 + cu128, pip install -e .)
pytest tests/test_video_ttt.py tests/test_video_ttt_architecture.py tests/test_video_ttt_real_wan.py -q
# 기대: 31 passed
```

### Step 1 — 체크포인트 받고 호환성 체크 (GPU 불필요)

```bash
python scripts/download_assets/download_openwam_checkpoints.py      # OpenWAM_Alpha → *-RoboDojo (내가 쓰는 것)
python scripts/ttt/check_checkpoint_compat.py assets/openwam_ckpt/openwam_alpha/<ckpt_name> \
    --filename checkpoint_step_60000.safetensors
# 기대: "RESULT: OK", missing (TTT, expected): 240
```

### Step 2 — 실가중치 smoke test (GPU 1장) ← **첫 번째로 새로 짤 코드**

목표: 실제 OpenWAM-α에서 (a) TTT on/off 출력 bit 동일, (b) TTT 활성 + 쓰기 후 출력 변화.

1. **deploy에서 TTT 켜는 방법이 아직 없음.** `openwam/deploy/model_loader.py::load_from_checkpoint_dir`는 ckpt 안의 `config.yaml`만 보고 아키텍처를 만듦 (deploy override는 빌드 *후* 적용).
   - 해결: `load_from_checkpoint_dir(..., model_overrides=...)` 같은 인자 추가해서 `model.architecture.ttt`를 빌드 전에 합치기. (임시로는 ckpt dir 복사본의 config.yaml에 `ttt:` 추가 — `check_checkpoint_compat.py::_staged_ckpt_dir`가 그 방식)
   - meta-training으로 저장된 ckpt는 config에 `ttt`가 들어가므로 이 문제 없음 (아래 Step 3 참고).
2. RoboDojo 샘플 1개로 `arch.compute_loss` 또는 `arch(...)` 를 TTT on/off로 돌려 비교 → `o_proj`를 작은 랜덤값으로 바꾸고 `with torch.no_grad(), arch.video_ttt.writing(lr=0.01): ...` 후 출력이 바뀌는지.
3. bf16에서 fast-weight 경로(fp32 계산) 수치 확인, 메모리 사용량 기록 (TTT 30레이어 기준 +194.6M params).

### Step 3 — meta-training (TTT slow weight만 학습)

- **config로 TTT 켜기는 이미 됨**: finetune 경로(`train/utils/ckpt_model_loader.py::build_architecture_from_ckpt_dir`)는 live `cfg.model`을 ckpt config 위에 merge(새 키 추가 허용)하고 `architecture.load_checkpoint`를 씀 → `model.architecture.ttt.enabled=true` + `training.finetune_ckpt_path=<RoboDojo ckpt>`로 TTT 붙은 모델 빌드·로드 가능 (코드상 확인, 실행은 안 해봄).
- 짜야 할 것:
  1. **trainer 플래그**: `openwam/train/openwam_trainer.py`의 freeze 블록(`freeze_modules` 호출 직후) 에 `if ttt.train_only: architecture.restrict_training_to_ttt()`.
  2. **episode 쌍 dataloader**: `openwam/dataloader/robodojo.py` 기반. 같은 task의 (context episode들 → target window). 규칙은 `docs/video-ttt.md` §7-2 (장면 공유 쌍, 체크포인트가 안 본 task 섞기, context 1~K개 순서대로).
  3. **train step**: `video_ttt.reset()` → `with video_ttt.writing(lr=η, create_graph=True): compute_loss(context)` (loss 버림) → `compute_loss(target)["loss"].backward()`. 사용 예시는 `docs/video-ttt.md` §3.
  4. 쓰기 입력 노이즈 레벨(`min/max_timestep_boundary`) / action 포함 여부 결정.

### Step 4 — evolving 평가 루프

- `openwam/deploy/policy.py::WAMPolicy.reset()`은 **episode마다** 불림 → 여기서 TTT를 리셋하면 안 됨. task 시작용 `reset_memory()`와 TC 종료 후 `write_episode(frames, actions)` 를 별도로 추가 (server 프로토콜에도).
- RoboDojo 평가는 XPolicyLab 경유 (`benchmarks/robodojo/README.md`).
- 곡선 3종 로깅: TTT 없음 / TC마다 리셋 / 누적. x=TC(1~50), y=score.

---

## 3. 주의 (함정)

- **meta-training 때 `video_backbone` / `action_backbone`을 `model.freeze:` 목록에 넣지 말 것.** `freeze_modules`는 하위 모듈 forward까지 `no_grad`로 감싸는데, Wan의 residual 덧셈이 `GateModule`(nn.Module) 안에 있어서 residual stream 전체가 끊김 → 앞쪽 TTT 레이어 gradient 0. 반드시 `restrict_training_to_ttt()` 사용. (실제 Wan 블록 테스트로 확인)
- **fast-weight state는 `state_dict`에 없음** (`VideoTTTMemory._state`). 체크포인트 저장/`.to(device)` 대상 아님 → device 이동 후 `reset()`, 프로세스 간 유지 필요하면 별도 저장 구현.
- TTT가 켜지면 MoT 루프 `torch.compile`을 끔 → deploy 속도 손해 가능.
- deploy loader는 component를 초기화 안 된 CPU 텐서로 만든 뒤 복사 → 로딩 시 CPU RAM ~30GB 필요.
- 테스트가 못 잡는 회귀 1개: driver에서 TTT 입력 `z_in`을 detach하는 변경.
- CFG(cond/uncond로 batch 2배) 상황에서 batch-1 fast weight broadcast는 코드상 되지만 `generate()` 경로로 검증 안 함.

## 4. 아직 내 결정이 필요한 것

1. 쓰기 loss: **A) KV 재구성만(현재)** / B) test time에만 video loss 추가 / C) 논문대로 — 현재 권장 A로 곡선 먼저 (`docs/video-ttt.md` §4-1).
2. TTT 크기·레이어: 현재 heads 8 × dim 64, hidden 128, 전 30블록 (임의값). WAM-TTT Appendix B(미확인) 값과 대조.
3. WAM이 혼자 성공한 TC도 써넣을지 (현재 계획: 씀, ablation으로 비교).
4. 쓰기 입력 노이즈 레벨과 action 포함 여부.
5. WAM-TTT 하이퍼파라미터(N, η, λ, d, f_h) PDF로 확인.

## 5. 브랜치 변경 파일

```
openwam/model/ttt/{__init__,video_ttt}.py                 (new)
openwam/model/architectures/dual_system/mot_driver.py     (ttt hook)
openwam/model/architectures/dual_system/joint_self_attn.py (cfg.ttt, video-only hook, restrict_training_to_ttt)
openwam/model/architectures/base.py                       (_optional_checkpoint_prefixes)
configs/model/dual_system.yaml                            (ttt block, default off)
scripts/ttt/check_checkpoint_compat.py                    (new)
tests/test_video_ttt*.py                                  (new, 31 tests)
docs/video-ttt.md, HANDOVER.md                            (new)
```
