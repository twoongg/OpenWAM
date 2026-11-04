# Video-expert TTT for OpenWAM-α — 설계 · 구현 현황 · 남은 작업

작성: 2026-11-04 (개인 Mac에서 1차 구현·검증). 브랜치 `feat/video-ttt`.
사내 GPU 환경에서 이어서 구현하기 위한 인수인계 문서.

---

## 1. 연구 그림 (agent-driven evolving WAM)

- 베이스: RoboDojo로 finetune된 OpenWAM-α 체크포인트 (Wan2.2-TI2V-5B video DiT + 1B ActionDiT, `dual_system / joint_self_attn`, mask `mutual`). **가중치는 고정.**
- 에이전트 없이 실패하는 task가 있음 (트레이닝셋 task + unseen task 모두).
- **Evolving 루프 (task마다 fast weight 리셋, TC 1→50 순서대로):**
  1. WAM이 현재 fast weight로 단독 시도 → 이 결과가 y축 score
  2. 실패하면 에이전트가 **실시간 개입**해서 성공시킴
  3. 성공한 관측을 video expert의 TTT fast weight에 써넣음 (WAM 자체 성공도 씀 — 내 결정, ablation 대상)
  4. 다음 TC
- 결과 그림: x축 TC 번호, y축 score. 비교 곡선: TTT 없음 / TC마다 리셋 / 누적(제안). 누적만 올라가야 "evolving" 주장이 섬.
- 대규모 학습은 안 함. TTT slow weight만 학습하는 meta-training은 "벤치마크 트레이닝셋 finetune" 규모로 허용.

## 2. 배경: 왜 이 구조인가

| 출처 | 가져온 것 |
|---|---|
| **WAM-TTT** (arXiv 2607.06988, v3, 프로젝트 페이지 wam-ttt.github.io, 코드 미공개) | TTT residual을 **video expert에만** 추가 (Eq. 1–2), action expert는 joint attention으로 간접 영향. 쓰기 = key-value 재구성(KVM) + video generation loss의 inner SGD. meta-training은 WAM + θ_{Q,K,V,O} + W_init 학습. 본문에 "Spatial-TTT-style fast-weight branches" 라고 명시 |
| **Spatial-TTT** (THU-SI/Spatial-TTT) | SwiGLU fast weight, zero-init 출력, 스트리밍으로 fast-weight state를 들고 다니는 구조 (evolving 누적과 대응). 단 Qwen3-VL(causal LLM)용이고 attention을 *교체*함 |
| **LaCT** (a1600012888/LaCT) | Spatial-TTT 커널의 원본. weight norm(업데이트 후 row norm을 초기값으로 되돌림), Wan2.1 기반 AR video 코드 |

WAM-TTT에서 **verbatim으로 확인 못 한 값** (Appendix B/Table B.1, 요약본에서만 봄 — 사용 전 PDF 확인 필요):
N=1 inner step, η=0.1(meta)/0.01(test), λ=1e-2, TTT head dim d=48, hidden f_h=128, 적용 블록 수, optimizer 설정.

WAM-TTT ablation (v1 Table 2, verbatim): full 100.0/88.9, w/o TTT 40.0/74.1, **w/o meta-training 9.0/0.0**, test-time LoRA 30.0/0.0, w/o memory recon 66.7/72.0 (Table Bussing / Swap Place).
→ meta-training 없이 TTT 레이어만 붙이면 오히려 나빠짐. 그래서 TTT slow weight 학습 단계가 필수.

## 3. 현재 구현

### 수식 (레이어 l)

```
읽기:  z^(l+1) = block_l(z^(l)) + Δz,   Δz = θ_O · RMSNorm_head( f_W( l2norm(θ_Q(RMSNorm(z^(l)))) ) )
       f_W(x) = W1 · (silu(W0 · x) ⊙ (W2 · x))          # per-head SwiGLU fast weight
쓰기:  K = l2norm(θ_K(RMSNorm(h))),  V = θ_V(RMSNorm(h))
       W ← W − η ∇_W mean‖f_W(K) − V‖²                  # KVM loss, 레이어 로컬
       (weight_norm=True면 W의 각 row norm을 W_init row norm으로 재조정 — LaCT 방식)
```

- θ_O는 **0으로 초기화** → TTT를 켜도 사전학습 모델 출력이 bit 단위로 동일.
- fast weight는 sample마다 따로 (B, H, r, c), fp32로 계산.
- fast weight state는 `state_dict`에 들어가지 않고 `reset()` 전까지 유지 → TC 간 누적.

### 파일

| 파일 | 내용 |
|---|---|
| `openwam/model/ttt/video_ttt.py` | `VideoTTTBranch`(slow weight + read/update), `VideoTTTMemory`(레이어 묶음, state, `writing()` 컨텍스트, `reset()`), `add_ttt_residual()` |
| `openwam/model/architectures/dual_system/mot_driver.py` | `DualSystemMoTDriver(ttt=...)`. `step()`에서 블록 실행 **후** residual 추가 — gradient checkpoint 구간 *바깥* (backward recompute 때 쓰기가 재실행되지 않도록) |
| `openwam/model/architectures/dual_system/joint_self_attn.py` | `cfg.ttt`로 `self.video_ttt` 생성, video-only 경로(`noisy_actions=None`)에도 hook, `restrict_training_to_ttt()`, TTT 켜지면 `torch.compile` 루프 비활성 |
| `openwam/model/architectures/base.py` | `_optional_checkpoint_prefixes()` — strict load에서 빠져도 되는 prefix (`video_ttt.`) |
| `configs/model/dual_system.yaml` | `model.architecture.ttt` (기본 `enabled: false`) |
| `scripts/ttt/check_checkpoint_compat.py` | 가중치 없이 safetensors 헤더만으로 deploy loader 전 과정을 meta device에서 실행해 키/shape 호환성 확인 |
| `tests/test_video_ttt*.py` | 단위(14) + mock 아키텍처(13) + 실제 Wan 블록 코드(4) |

### 사용법

```python
arch.video_ttt.reset()                                   # task 시작
with torch.no_grad(), arch.video_ttt.writing(lr=0.01):   # 성공 episode 써넣기 (test time)
    arch.compute_loss(**success_window_inputs)           # 또는 arch(...) forward 한 번
# 이후 평소처럼 rollout — 써넣은 fast weight가 읽힘

# meta-training (TTT slow weight만 학습)
arch.restrict_training_to_ttt()
arch.video_ttt.reset()
with arch.video_ttt.writing(lr=0.1, create_graph=True):
    arch.compute_loss(**context_inputs)                  # context episode 쓰기 (loss는 버림)
loss = arch.compute_loss(**target_inputs)["loss"]         # target episode
loss.backward()
```

config: `model.architecture.ttt.enabled=true model.architecture.ttt.layers=[0,10,20]` 등.

## 4. 설계 결정 로그 (내 결정 — 바꿔도 됨)

1. **쓰기는 KVM loss만 (WAM-TTT의 video generation loss 미포함).**
   - KVM은 레이어 로컬이라 쓰기가 forward 한 번에 끝나고, meta-training도 일반 backprop + 작은 fast-weight 네트워크 안의 2차 미분만 필요.
   - video loss를 inner loop에 넣으면 쓰기마다 6B 전체 backward, meta-training은 전체 네트워크 2차 미분 → "벤치마크 finetune 규모" 초과 가능성 큼, 실시간 TC 간 지연도 커짐.
   - KVM이 행동 성공과 직접 연결되지 않는 약점은 meta-training이 θ_K/θ_V를 학습해서 메움 (LaCT/Spatial-TTT의 전제).
   - **근거 공백**: WAM-TTT는 KVM을 뺀 ablation만 있고 "KVM만" ablation은 없음 → 손해 크기는 실험으로 확인해야 함.
   - 대안: B) meta-training은 KVM, test time에만 video loss 추가 / C) 논문대로 (first-order 근사 고려). 권장: A로 evolving 곡선 확인 → 안 오르면 B를 ablation.
2. **읽고 나서 쓰기** (한 forward 안에서 현재 W로 residual 계산 → 같은 입력으로 W 갱신).
3. **기본 크기**: heads 8 × head_dim 64, hidden 128, 30개 전 블록 → TTT 파라미터 **194.6M** (블록당 ~6.5M, 대부분 θ_{Q,K,V,O}). WAM-TTT 값(d=48)은 미확인이라 임의로 정함.
4. **`freeze:` 대신 `restrict_training_to_ttt()`**: OpenWAM의 `freeze_modules`는 고정 모듈의 forward를 하위 모듈까지 전부 `no_grad`로 감쌈. Wan은 residual 덧셈을 `GateModule`(nn.Module) 안에서 하므로 그대로 쓰면 residual stream이 끊겨 앞쪽 TTT 레이어에 gradient가 안 옴 (실제 Wan 블록 테스트로 확인). → `requires_grad`만 끄는 경로를 따로 둠. **meta-training 설정에서 `video_backbone`/`action_backbone`을 `freeze:` 목록에 넣으면 안 됨.**
5. **TTT 켜지면 `torch.compile` 루프 끔**: fast weight state가 호출마다 바뀌고 쓰기 때 autograd가 돌아서. deploy 속도 손해 가능.
6. **weight norm 기본 on**: 50번 연속 쓰기에서 fast weight 크기 폭주 방지 (테스트로 row norm 유지 확인).
7. **meta-training 쌍 = 같은 task, 같은 조건(표준 장면), 다른 episode** (`robodojo_ttt_pairs`). GenRand(`_random`) demo가 공식 데이터에 없고, test time의 context도 "같은 조건에서의 성공"이라서. gen→genrand 쌍은 데이터가 없고 학습 목표도 test time과 다름 (깨끗한 demo를 context로 쓰는 건 test-time baseline으로만).
8. **context pass는 backbone `no_grad`가 기본**: context window마다 6B backbone activation을 backward까지 들고 있지 않도록. TTT 쓰기(`update`)는 `enable_grad` 안에서 그래프를 만듦 → θ_K/θ_V/W_init까지 meta-gradient 도달(테스트로 확인). 대신 context 쪽 레이어 간 상호작용은 미분 안 함 (`context_backbone_grad=true`로 정확한 버전).
9. **context window**: context episode마다 33프레임 window 4개를 처음~끝 균등 간격, 시간 순서로 (`ttt_context_windows`). test time엔 성공 episode 전체를 window 단위로 쓰게 되므로 학습/테스트 window 수가 다름 — 1-b에서 조정 대상.
10. **`ttt_context_gain` 로깅**: 같은 noise/timestep으로 context 있을 때·없을 때 target loss 차이. TTT가 "고정 adapter"가 됐는지(≈0) "써넣은 걸 쓰는지"(>0) 구분용.

## 5. 검증 현황 (Mac M4 16GB, CPU)

- 전체 테스트: **1967 passed, 40 skipped** (기준선 1936 passed + 새 테스트 31개; skip은 전부 기존 GPU/체크포인트 필요 테스트).
- 확인된 성질: zero-init 시 출력 동일(mock + 실제 Wan 블록) / 쓰기가 KVM loss를 줄임 / 누적·리셋 / sample별 독립 / 50회 쓰기 후 row norm 유지 / `no_grad` 안에서 쓰기 / meta-gradient가 쓰기를 거쳐 θ_K·θ_V·W_init에 도달 / gradient checkpointing 유무로 TTT gradient 동일 / TTT-only 학습 시 frozen 실제 Wan 블록을 지나 첫 TTT 레이어까지 gradient 도달.
- mutation 검사: 핵심 줄을 일부러 망가뜨린 9가지 중 8가지를 테스트가 잡음. **못 잡는 것 1개**: driver에서 TTT 입력 `z_in`을 detach하는 경우(일부 gradient 성분만 사라짐).
- **RoboDojo 체크포인트 호환성** (`OpenWAM/OpenWAM-Alpha-Sim-RoboDojo`, `checkpoint_step_60000.safetensors`, 2089 tensors, 12.41B params, 전부 BF16):
  - TTT 없음: missing 0 / unexpected 0 / shape mismatch 0
  - TTT 30 레이어: missing은 `video_ttt.*` 240개뿐(허용됨), 나머지 0, strict load 통과
  - TTT 키 허용 로직을 되돌리면 strict load 실패 → 체크가 실제로 회귀를 잡음
  - 실제 가중치 값 로딩·forward는 **안 함** (아래 6절).
- 발견: deploy loader는 component를 meta가 아닌 **초기화 안 된 CPU 텐서**로 만들고 거기에 복사함 → 실제 로딩 시 RAM ~30GB 필요.

```bash
python scripts/ttt/check_checkpoint_compat.py <ckpt_dir> \
  --hf-repo OpenWAM/OpenWAM-Alpha-Sim-RoboDojo --filename checkpoint_step_60000.safetensors
```

## 6. 개인 Mac(M4, 16GB RAM, 디스크 여유 ~27GB)에서 가능한 범위

| 작업 | 가능? | 비고 |
|---|---|---|
| TTT 모듈/훅/로더 코드 수정 + CPU 단위·통합 테스트 | ✅ | 전체 suite 약 1.5분 |
| 실제 Wan 블록 코드로 동작 검증 (작은 random 가중치) | ✅ | `tests/test_video_ttt_real_wan.py` |
| 체크포인트 키/shape 호환성 | ✅ | 헤더만 원격으로 읽음, RAM 0.8GB |
| meta-training dataloader 로직 (episode 쌍 구성) | ✅ (로직만) | RoboDojo 데이터 일부만 받아서 / 가짜 데이터로 단위 테스트 |
| meta-training step / evolving 루프 로직 | ✅ (mock·tiny Wan으로) | 실제 규모 아님 |
| 실제 체크포인트 전체 다운로드 | ⚠️ | 24.8GB 단일 파일, 디스크가 거의 꽉 참 |
| 실제 가중치 로딩 | ❌ | ~30GB RAM 필요. 시도 시 swap 15GB까지 차서 중단함 |
| 실제 6B forward / zero-init 동일성 확인 (실가중치) | ❌ | 위와 동일 |
| meta-training 실행 | ❌ | GPU 필요 (OpenWAM 권장 80GB × 8) |
| RoboDojo 시뮬 평가 / 실시간 에이전트 루프 | ❌ | GPU + XPolicyLab 환경 |

## 7. 사내에서 이어갈 작업 (순서대로)

> 2026-11-04 추가: 아래 2·3번(쌍 dataloader, trainer 통합)은 구현됨 — 실행법은 `HANDOVER.md` Step 2. RoboDojo 구조 확인 결과도 HANDOVER §1.

1. **실가중치 smoke test (GPU 1장)**: `load_from_checkpoint_dir` + `ttt.enabled=true`로 로드 → 같은 입력에서 TTT on/off 출력이 bit 동일한지 확인 → `o_proj`를 작은 랜덤값으로 바꾸고 쓰기 전후 출력이 달라지는지 확인. bf16에서 fast weight fp32 경로 확인.
2. **meta-training 데이터**: RoboDojo 트레이닝셋에서 같은 task의 episode 쌍 (context → target).
   - TTT branch가 "쓸모없음"으로 학습되지 않도록, context를 봐야만 알 수 있는 정보를 공유하는 쌍으로 구성 (같은 장면 배치/변형 등).
   - unseen task 대비: 체크포인트가 **안 본** task 데모(다른 suite/embodiment)를 섞어야 "모르는 task를 context로 배우는" 상황을 연습. 단 test-time unseen task와 겹치면 안 됨.
   - evolving 대비: context episode를 1~K개 **순서대로** 써넣은 뒤 target을 맞히게 (50번 누적 상황을 학습 때 연습).
   - 33프레임(9 sampled frames) 윈도우 단위로 episode를 잘라 순서대로 쓰기.
3. **trainer 통합**: `model.architecture.ttt.train_only` 같은 플래그 → `freeze_modules` 이후 `restrict_training_to_ttt()` 호출. step = `reset` → context 쓰기(`create_graph=True`) → target `compute_loss` → backward. `finetune_ckpt_path`로 RoboDojo 체크포인트 로드 시 TTT 키 누락 허용되는지 trainer 경로(`train/utils/ckpt_model_loader.py`)도 확인.
4. **쓰기 입력 결정**: context를 어떤 noise level로 넣을지 (`min/max_timestep_boundary`로 저노이즈 제한 권장), action 포함 여부 (`mutual` mask라 video가 action을 봄 → 에이전트 episode의 실제 action을 넣는 쪽이 학습 분포에 가까움).
5. **evolving 평가 루프**: deploy/policy server에 task 시작 시 `reset`, TC 종료 후 성공 episode 쓰기 API 추가. 곡선 3종(없음/리셋/누적) 로깅. fast weight state가 `state_dict`에 없으므로 프로세스 간 유지가 필요하면 별도 저장 추가.
6. WAM-TTT Appendix B 하이퍼파라미터 PDF로 확인 후 기본값 조정.
7. (필요 시) 옵션 B/C: inner loop에 video generation loss 추가.

## 8. 환경 재현

```bash
git checkout feat/video-ttt
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python torch==2.7.1 torchvision==0.22.1   # GPU 서버는 CUDA 빌드로
uv pip install --python .venv/bin/python -e .        # 사내 GPU 환경은 README의 정식 설치 절차
.venv/bin/python -m pytest tests/test_video_ttt.py tests/test_video_ttt_architecture.py tests/test_video_ttt_real_wan.py -q
```

## 참고

- WAM-TTT: https://arxiv.org/abs/2607.06988 (v3 HTML: https://arxiv.org/html/2607.06988v3)
- Spatial-TTT: https://github.com/THU-SI/Spatial-TTT (arXiv 2603.12255)
- LaCT: https://github.com/a1600012888/LaCT (arXiv 2505.23884)
- OpenWAM: https://github.com/OpenWAM-Official/OpenWAM (arXiv 2609.07398), RoboDojo 체크포인트 https://huggingface.co/OpenWAM/OpenWAM-Alpha-Sim-RoboDojo
