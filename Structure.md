# vjepa2 Repository Structure

## Top-level Layout

```
vjepa2/
├── app/                    # 학습 entry points (3가지 variant)
│   ├── vjepa/              # V-JEPA 2 기본 pretraining (video masking, unconditioned)
│   ├── vjepa_2_1/          # V-JEPA 2.1 (distillation 기반 개선 버전)
│   └── vjepa_droid/        # Action-conditioned pretraining on DROID robot dataset
├── configs/
│   ├── train/              # pretraining yaml (vitg16, vith16, vitl16)
│   ├── train_2_1/          # vjepa2.1 pretraining yaml
│   └── eval/               # downstream eval yaml (k400, ssv2, ek100, in1k 등)
├── evals/                  # downstream eval scripts (frozen probing)
│   ├── video_classification_frozen/
│   ├── image_classification_frozen/
│   └── action_anticipation_frozen/
├── notebooks/              # ⭐ MPC/CEM planning demo (robot manipulation)
│   ├── energy_landscape_example.ipynb
│   └── utils/
│       ├── mpc_utils.py         # CEM planner 구현
│       └── world_model_wrapper.py
├── src/                    # 핵심 라이브러리
│   ├── models/             # encoder, predictor, ac_predictor, attentive_pooler
│   ├── datasets/           # video dataset, dataloader, augmentation
│   ├── masks/              # multiblock3D masking collator
│   ├── hub/                # torch.hub interface, backbone registry
│   └── utils/              # logging, distributed, schedulers, checkpoint
└── tests/                  # unit tests (models, datasets)
```

---

## Core Components

### 1. 모델 (Encoder + Predictor)

**Encoder**: `src/models/vision_transformer.py` — `VisionTransformer` class
- ViT 기반; 2D(이미지) / 3D(비디오, tubelet patch embed) 지원
- `out_layers` 인자로 multi-layer feature 추출 가능
- `use_rope`, `use_sdpa`, `use_silu`, `wide_silu`, `uniform_power` 등 구성 옵션 풍부

**Predictor (unconditioned)**: `src/models/predictor.py` — `VisionTransformerPredictor`
- V-JEPA 2 pretraining용; context token과 mask token을 입력받아 target token 예측
- `use_rope`, `use_mask_tokens`, `chop_last_n_tokens`, `return_all_tokens` 지원
- forward signature: `(x, masks_x, masks_y)` — 마스킹 기반 예측

**Action-Conditioned Predictor**: `src/models/ac_predictor.py` — `VisionTransformerPredictorAC`
- V-JEPA 2-AC (DROID fine-tuning용)
- 각 프레임 토큰 앞에 action token + state token (선택적으로 extrinsics token)을 인터리빙
- forward signature: `(x, actions, states, extrinsics=None)`
- `is_frame_causal=True` → causal attention mask로 프레임 간 인과관계 강제
- 7-DoF 로봇 action (`action_embed_dim=7`)

**Attentive Pooler**: `src/models/attentive_pooler.py` — `AttentivePooler`, `AttentiveClassifier`
- downstream classification probe 전용; CrossAttention으로 고정된 encoder repr 요약

**Hub Backbones**: `src/hub/backbones.py`
- `ARCH_NAME_MAP`: vit_large, vit_huge, vit_giant, vit_ac_giant, vjepa2.1 변종들
- `torch.hub.load("facebookresearch/vjepa2", "vjepa2_ac_vit_giant")` 인터페이스

---

### 2. 학습 (Training)

**V-JEPA 2 Pretraining**: `app/vjepa/train.py`
- **손실함수**: JEPA prediction loss — `|z - h|^loss_exp / loss_exp` (기본 L1, `loss_exp=1.0`)
  - `z`: context encoder + predictor의 예측, `h`: target encoder의 teacher signal (layer-norm 후)
- **EMA target encoder**: momentum schedule로 online encoder → target encoder 업데이트
- **마스킹**: `MaskCollator` (multiblock3D) — spatial/temporal 영역을 블록 단위로 마스킹
- **옵티마이저**: Adam, cosine LR schedule + warmup + weight decay schedule
- **Mixed precision**: bfloat16 기본
- Encoder + Predictor 모두 학습; target encoder는 grad 없음

**V-JEPA 2-AC (DROID) Pretraining**: `app/vjepa_droid/train.py`
- 사전학습된 V-JEPA 2 encoder에서 초기화 후 action-conditioned predictor와 함께 fine-tuning
- 데이터: `clips [B,C,T,H,W]`, `actions [B,T-1,7]`, `states [B,T,7]`, `extrinsics [B,T,6]`
- 손실: teacher-forcing 1-step loss (`jloss`) + autoregressive rollout loss (`sloss`)
  - `auto_steps`번의 AR rollout으로 장기 예측 능력 강화
- EMA 없음 (target encoder는 frozen V-JEPA 2 encoder)

**V-JEPA 2.1 Pretraining**: `app/vjepa_2_1/train.py`
- `compute_mask_distance` 기반 mask-aware distillation 추가
- `Lambda_LinearWarmupHold` 스케줄러 사용 (vjepa2와 다른 학습률 전략)

---

### 3. 데이터 (Datasets)

**VideoDataset**: `src/datasets/video_dataset.py`
- CSV 경로 목록 기반 비디오 로딩 (`decord` VideoReader 사용)
- 다중 데이터셋 혼합 (`datasets_weights`), 다양한 `frames_per_clip` 동시 지원

**ImageNet1K**: `src/datasets/imagenet1k.py` — 이미지 분류 eval용

**MaskCollator**: `src/masks/multiseq_multiblock3d.py`
- pretraining 전용; 시공간 multiblock 마스크를 배치 단위로 생성
- `aspect_ratio`, `spatial_scale`, `temporal_scale`, `num_blocks` 등으로 구성

**Augmentation**: `src/datasets/utils/video/transforms.py`, `transforms_builder.py`
- `RandomResizedCrop`, `RandomHorizontalFlip`, `RandAugment`, `RandomErasing` 지원
- 각 train script의 `make_transforms()`로 조립

**DROID Robot Dataset**: `app/vjepa_droid/droid.py`
- 로봇 trajectory (비디오 + action + state + extrinsics) 로딩; CSV 경로 기반

---

### 4. 평가 (Evaluation)

모든 eval은 encoder frozen (linear/attentive probing) 방식.

| Eval 종류 | 위치 | 대상 벤치마크 |
|---|---|---|
| Video classification | `evals/video_classification_frozen/` | K400, SSv2, Diving48, COIN, Jester |
| Image classification | `evals/image_classification_frozen/` | ImageNet-1K |
| Action anticipation | `evals/action_anticipation_frozen/` | EpicKitchens-100 |

- `evals/hub/` — `torch.hub` 기반 모델 로드 헬퍼
- `evals/scaffold.py`, `evals/main.py` — eval entry point

---

### 5. Planning

**결론: Planning 코드 존재함. 단, notebook demo 수준이며 production training loop에는 통합되어 있지 않음.**

| 파일 | 내용 |
|---|---|
| `notebooks/utils/mpc_utils.py` | **CEM (Cross-Entropy Method)** 전체 구현 — `cem()` 함수, `compute_new_pose()`, `poses_to_diff()` |
| `notebooks/utils/world_model_wrapper.py` | `WorldModel` class — encoder로 이미지를 latent로 인코딩, `infer_next_action()`이 CEM 호출 |
| `notebooks/energy_landscape_example.ipynb` | CEM으로 robot action 최적화 데모; energy landscape 시각화 포함 |

**CEM 구현 요약**:
- `cem(context_frame, context_pose, goal_frame, world_model, ...)`
- 입력: 현재 latent frame `[B=1,T=1,HW,D]` + goal latent frame + world model callable
- 7-DoF 로봇 action (xyz + euler + gripper) 분포를 반복적으로 update
- `samples`개 action trajectory를 샘플링 → predictor로 rollout → goal과 L1 거리 계산 → topk 선택 → 분포 update
- `momentum_mean`, `momentum_std`로 분포 update 시 이전 분포와 혼합
- **world model callable**: `VisionTransformerPredictorAC.forward(reps, actions, poses)` 래핑

**제한**:
- CEM planning은 학습에 사용되지 않음 — inference-only demo
- Hierarchical planning(high-level / low-level 구분) 없음
- Open-loop planning; replanning 루프는 데모 코드에만 암시적으로 존재

---

### 6. Configs

**Pretrain configs** (`configs/train/`, `configs/train_2_1/`):
- `app`: 어떤 train script를 실행할지 지정 (`vjepa`, `vjepa_2_1`, `vjepa_droid`)
- 주요 hyperparameter: `model_name`, `pred_depth`, `pred_embed_dim`, `crop_size`, `tubelet_size`, `loss_exp`, `ema`, `epochs`, `lr`, `warmup`
- `use_rope: true`, `use_mask_tokens: true` (vitg 기본)
- DROID config (`configs/train/vitg16/droid-256px-8f.yaml`): `auto_steps`, `normalize_reps`, `pred_is_frame_causal`, `pretrain_checkpoint` 추가

**Eval configs** (`configs/eval/`, `configs/eval_2_1/`):
- 벤치마크별 (k400, ssv2, in1k, ek100, diving48, coin, jester) × 모델 크기별 구성

---

## HWM_PLDM과의 비교 / 확장 가능성

| 항목 | V-JEPA 2 | HWM_PLDM |
|---|---|---|
| Encoder | ViT (frozen after pretrain) | CNN-based (frozen) |
| Predictor | ViT (mask-based / action-conditioned) | Conv / RNN, bilevel (L1+L2) |
| Action conditioning | 있음 (DROID fine-tuning) | 없음 (observation-only WM) |
| Planning | CEM (inference-only demo) | PLDM-style latent diffusion planning |
| Hierarchy | 없음 | 있음 (L1 abstract goal + L2 step predictor) |
| Downstream | Video/image classification, action anticipation | Robot manipulation (block push 등) |

**확장 포인트 (HWM idea를 얹으려면)**:
1. `VisionTransformerPredictorAC`를 두 레벨로 분리 — high-level predictor가 abstract goal latent를 생성하고, low-level predictor가 step-by-step action token을 생성하는 계층 구조를 추가해야 함.
2. `notebooks/utils/mpc_utils.py`의 CEM은 현재 single-level이며 open-loop임 — hierarchical CEM (high-level goal sampling → low-level action refinement)으로 확장하거나, PLDM 방식의 diffusion-based planning으로 대체 가능.
3. DROID train script(`app/vjepa_droid/train.py`)의 AR rollout loss (`sloss`)는 이미 multi-step consistency를 장려하므로, HWM의 L2 consistency loss와 역할이 유사함 — 이 부분을 HWM bilevel loss로 교체하는 것이 가장 natural한 진입점.

---

## 주요 파일별 한줄 설명 (Index)

| 파일 | 역할 |
|---|---|
| `app/vjepa/train.py` | V-JEPA 2 pretraining loop (unconditioned, JEPA loss + EMA) |
| `app/vjepa_droid/train.py` | Action-conditioned fine-tuning on DROID robot data |
| `app/vjepa_2_1/train.py` | V-JEPA 2.1 pretraining (distillation 기반 개선) |
| `src/models/vision_transformer.py` | ViT encoder (2D/3D, tubelet embed, RoPE 지원) |
| `src/models/predictor.py` | Unconditioned ViT predictor (mask-based target prediction) |
| `src/models/ac_predictor.py` | Action-conditioned predictor (action/state 토큰 인터리빙, causal attn) |
| `src/models/attentive_pooler.py` | CrossAttention 기반 pooler (eval probe용) |
| `src/hub/backbones.py` | torch.hub backbone 레지스트리 (모델 로드 인터페이스) |
| `src/masks/multiseq_multiblock3d.py` | Spatial/temporal multiblock masking collator |
| `src/datasets/video_dataset.py` | CSV 기반 비디오 dataset (decord, 다중 데이터셋 혼합) |
| `notebooks/utils/mpc_utils.py` | **CEM planner** — 7-DoF robot action 최적화 |
| `notebooks/utils/world_model_wrapper.py` | `WorldModel` wrapper — encode + CEM planning 통합 |
| `notebooks/energy_landscape_example.ipynb` | CEM planning + energy landscape 시각화 demo |
| `evals/video_classification_frozen/eval.py` | Frozen encoder probe (K400, SSv2 등) |
| `evals/action_anticipation_frozen/eval.py` | EpicKitchens action anticipation frozen eval |
| `configs/train/vitg16/pretrain-256px-16f.yaml` | vitG 기본 pretraining config (16f, K710+SSv2+HowTo) |
| `configs/train/vitg16/droid-256px-8f.yaml` | DROID action-conditioned fine-tuning config |
