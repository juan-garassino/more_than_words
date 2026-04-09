# =============================================================================
# Living Tales — Makefile
#
# Naming convention:
#   ac-   = amber_cipher   (small, 3-dim, 72 tokens)
#   acm-  = amber_cipher_M (medium, 5-dim, 152 tokens — was amber_cipher_L)
#
# Full pipeline (runs all 5 stages in order):
#   make ac-pipeline          amber_cipher, CPU
#   make ac-pipeline-gpu      amber_cipher, GPU
#   make acm-pipeline         amber_cipher_M, CPU
#   make acm-pipeline-gpu     amber_cipher_M, GPU
#
# Individual stages:
#   s01-validate    check case JSON structure
#   s02-pack        pack JSON → trainer/cases/<id>/
#   s03-train-hopfield   train symbolic Hopfield model → model.pt
#   s04-train-policy     supervised + REINFORCE → policy.pt
#   s05-benchmark        compare both, print XCODE recommendation
#
# Utilities:
#   colab-install   pip install requirements (run once on fresh Colab)
# =============================================================================

.PHONY: \
  colab-install \
  creature-m-report creature-m-baseline \
  ac-s01-validate ac-s02-pack \
  ac-s03-train-hopfield ac-s03-train-hopfield-gpu ac-s03-train-hopfield-fastproof \
  ac-s04-train-policy ac-s04-train-policy-gpu \
  ac-s05-benchmark \
  ac-pipeline ac-pipeline-gpu \
  acm-s01-validate acm-s02-pack \
  acm-s03-train-hopfield acm-s03-train-hopfield-gpu \
  acm-s04-train-policy acm-s04-train-policy-gpu \
  acm-s05-benchmark \
  acm-pipeline acm-pipeline-gpu \
  validate-amber-cipher pack-amber-cipher train-amber-cipher-colab-cpu

# Shared env flags (suppress threading conflicts on macOS / Colab)
ENV := PYTHONUNBUFFERED=1 KMP_DUPLICATE_LIB_OK=TRUE KMP_USE_SHM=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# =============================================================================
# Utilities
# =============================================================================

colab-install:
	@echo ""
	@echo "================================================================"
	@echo "  INSTALL — python dependencies"
	@echo "================================================================"
	python3 -m pip install -r living_tales/trainer/requirements.txt
	@echo "  [OK] dependencies installed"

creature-m-baseline:
	python3 -m evals.utils.baseline_runner little_creature_M --n-games 100

creature-m-report:
	python3 living_tales/trainer/tools/report_creature_case.py little_creature_M --runs 100

# =============================================================================
# amber_cipher  (ac-)
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 01 — Validate amber_cipher.json
# -----------------------------------------------------------------------------

ac-s01-validate:
	@echo ""
	@echo "================================================================"
	@echo "  AC  STAGE 01 — VALIDATE  (amber_cipher.json)"
	@echo "================================================================"
	python3 living_tales_case_validator.py cases/amber_cipher.json
	@echo "  [OK] amber_cipher.json is valid"

validate-amber-cipher: ac-s01-validate   # backwards-compat alias

# -----------------------------------------------------------------------------
# Stage 02 — Pack amber_cipher → trainer/cases/amber_cipher/
# -----------------------------------------------------------------------------

ac-s02-pack: ac-s01-validate
	@echo ""
	@echo "================================================================"
	@echo "  AC  STAGE 02 — PACK  (amber_cipher → cases/amber_cipher/)"
	@echo "================================================================"
	python3 living_tales/trainer/tools/pack_case.py cases/amber_cipher.json
	@echo "  [OK] packed to living_tales/trainer/cases/amber_cipher/"

pack-amber-cipher: ac-s02-pack   # backwards-compat alias

# -----------------------------------------------------------------------------
# Stage 03 — Train Hopfield model
# Output: living_tales/trainer/outputs/amber_cipher/model.pt
# -----------------------------------------------------------------------------

ac-s03-train-hopfield: ac-s02-pack colab-install
	@echo ""
	@echo "================================================================"
	@echo "  AC  STAGE 03 — TRAIN HOPFIELD MODEL  (CPU)"
	@echo "  output: outputs/amber_cipher/model.pt"
	@echo "================================================================"
	cd living_tales/trainer && $(ENV) PYTHONPATH=. \
	python3 tools/train_single_case.py amber_cipher \
	  --paths 300 --epochs 20 --proof-paths 200 --proof-max-attempts 2000 --device cpu
	@echo "  [OK] model.pt saved"

ac-s03-train-hopfield-gpu: ac-s02-pack colab-install
	@echo ""
	@echo "================================================================"
	@echo "  AC  STAGE 03 — TRAIN HOPFIELD MODEL  (GPU)"
	@echo "  output: outputs/amber_cipher/model.pt"
	@echo "================================================================"
	cd living_tales/trainer && $(ENV) PYTHONPATH=. \
	python3 tools/train_single_case.py amber_cipher \
	  --paths 300 --epochs 20 --proof-paths 200 --proof-max-attempts 2000 --device cuda
	@echo "  [OK] model.pt saved"

ac-s03-train-hopfield-fastproof: ac-s02-pack colab-install
	@echo ""
	@echo "================================================================"
	@echo "  AC  STAGE 03 — TRAIN HOPFIELD MODEL  (GPU, fast proof)"
	@echo "================================================================"
	cd living_tales/trainer && $(ENV) PYTHONPATH=. \
	python3 tools/train_single_case.py amber_cipher \
	  --paths 300 --epochs 20 --proof-paths 50 --proof-max-attempts 500 --device cuda
	@echo "  [OK] model.pt saved"

train-amber-cipher-colab-cpu: ac-s03-train-hopfield   # backwards-compat alias

# -----------------------------------------------------------------------------
# Stage 04 — Train transformer policy
# Requires: model.pt from stage 03
# Output:   living_tales/trainer/outputs/amber_cipher/policy.pt
# -----------------------------------------------------------------------------

ac-s04-train-policy: ac-s02-pack colab-install
	@echo ""
	@echo "================================================================"
	@echo "  AC  STAGE 04 — TRAIN TRANSFORMER POLICY  (CPU)"
	@echo "  supervised pretraining + REINFORCE fine-tuning"
	@echo "  requires: outputs/amber_cipher/model.pt  (run s03 first)"
	@echo "  output:   outputs/amber_cipher/policy.pt"
	@echo "================================================================"
	cd living_tales/trainer && $(ENV) PYTHONPATH=. \
	python3 trainer/train_policy.py amber_cipher \
	  --supervised-paths 2000 --supervised-epochs 20 --rl-episodes 500
	@echo "  [OK] policy.pt saved"

ac-s04-train-policy-gpu: ac-s02-pack colab-install
	@echo ""
	@echo "================================================================"
	@echo "  AC  STAGE 04 — TRAIN TRANSFORMER POLICY  (GPU)"
	@echo "  output: outputs/amber_cipher/policy.pt"
	@echo "================================================================"
	cd living_tales/trainer && $(ENV) PYTHONPATH=. \
	python3 trainer/train_policy.py amber_cipher \
	  --supervised-paths 2000 --supervised-epochs 20 --rl-episodes 500 --device cuda
	@echo "  [OK] policy.pt saved"

ac-s04-train-policy-fast: ac-s02-pack colab-install
	@echo ""
	@echo "================================================================"
	@echo "  AC  STAGE 04 — TRAIN TRANSFORMER POLICY  (fast / GPU)"
	@echo "  500 paths · 5 KD epochs · 200 RL episodes"
	@echo "  smoke-test: ~8 min on GPU"
	@echo "================================================================"
	cd living_tales/trainer && $(ENV) PYTHONPATH=. \
	python3 trainer/train_policy.py amber_cipher \
	  --supervised-paths 500 --supervised-epochs 5 --rl-episodes 200 --device cuda
	@echo "  [OK] policy.pt saved"

# -----------------------------------------------------------------------------
# Stage 04d — Train dialogue transformer (supervised KD + REINFORCE)
# Output: living_tales/trainer/outputs/amber_cipher/dialogue_model.pt
# -----------------------------------------------------------------------------

ac-s04-train-dialogue: ac-s02-pack colab-install
	@echo ""
	@echo "================================================================"
	@echo "  AC  STAGE 04d — TRAIN DIALOGUE TRANSFORMER  (CPU)"
	@echo "  supervised KD + REINFORCE fine-tuning"
	@echo "  output: outputs/amber_cipher/dialogue_model.pt"
	@echo "================================================================"
	cd living_tales/trainer && $(ENV) PYTHONPATH=. \
	python3 tools/train_single_case.py amber_cipher \
	  --model-type dialogue --paths 2000 --epochs 100 --rl-episodes 500 --device cpu --skip-proof
	@echo "  [OK] dialogue_model.pt saved"

ac-s04-train-dialogue-gpu: ac-s02-pack colab-install
	@echo ""
	@echo "================================================================"
	@echo "  AC  STAGE 04d — TRAIN DIALOGUE TRANSFORMER  (GPU)"
	@echo "  output: outputs/amber_cipher/dialogue_model.pt"
	@echo "================================================================"
	cd living_tales/trainer && $(ENV) PYTHONPATH=. \
	python3 tools/train_single_case.py amber_cipher \
	  --model-type dialogue --paths 2000 --epochs 100 --rl-episodes 500 --device cuda --skip-proof
	@echo "  [OK] dialogue_model.pt saved"

ac-s04-train-dialogue-fast: ac-s02-pack colab-install
	@echo ""
	@echo "================================================================"
	@echo "  AC  STAGE 04d — TRAIN DIALOGUE TRANSFORMER  (fast / CPU)"
	@echo "  500 paths · 30 KD epochs · 100 RL episodes"
	@echo "================================================================"
	cd living_tales/trainer && $(ENV) PYTHONPATH=. \
	python3 tools/train_single_case.py amber_cipher \
	  --model-type dialogue --paths 500 --epochs 30 --rl-episodes 100 --device cpu --skip-proof
	@echo "  [OK] dialogue_model.pt saved"

# -----------------------------------------------------------------------------
# Stage 05 — Benchmark + Xcode recommendation
# Requires: policy.pt from stage 04
# -----------------------------------------------------------------------------

ac-s05-benchmark: colab-install
	@echo ""
	@echo "================================================================"
	@echo "  AC  STAGE 05 — BENCHMARK  (Hopfield vs Transformer)"
	@echo "  requires: outputs/amber_cipher/policy.pt  (run s04 first)"
	@echo "================================================================"
	cd living_tales/trainer && $(ENV) PYTHONPATH=. \
	python3 tools/benchmark_models.py amber_cipher --n-episodes 200
	@echo "  [OK] see XCODE MODEL RECOMMENDATION above"

# -----------------------------------------------------------------------------
# Full pipeline — amber_cipher
# Runs all 5 stages in sequence.  Check the DECISION line at the end.
# -----------------------------------------------------------------------------

ac-pipeline: colab-install
	@echo ""
	@echo "################################################################"
	@echo "  LIVING TALES — AMBER CIPHER FULL PIPELINE  (CPU)"
	@echo "  s01 validate → s02 pack → s03 hopfield → s04 policy → s05 benchmark"
	@echo "################################################################"
	$(MAKE) ac-s01-validate
	$(MAKE) ac-s02-pack
	$(MAKE) ac-s03-train-hopfield
	$(MAKE) ac-s04-train-policy
	$(MAKE) ac-s05-benchmark
	@echo ""
	@echo "################################################################"
	@echo "  PIPELINE COMPLETE"
	@echo "  Check the XCODE MODEL RECOMMENDATION block above for the"
	@echo "  .pt file to load in Xcode."
	@echo "################################################################"

ac-pipeline-gpu: colab-install
	@echo ""
	@echo "################################################################"
	@echo "  LIVING TALES — AMBER CIPHER FULL PIPELINE  (GPU)"
	@echo "################################################################"
	$(MAKE) ac-s01-validate
	$(MAKE) ac-s02-pack
	$(MAKE) ac-s03-train-hopfield-gpu
	$(MAKE) ac-s04-train-policy-gpu
	$(MAKE) ac-s05-benchmark
	@echo ""
	@echo "################################################################"
	@echo "  PIPELINE COMPLETE"
	@echo "  Check the XCODE MODEL RECOMMENDATION block above for the"
	@echo "  .pt file to load in Xcode."
	@echo "################################################################"

ac-pipeline-fast: colab-install
	@echo ""
	@echo "################################################################"
	@echo "  LIVING TALES — AMBER CIPHER FAST PIPELINE  (GPU smoke-test)"
	@echo "  s03 hopfield (GPU) → s04 policy fast (500p/5e/200rl) → s05"
	@echo "  ~25 min total: hopfield+proof ~15 min, policy ~8 min"
	@echo "################################################################"
	$(MAKE) ac-s01-validate
	$(MAKE) ac-s02-pack
	$(MAKE) ac-s03-train-hopfield-gpu
	$(MAKE) ac-s04-train-policy-fast
	$(MAKE) ac-s05-benchmark
	@echo ""
	@echo "################################################################"
	@echo "  FAST PIPELINE COMPLETE"
	@echo "################################################################"

# =============================================================================
# amber_cipher_M  (acm-)  — 5-dim expansion, 152 tokens
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 01 — Validate amber_cipher_M.json
# -----------------------------------------------------------------------------

acm-s01-validate:
	@echo ""
	@echo "================================================================"
	@echo "  ACM STAGE 01 — VALIDATE  (amber_cipher_M.json)"
	@echo "================================================================"
	python3 living_tales_case_validator.py cases/amber_cipher_M.json
	@echo "  [OK] amber_cipher_M.json is valid"

validate-amber-cipher-M: acm-s01-validate

# -----------------------------------------------------------------------------
# Stage 02 — Pack amber_cipher_M → trainer/cases/amber_cipher_M/
# -----------------------------------------------------------------------------

acm-s02-pack: acm-s01-validate
	@echo ""
	@echo "================================================================"
	@echo "  ACM STAGE 02 — PACK  (amber_cipher_M → cases/amber_cipher_M/)"
	@echo "================================================================"
	python3 living_tales/trainer/tools/pack_case.py cases/amber_cipher_M.json
	@echo "  [OK] packed to living_tales/trainer/cases/amber_cipher_M/"

pack-amber-cipher-M: acm-s02-pack

# -----------------------------------------------------------------------------
# Stage 03 — Train Hopfield model
# Output: living_tales/trainer/outputs/amber_cipher_M/model.pt
# -----------------------------------------------------------------------------

acm-s03-train-hopfield: acm-s02-pack colab-install
	@echo ""
	@echo "================================================================"
	@echo "  ACM STAGE 03 — TRAIN HOPFIELD MODEL  (CPU)"
	@echo "  output: outputs/amber_cipher_M/model.pt"
	@echo "================================================================"
	cd living_tales/trainer && $(ENV) PYTHONPATH=. \
	python3 tools/train_single_case.py amber_cipher_M \
	  --paths 500 --epochs 20 --proof-paths 200 --proof-max-attempts 2000 --device cpu
	@echo "  [OK] model.pt saved"

acm-s03-train-hopfield-gpu: acm-s02-pack colab-install
	@echo ""
	@echo "================================================================"
	@echo "  ACM STAGE 03 — TRAIN HOPFIELD MODEL  (GPU)"
	@echo "  output: outputs/amber_cipher_M/model.pt"
	@echo "================================================================"
	cd living_tales/trainer && $(ENV) PYTHONPATH=. \
	python3 tools/train_single_case.py amber_cipher_M \
	  --paths 500 --epochs 20 --proof-paths 200 --proof-max-attempts 2000 --device cuda
	@echo "  [OK] model.pt saved"

# -----------------------------------------------------------------------------
# Stage 04 — Train transformer policy
# Requires: model.pt from stage 03
# Output:   living_tales/trainer/outputs/amber_cipher_M/policy.pt
# -----------------------------------------------------------------------------

acm-s04-train-policy: acm-s02-pack colab-install
	@echo ""
	@echo "================================================================"
	@echo "  ACM STAGE 04 — TRAIN TRANSFORMER POLICY  (CPU)"
	@echo "  requires: outputs/amber_cipher_M/model.pt  (run s03 first)"
	@echo "  output:   outputs/amber_cipher_M/policy.pt"
	@echo "================================================================"
	cd living_tales/trainer && $(ENV) PYTHONPATH=. \
	python3 trainer/train_policy.py amber_cipher_M \
	  --supervised-paths 2000 --supervised-epochs 20 --rl-episodes 500
	@echo "  [OK] policy.pt saved"

acm-s04-train-policy-gpu: acm-s02-pack colab-install
	@echo ""
	@echo "================================================================"
	@echo "  ACM STAGE 04 — TRAIN TRANSFORMER POLICY  (GPU)"
	@echo "  output: outputs/amber_cipher_M/policy.pt"
	@echo "================================================================"
	cd living_tales/trainer && $(ENV) PYTHONPATH=. \
	python3 trainer/train_policy.py amber_cipher_M \
	  --supervised-paths 2000 --supervised-epochs 20 --rl-episodes 500 --device cuda
	@echo "  [OK] policy.pt saved"

# -----------------------------------------------------------------------------
# Stage 05 — Benchmark + Xcode recommendation
# -----------------------------------------------------------------------------

acm-s05-benchmark: colab-install
	@echo ""
	@echo "================================================================"
	@echo "  ACM STAGE 05 — BENCHMARK  (Hopfield vs Transformer)"
	@echo "  requires: outputs/amber_cipher_M/policy.pt  (run s04 first)"
	@echo "================================================================"
	cd living_tales/trainer && $(ENV) PYTHONPATH=. \
	python3 tools/benchmark_models.py amber_cipher_M --n-episodes 200
	@echo "  [OK] see XCODE MODEL RECOMMENDATION above"

# -----------------------------------------------------------------------------
# Full pipeline — amber_cipher_M
# -----------------------------------------------------------------------------

acm-pipeline: colab-install
	@echo ""
	@echo "################################################################"
	@echo "  LIVING TALES — AMBER CIPHER M FULL PIPELINE  (CPU)"
	@echo "  s01 validate → s02 pack → s03 hopfield → s04 policy → s05 benchmark"
	@echo "################################################################"
	$(MAKE) acm-s01-validate
	$(MAKE) acm-s02-pack
	$(MAKE) acm-s03-train-hopfield
	$(MAKE) acm-s04-train-policy
	$(MAKE) acm-s05-benchmark
	@echo ""
	@echo "################################################################"
	@echo "  PIPELINE COMPLETE"
	@echo "  Check the XCODE MODEL RECOMMENDATION block above for the"
	@echo "  .pt file to load in Xcode."
	@echo "################################################################"

acm-pipeline-gpu: colab-install
	@echo ""
	@echo "################################################################"
	@echo "  LIVING TALES — AMBER CIPHER M FULL PIPELINE  (GPU)"
	@echo "################################################################"
	$(MAKE) acm-s01-validate
	$(MAKE) acm-s02-pack
	$(MAKE) acm-s03-train-hopfield-gpu
	$(MAKE) acm-s04-train-policy-gpu
	$(MAKE) acm-s05-benchmark
	@echo ""
	@echo "################################################################"
	@echo "  PIPELINE COMPLETE"
	@echo "  Check the XCODE MODEL RECOMMENDATION block above for the"
	@echo "  .pt file to load in Xcode."
	@echo "################################################################"

# =============================================================================
# Pre-fitting infrastructure (audit, baselines, proof, visualization)
# =============================================================================

ac-audit: ac-s02-pack
	@echo ""
	@echo "================================================================"
	@echo "  AC  AUDIT — DIALOGUE VIABILITY  (amber_cipher)"
	@echo "================================================================"
	cd living_tales/trainer && $(ENV) PYTHONPATH=. \
	python3 tools/audit_case_dialogue.py amber_cipher
	@echo "  [OK] audit complete"

ac-s00-baselines: ac-s02-pack colab-install
	@echo ""
	@echo "================================================================"
	@echo "  AC  BASELINES — RANDOM PLAY METRICS  (amber_cipher)"
	@echo "================================================================"
	cd evals && PYTHONPATH=../living_tales/trainer python3 -m utils.baseline_runner amber_cipher
	@echo "  [OK] baselines saved"

ac-dialogue-proof: ac-s02-pack colab-install
	@echo ""
	@echo "================================================================"
	@echo "  AC  DIALOGUE CONVERGENCE PROOF  (amber_cipher)"
	@echo "================================================================"
	cd living_tales/trainer && $(ENV) PYTHONPATH=. \
	python3 -c "from validator.dialogue_proof import DialogueConvergenceProof; from core.cartridge import CartridgeSpec; spec=CartridgeSpec.load('cases/amber_cipher/spec.json'); DialogueConvergenceProof().run(spec, n_test_dialogues=200)"
	@echo "  [OK] proof complete"

ac-visualize: ac-s02-pack colab-install
	@echo ""
	@echo "================================================================"
	@echo "  AC  VISUALIZE  (amber_cipher)"
	@echo "================================================================"
	cd living_tales/trainer && $(ENV) PYTHONPATH=. \
	python3 tools/visualize.py trajectories amber_cipher --output-dir outputs/amber_cipher
	@echo "  [OK] visualizations saved"

ac-preflight: ac-audit ac-s00-baselines ac-dialogue-proof
	@echo ""
	@echo "################################################################"
	@echo "  AMBER CIPHER PREFLIGHT COMPLETE"
	@echo "  Case is ready for dialogue training."
	@echo "################################################################"

# =============================================================================
# fog_over_brussels (fob-)
# =============================================================================

fob-s01-validate:
	@echo ""
	@echo "================================================================"
	@echo "  FOB  STAGE 01 — VALIDATE  (fog_over_brussels.json)"
	@echo "================================================================"
	python3 living_tales_case_validator.py cases/fog_over_brussels.json
	@echo "  [OK] fog_over_brussels.json is valid"

fob-s02-pack: fob-s01-validate
	@echo ""
	@echo "================================================================"
	@echo "  FOB  STAGE 02 — PACK  (fog_over_brussels)"
	@echo "================================================================"
	python3 living_tales/trainer/tools/pack_case.py cases/fog_over_brussels.json
	@echo "  [OK] packed to living_tales/trainer/cases/fog_over_brussels/"

fob-audit: fob-s02-pack
	cd living_tales/trainer && $(ENV) PYTHONPATH=. \
	python3 tools/audit_case_dialogue.py fog_over_brussels

fob-baselines: fob-s02-pack colab-install
	cd evals && PYTHONPATH=../living_tales/trainer python3 -m utils.baseline_runner fog_over_brussels

fob-dialogue-proof: fob-s02-pack colab-install
	cd living_tales/trainer && $(ENV) PYTHONPATH=. \
	python3 -c "from validator.dialogue_proof import DialogueConvergenceProof; from core.cartridge import CartridgeSpec; spec=CartridgeSpec.load('cases/fog_over_brussels/spec.json'); DialogueConvergenceProof().run(spec, n_test_dialogues=200)"

fob-preflight: fob-audit fob-baselines fob-dialogue-proof
	@echo ""
	@echo "################################################################"
	@echo "  FOG OVER BRUSSELS PREFLIGHT COMPLETE"
	@echo "################################################################"

# =============================================================================
# Fit · Play · Report (end-to-end smoke test)
# =============================================================================

ac-fit-play-report: ac-s02-pack colab-install
	@echo ""
	@echo "================================================================"
	@echo "  LIVING TALES FIT PLAY REPORT  (amber_cipher)"
	@echo "================================================================"
	cd living_tales/trainer && $(ENV) PYTHONPATH=. \
	python3 tools/fit_play_report.py amber_cipher \
	  --paths 500 --epochs 30 --rl-episodes 100 --games 3

# =============================================================================
# Train all valid cases (batch)
# =============================================================================

train-all-fast: colab-install
	@echo ""
	@echo "################################################################"
	@echo "  LIVING TALES — TRAIN ALL CASES (fast)"
	@echo "################################################################"
	cd living_tales/trainer && $(ENV) PYTHONPATH=. \
	python3 tools/train_all_cases.py --paths 500 --epochs 30 --rl-episodes 100 --games 3

train-all: colab-install
	@echo ""
	@echo "################################################################"
	@echo "  LIVING TALES — TRAIN ALL CASES"
	@echo "################################################################"
	cd living_tales/trainer && $(ENV) PYTHONPATH=. \
	python3 tools/train_all_cases.py --paths 2000 --epochs 100 --rl-episodes 500 --games 5

train-all-production: colab-install
	@echo ""
	@echo "################################################################"
	@echo "  LIVING TALES — TRAIN ALL CASES (production)"
	@echo "################################################################"
	cd living_tales/trainer && $(ENV) PYTHONPATH=. \
	python3 tools/train_all_cases.py --production
