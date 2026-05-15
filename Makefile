.PHONY: help install test quick demo synthetic tabular drift sweep all clean

# Override on the command line to relocate the dataset cache:
#   make tabular CFL_DATA_CACHE=$HOME/.cache/cfl_data
export CFL_DATA_CACHE ?= /tmp/cfl_data_cache

help:
	@echo "Available targets:"
	@echo "  install     pip install -r requirements.txt"
	@echo "  test        run all unit tests (~50 tests, < 1 s)"
	@echo "  quick       fast smoke checks - synthetic demo + Z-drift demo"
	@echo "  synthetic   end-to-end sweep on synthetic, 10 seeds, writes results/end_to_end_sweep.md"
	@echo "  tabular     end-to-end on UCI Adult and HAR; downloads ~25 MB to \$$CFL_DATA_CACHE"
	@echo "  baselines   6-way method comparison on synthetic + Adult + HAR"
	@echo "  drift       4 methods x 2 detectors x 3 datasets drift matrix"
	@echo "  sweep       boundary-fraction regime sweep on synthetic"
	@echo "  audit       fairness audits: defuzz ablation, tau sweep, HFLTS tuning"
	@echo "  all         everything below 'audit', reproducing every table in README"
	@echo "  clean       drop results/*.md (re-run targets to regenerate) + __pycache__"

install:
	pip install -r requirements.txt

test:
	python -m unittest discover -s tests -v

quick:
	python -m experiments.synthetic_demo --seed 0
	python -m experiments.drift_demo --seed 0

synthetic:
	python -m experiments.end_to_end_sweep --n-seeds 10 --out-file results/end_to_end_sweep.md

tabular:
	python -m experiments.end_to_end_adult --n-seeds 5 --n-rounds 20 --out-file results/adult.md
	python -m experiments.end_to_end_har --n-seeds 3 --n-rounds 20 --out-file results/har_easy.md
	python -m experiments.end_to_end_har --n-seeds 3 --n-profiles 5 --boundary-fraction 0.5 \
		--n-rounds 25 --out-file results/har_hard.md

baselines:
	python -m experiments.compare_baselines --datasets synthetic adult har \
		--n-seeds 5 --n-rounds 20 --out-file results/baselines.md

drift:
	python -m experiments.drift_matrix --datasets synthetic adult har \
		--n-seeds 3 --n-rounds 20 --drift-round 10 --out-file results/drift_matrix.md

sweep:
	python -m experiments.regime_sweep --n-seeds 5 --n-rounds 20 --out-file results/regime_sweep.md

audit:
	python -m experiments.sweep_defuzz --out-file results/sweep_defuzz.md
	python -m experiments.sweep_granularity --out-file results/sweep_granularity.md
	python -m experiments.tau_sweep --n-seeds 10 --out-file results/tau_sweep.md
	python -m experiments.ablation --n-seeds 10 --out-file results/ablation.md

all: test synthetic tabular baselines drift sweep

clean:
	rm -f results/*.md
	find . -type d -name __pycache__ -exec rm -rf {} +
