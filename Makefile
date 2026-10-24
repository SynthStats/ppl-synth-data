PROJECT_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

RUN_ID ?= validation
RUN_ROOT ?= artifacts/runs
RUN_LIMIT ?= 5

.PHONY: help test lint typecheck check bandit doctor pipeline-check harbor-e2e

help:
	@printf '%s\n' \
		'Targets:' \
		'  make test                           Run the offline pytest suite.' \
		'  make lint                           Run Ruff.' \
		'  make typecheck                      Run mypy over src.' \
		'  make check                          Run test, lint, and typecheck.' \
		'  make bandit                         Run the security scan.' \
		'  make doctor                         Verify CLI environment setup.' \
		'  make pipeline-check RUN_ID=validation SPLIT_POLICY=auto VISIBLE_RATIO=0.8 Run a bounded end-to-end pipeline check (requires API access).' \
		'  make harbor-e2e RUN_ID=validation   Export Harbor tasks and validate one task in Docker.'

test:
	uv run pytest -q

lint:
	uv run ruff check .

typecheck:
	uv run mypy src

check: test lint typecheck

bandit:
	uv run bandit -r src/ppl_synth_data scripts

doctor:
	uv run ppldf doctor

pipeline-check:
	$(if $(strip $(SPLIT_POLICY)),,$(error SPLIT_POLICY is required))
	$(if $(strip $(VISIBLE_RATIO)),,$(error VISIBLE_RATIO is required))
	uv run --project $(PROJECT_ROOT) --frozen ppldf run all --run-id "$(RUN_ID)" --limit "$(RUN_LIMIT)" --split-policy "$(SPLIT_POLICY)" --visible-ratio "$(VISIBLE_RATIO)" --skip-upload

harbor-e2e:
	bash scripts/harbor_e2e.sh "$(RUN_ID)" "$(RUN_ROOT)"
