.PHONY: setup test leakage-test backtest props-backtest predict report deploy

VENV := .venv/bin
PY := $(VENV)/python3

setup:
	python3.11 -m venv .venv
	$(VENV)/pip install --upgrade pip setuptools wheel
	$(VENV)/pip install -e ".[dev]"

test:
	$(VENV)/pytest tests/ -v

leakage-test:
	$(VENV)/pytest tests/leakage -v

backtest:
	$(PY) -m src.nfl.backtest.moneyline_backtest
	$(PY) -m src.nfl.backtest.spread_total_backtest

props-backtest:
	$(PY) -m src.nfl.backtest.player_props_backtest

predict:
	$(PY) -m src.nfl.reporting.predict

record-results:
	$(PY) -m src.nfl.reporting.record_results

report:
	$(PY) -m src.nfl.reporting.plots
	$(PY) -m src.nfl.reporting.site

# Full weekly refresh: predict -> record last week's results -> rebuild backtest
# artifacts -> regenerate plots -> rebuild the static site.
weekly: predict record-results backtest report

deploy: report
	@echo "Site built to docs/. Commit and push -- GitHub Pages (configured to serve /docs on main) will publish it."
