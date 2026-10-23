.PHONY: install test

install:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	@echo ""
	@echo "Done. Next:  cp .env.example .env"

test:
	.venv/bin/pytest -q
