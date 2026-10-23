.PHONY: install ingest api ui eval test clean

install:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	@echo ""
	@echo "Done. Next:  cp .env.example .env"

ingest:
	.venv/bin/python scripts/ingest.py

api:
	.venv/bin/uvicorn app.main:app --reload --port 8001

ui:
	.venv/bin/streamlit run ui/streamlit_app.py

eval:
	.venv/bin/python eval/run_eval.py

test:
	.venv/bin/pytest -q

clean:
	rm -rf data/processed/* && touch data/processed/.gitkeep
