install:
	pip install -r requirements.txt

train:
	python -m src.train --episodes 2000

evaluate:
	python -m src.evaluate --episodes 500

test:
	pytest -q

run:
	uvicorn src.api:app --reload

docker:
	docker compose up --build
