.PHONY: up down logs restart clean

up:
	docker-compose up -d
	@echo "Waiting for services to start..."
	@sleep 10
	@echo "Services started. Backend: http://localhost:8000, Neo4j: http://localhost:7474"

down:
	docker-compose down

logs:
	docker-compose logs -f

restart:
	docker-compose restart

clean:
	docker-compose down -v
	docker volume prune -f

