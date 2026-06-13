.PHONY: build run stop clean

build:
	docker compose -f docker/docker-compose.yml build

run:
	docker compose -f docker/docker-compose.yml up

stop:
	docker compose -f docker/docker-compose.yml down

clean:
	docker compose -f docker/docker-compose.yml down --rmi all --volumes
