SHELL := /bin/bash

MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MKFILE_DIR := $(dir $(MKFILE_PATH))

DOCKER_COMPOSE_FILES := \
	-f docker-compose.yml

DATA_DIR ?= $(MKFILE_DIR)../../data/
OUTPUTS_DIR ?= $(MKFILE_DIR)../../prediction_outputs/
PREPROCESSED_DATA_DIR ?= $(MKFILE_DIR)../../preprocessed_data/

PARAMETERS := MKFILE_DIR=$(MKFILE_DIR) \
	DATA_DIR=$(DATA_DIR) \
	OUTPUTS_DIR=$(OUTPUTS_DIR) \
	PREPROCESSED_DATA_DIR=$(PREPROCESSED_DATA_DIR)

BASE_DIR := $(MKFILE_DIR)/PCGrounder/

prepare-terminal-for-visualization:
	xhost +local:docker

build:
	cd $(MKFILE_DIR) && \
	$(PARAMETERS) \
	BASE_DIR=$(BASE_DIR) \
	docker compose build base

up:
	cd $(MKFILE_DIR) && \
	$(PARAMETERS) \
	BASE_DIR=$(BASE_DIR) \
	docker compose up base

into:
	cd $(MKFILE_DIR) && \
	$(PARAMETERS) \
	BASE_DIR=$(BASE_DIR) \
	docker compose exec base bash

stop:
	cd $(MKFILE_DIR) && \
	$(PARAMETERS) \
	BASE_DIR=$(BASE_DIR) \
	docker compose stop base