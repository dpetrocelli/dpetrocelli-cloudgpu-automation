.PHONY: install dev lint format check clean help
.PHONY: search cheap list balance billing images
.PHONY: launch destroy destroy-all start stop ssh
.PHONY: ssh-keygen ssh-upload ssh-setup
.PHONY: upload download run

# Default Python
PYTHON := ./venv/bin/python3
PIP := ./venv/bin/pip

# Variables for commands (override with: make launch ID=123)
ID ?=
PRICE ?=
GPU ?=
IMAGE ?= pytorch
SRC ?=
DST ?=
CMD ?=

help:
	@echo "Vast.ai GPU CLI - Available commands:"
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install dependencies"
	@echo "  make dev            Install dev dependencies (pre-commit, ruff)"
	@echo "  make ssh-setup      Generate SSH key and upload to Vast.ai"
	@echo "  make ssh-keygen     Generate SSH key (if not exists)"
	@echo "  make ssh-upload     Upload SSH key to Vast.ai"
	@echo ""
	@echo "Code quality:"
	@echo "  make lint           Run linter (ruff check)"
	@echo "  make format         Format code (ruff format)"
	@echo "  make check          Run all pre-commit hooks"
	@echo ""
	@echo "Search & Info:"
	@echo "  make search                   Search all GPUs (cheapest first)"
	@echo "  make search PRICE=0.10        Search GPUs under \$$0.10/hr"
	@echo "  make search GPU=RTX_4090      Search specific GPU"
	@echo "  make cheap                    Search cheap GPUs (<\$$0.04, best value)"
	@echo "  make cheap PRICE=0.06         Override max price"
	@echo "  make list                     List your instances"
	@echo "  make balance                  Show account balance"
	@echo "  make billing                  Show billing history"
	@echo "  make images                   Show available Docker images"
	@echo ""
	@echo "Instance Management:"
	@echo "  make launch ID=123            Launch instance by offer ID"
	@echo "  make launch ID=123,456        Launch multiple instances"
	@echo "  make ssh ID=123               Get SSH command for instance"
	@echo "  make start ID=123             Start a stopped instance"
	@echo "  make stop ID=123              Stop a running instance"
	@echo "  make destroy ID=123           Destroy an instance"
	@echo "  make destroy-all              Destroy ALL instances (with confirm)"
	@echo "  make destroy-all-force        Destroy ALL instances (no confirm)"
	@echo ""
	@echo "File Transfer & Remote Execution:"
	@echo "  make upload ID=123 SRC=./file DST=/root/   Upload files to instance"
	@echo "  make download ID=123 SRC=/root/file DST=./ Download files from instance"
	@echo "  make run ID=123 CMD='python3 test.py'      Run command on instance"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean                    Remove cache files"

# Setup
install:
	python3 -m venv venv
	$(PIP) install -r requirements.txt

dev: install
	$(PIP) install pre-commit ruff
	./venv/bin/pre-commit install

# SSH key path
SSH_KEY ?= ~/.ssh/id_ed25519

ssh-keygen:
	@if [ -f $(SSH_KEY) ]; then \
		echo "SSH key already exists: $(SSH_KEY)"; \
	else \
		echo "Generating SSH key..."; \
		ssh-keygen -t ed25519 -f $(SSH_KEY) -N ""; \
	fi

ssh-upload:
	$(PYTHON) cli.py ssh-key $(SSH_KEY).pub

ssh-setup: ssh-keygen ssh-upload

# Code quality
lint:
	./venv/bin/ruff check .

format:
	./venv/bin/ruff format .
	./venv/bin/ruff check --fix .

check:
	./venv/bin/pre-commit run --all-files

# Search & Info
search:
ifdef PRICE
ifdef GPU
	$(PYTHON) cli.py search --gpu $(GPU) --max-price $(PRICE)
else
	$(PYTHON) cli.py search --max-price $(PRICE)
endif
else ifdef GPU
	$(PYTHON) cli.py search --gpu $(GPU)
else
	$(PYTHON) cli.py search
endif

# Default price for cheap search
CHEAP_PRICE ?= 0.04

cheap:
ifdef PRICE
	$(PYTHON) cli.py search --max-price $(PRICE) --order-by price_power $(if $(VRAM),--min-gpu-ram $(VRAM))
else
	$(PYTHON) cli.py search --max-price $(CHEAP_PRICE) --order-by price_power $(if $(VRAM),--min-gpu-ram $(VRAM))
endif

list:
	$(PYTHON) cli.py list

balance:
	$(PYTHON) cli.py balance

billing:
	$(PYTHON) cli.py billing

images:
	$(PYTHON) cli.py images

# Instance Management
launch:
ifndef ID
	@echo "Error: ID required. Usage: make launch ID=123"
	@exit 1
endif
	$(PYTHON) cli.py launch --id $(ID) --image $(IMAGE)

ssh:
ifndef ID
	@echo "Error: ID required. Usage: make ssh ID=123"
	@exit 1
endif
	$(PYTHON) cli.py ssh $(ID)

start:
ifndef ID
	@echo "Error: ID required. Usage: make start ID=123"
	@exit 1
endif
	$(PYTHON) cli.py start $(ID)

stop:
ifndef ID
	@echo "Error: ID required. Usage: make stop ID=123"
	@exit 1
endif
	$(PYTHON) cli.py stop $(ID)

destroy:
ifndef ID
	@echo "Error: ID required. Usage: make destroy ID=123"
	@exit 1
endif
	$(PYTHON) cli.py destroy $(ID)

destroy-all:
	$(PYTHON) cli.py destroy --all

destroy-all-force:
	$(PYTHON) cli.py destroy --all --force

# File Transfer & Remote Execution
upload:
ifndef ID
	@echo "Error: ID required. Usage: make upload ID=123 SRC=./file"
	@exit 1
endif
ifndef SRC
	@echo "Error: SRC required. Usage: make upload ID=123 SRC=./file"
	@exit 1
endif
ifdef DST
	$(PYTHON) cli.py upload $(ID) $(SRC) --dst $(DST)
else
	$(PYTHON) cli.py upload $(ID) $(SRC)
endif

download:
ifndef ID
	@echo "Error: ID required. Usage: make download ID=123 SRC=/root/file"
	@exit 1
endif
ifndef SRC
	@echo "Error: SRC required. Usage: make download ID=123 SRC=/root/file"
	@exit 1
endif
ifdef DST
	$(PYTHON) cli.py download $(ID) $(SRC) --dst $(DST)
else
	$(PYTHON) cli.py download $(ID) $(SRC)
endif

run:
ifndef ID
	@echo "Error: ID required. Usage: make run ID=123 CMD='python test.py'"
	@exit 1
endif
ifndef CMD
	@echo "Error: CMD required. Usage: make run ID=123 CMD='python test.py'"
	@exit 1
endif
	$(PYTHON) cli.py run $(ID) "$(CMD)"

# Utilities
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
