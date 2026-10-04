# olmo_cl: frozen OlmoEarth encoder + wheat segmentation head, C and OpenCL.
# Standalone build (not a GRASS module).
#
#   make                  build ./olmo_cl
#   make test             quick checks: head vs PyTorch, AdamW/plateau vs torch
#   make selftest         OpenCL kernels vs C loops (DEVICE=, PLATFORM=)
#   make check            full encoder vs PyTorch golden dumps (slow on CPU)
#   make train RUN=dir    fine-tune (EPOCHS=80, DEVICE=gpu, PLATFORM=rusticl)
#   make ckpt RUN=dir     export the run's .head files to Lightning .ckpt
#   make golden           regenerate golden/ and data/val_test_crops.tsv
#   make format           clang-format the C sources
#   make install          copy olmo_cl to $(PREFIX)/bin
#
# On the AMD GPU server: export RUSTICL_ENABLE=radeonsi, then use
# DEVICE=gpu PLATFORM=rusticl.

CC ?= gcc
CFLAGS ?= -O3 -g
CFLAGS += -std=gnu11 -Wall -Wextra -fopenmp -Isrc -Ibuild \
	$(shell pkg-config --cflags OpenCL gdal)
LDLIBS += $(shell pkg-config --libs OpenCL gdal) -fopenmp -lm
PREFIX ?= $(HOME)/.local

SRCS = $(wildcard src/*.c)
OBJS = $(patsubst src/%.c,build/%.o,$(SRCS))

# Inputs. The Python tools need the olmoearth_projects venv (torch,
# rslearn, olmoearth_pretrain); olmo_cl itself needs no Python.
PROJECT ?= $(HOME)/dev/olmoearth_projects
PYTHON ?= $(PROJECT)/.venv/bin/python
MODEL_YAML ?= $(PROJECT)/olmoearth_run_data/wheat_festival/model.yaml
WEIGHTS ?= $(HOME)/models/olmoearth/OlmoEarth-v1-Tiny/weights.pth
DATASET ?= $(HOME)/RSDATA/alma_wheat_festival/dataset
GOLDEN ?= golden/val0,golden/val1,golden/test0,golden/test1,golden/train_mode
HEAD_GOLDEN ?= golden/val0,golden/val1,golden/test0,golden/test1
DEVICE ?= auto
PLATFORM ?=
RUN ?=
EPOCHS ?= 80
SEED ?= 42
VALIDATE ?= 0
CLANG_FORMAT ?= $(firstword $(shell command -v clang-format 2>/dev/null) \
	$(wildcard $(HOME)/.cache/pre-commit/*/py_env-*/bin/clang-format))

all: olmo_cl

olmo_cl: $(OBJS)
	$(CC) -o $@ $^ $(LDLIBS)

build/%.o: src/%.c $(wildcard src/*.h) | build
	$(CC) $(CFLAGS) -c -o $@ $<

# Embed the OpenCL kernels as a C string literal (as in i.sam.opencl), so
# no .cl file has to be deployed next to the binary.
build/ocl_backend.o: build/olmo_kernels_cl.h

build/olmo_kernels_cl.h: kernels/olmo_kernels.cl | build
	sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/^/"/' -e 's/$$/\\n"/' $< > $@

build:
	mkdir -p build

# --- Tests -------------------------------------------------------------

test: headcheck test_optim

selftest: olmo_cl
	./olmo_cl selftest device=$(DEVICE) platform=$(PLATFORM) tokens=4608

headcheck: olmo_cl
	./olmo_cl headcheck golden=$(HEAD_GOLDEN)

build/test_optim: tests/test_optim.c src/optim.c src/util.c | build
	$(CC) -O2 -std=gnu11 -Isrc -o $@ $^ -lm

test_optim: build/test_optim
	$(PYTHON) tests/test_optim.py build/test_optim

check: olmo_cl
	./olmo_cl check golden=$(GOLDEN) weights=$(WEIGHTS) dataset=$(DATASET) \
		device=$(DEVICE) platform=$(PLATFORM)

golden:
	$(PYTHON) tools/make_golden.py $(MODEL_YAML) $(DATASET) golden
	cp golden/crops.tsv data/val_test_crops.tsv

# --- Training and export -----------------------------------------------

train: olmo_cl
	@test -n "$(RUN)" || { echo "make train: set RUN=<output dir>"; exit 1; }
	./olmo_cl train dataset=$(DATASET) weights=$(WEIGHTS) out=$(RUN) \
		epochs=$(EPOCHS) seed=$(SEED) device=$(DEVICE) platform=$(PLATFORM)

# One .ckpt per .head of the run; VALIDATE=1 also runs Lightning validate
# on each with the production PyTorch model (CPU).
ckpt:
	@test -n "$(RUN)" || { echo "make ckpt: set RUN=<output dir>"; exit 1; }
	@for h in $(RUN)/*.head; do \
		echo "== $$h"; \
		$(PYTHON) tools/export_ckpt.py $(MODEL_YAML) $(DATASET) "$$h" \
			"$${h%.head}.ckpt" $(if $(filter 1,$(VALIDATE)),--validate) \
			|| exit 1; \
	done

# --- Maintenance -------------------------------------------------------

format:
	@test -n "$(CLANG_FORMAT)" || { echo "clang-format not found"; exit 1; }
	$(CLANG_FORMAT) -i src/*.c src/*.h tests/*.c

install: olmo_cl
	install -D -m 755 olmo_cl $(PREFIX)/bin/olmo_cl

clean:
	rm -rf build olmo_cl

.PHONY: all test selftest headcheck test_optim check golden train ckpt \
	format install clean
