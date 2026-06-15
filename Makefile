# argus container build + multi-platform / CPU-only test harness.
#
#   make buildx-setup                      # one-time: QEMU binfmt + a buildx builder
#   make build-cpu                         # CPU image for amd64+arm64 (built, not loaded)
#   make build-amd64 / build-arm64         # single-arch CPU image, loaded into the local daemon
#   make build-cuda                        # CUDA image (amd64), loaded
#   make smoke-amd64 / smoke-arm64 / smoke-cuda   # CPU-only smoke test (arm64 via QEMU)
#   make test-amd64 / test-arm64           # full synthetic pytest suite, in-container
#   make ci                                # the whole matrix
#
# CUDA_TORCH_INDEX defaults to a cu124 wheel index — adjust to match the torch build you want.

IMAGE       ?= argus
# torch 2.12.0 CUDA wheels are published for cu126 and cu130 (not cu124). cu126 is the safer
# default; override with `make build-cuda CUDA_TORCH_INDEX=.../whl/cu130` for newer CUDA.
CUDA_TORCH_INDEX ?= https://download.pytorch.org/whl/cu126
SMOKE_RUN    = docker run --rm --entrypoint bash -v "$(PWD)/scripts:/smoke:ro"

.PHONY: buildx-setup build-cpu build-amd64 build-arm64 build-cuda \
        smoke-amd64 smoke-arm64 smoke-cuda test-amd64 test-arm64 ci

buildx-setup:
	docker run --privileged --rm tonistiigi/binfmt --install all
	docker buildx create --name argusbuilder --use --bootstrap || docker buildx use argusbuilder

# Verifies wheels resolve/compile for BOTH arches. (buildx can't --load a multi-arch image;
# use build-amd64 / build-arm64 to load a single arch for running.)
build-cpu:
	docker buildx build --target runtime --platform linux/amd64,linux/arm64 -t $(IMAGE):cpu .

build-amd64:
	docker buildx build --target runtime --platform linux/amd64 -t $(IMAGE):amd64 --load .

build-arm64:
	docker buildx build --target runtime --platform linux/arm64 -t $(IMAGE):arm64 --load .

build-cuda:
	docker buildx build --target runtime --platform linux/amd64 \
		--build-arg TORCH_INDEX=$(CUDA_TORCH_INDEX) \
		-t $(IMAGE):cuda --load .

smoke-amd64: build-amd64
	$(SMOKE_RUN) $(IMAGE):amd64 /smoke/smoke.sh

smoke-arm64: build-arm64
	$(SMOKE_RUN) $(IMAGE):arm64 /smoke/smoke.sh

# CUDA image on a GPU-less host: device defaults to auto → CPU fallback must not crash.
smoke-cuda: build-cuda
	$(SMOKE_RUN) $(IMAGE):cuda /smoke/smoke.sh

test-amd64:
	docker buildx build --platform linux/amd64 --target test -t $(IMAGE):test-amd64 --load .
	docker run --rm $(IMAGE):test-amd64

test-arm64:
	docker buildx build --platform linux/arm64 --target test -t $(IMAGE):test-arm64 --load .
	docker run --rm $(IMAGE):test-arm64

ci: buildx-setup build-cpu smoke-amd64 test-amd64 smoke-arm64 test-arm64 build-cuda smoke-cuda
