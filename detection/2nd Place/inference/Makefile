.PHONY: build build-full pull pack-benchmark pack-submission repack-submission repack-example test-submission

# ================================================================================================
# Settings
# ================================================================================================
TAG = latest
LOCAL_TAG = local

TASK = object-detection
REPO = spacecraft-pose-${TASK}
REGISTRY_IMAGE = spacecraftpose.azurecr.io/${REPO}:${TAG}
LOCAL_IMAGE = ${REPO}:${LOCAL_TAG}
CONTAINER_NAME = spacecraft-pose-${TASK}

# if not TTY (for example GithubActions CI) no interactive tty commands for docker
ifneq (true, ${GITHUB_ACTIONS_NO_TTY})
TTY_ARGS = -it
endif

# To run a submission, use local version if that exists; otherwise, use official version
# setting SUBMISSION_IMAGE as an environment variable will override the image
SUBMISSION_IMAGE ?= $(shell docker images -q ${LOCAL_IMAGE})
ifeq (,${SUBMISSION_IMAGE})
SUBMISSION_IMAGE := $(shell docker images -q ${REGISTRY_IMAGE})
endif

# Give write access to the submission folder to everyone so Docker user can write when mounted
_submission_write_perms:
	mkdir -p submission/
	chmod -R 0777 submission/

# ================================================================================================
# Commands for building the container if you are changing the requirements
# ================================================================================================

## Builds the container locally
build:
	docker build -t ${LOCAL_IMAGE} runtime

build-full:
	docker build --no-cache -t ${LOCAL_IMAGE} runtime

## Ensures that your locally built container can import all the Python packages successfully when it runs
test-container: build _submission_write_perms
	docker run \
		${TTY_ARGS} \
		--mount type=bind,source="$(shell pwd)"/runtime/tests,target=/tests,readonly \
		${LOCAL_IMAGE} \
		pytest tests/test_packages.py

## Start your locally built container and open a bash shell within the running container; same as submission setup except has network access
interact-container: build _submission_write_perms
	docker run \
		--mount type=bind,source="$(shell pwd)"/data,target=/code_execution/data,readonly \
		--mount type=bind,source="$(shell pwd)"/submission,target=/code_execution/submission \
		--shm-size 8g \
		-it \
		--entrypoint /bin/bash \
		${LOCAL_IMAGE}

## Pulls the official container from Azure Container Registry
pull:
	docker pull ${REGISTRY_IMAGE}

## Creates a submission/submission.zip file from the source code in examples_src
pack-example:
# Don't overwrite so no work is lost accidentally
ifneq (,$(wildcard ./submission/submission.zip))
	$(error You already have a submission/submission.zip file. Rename or remove that file (e.g., rm submission/submission.zip).)
endif
	mkdir -p submission/
	cd example_src; zip -r ../submission/submission.zip ./*

## Creates a submission/submission.zip file from the source code in example_benchmark
pack-benchmark:
# Don't overwrite so no work is lost accidentally
ifneq (,$(wildcard ./submission/submission.zip))
	$(error You already have a submission/submission.zip file. Rename or remove that file (e.g., rm submission/submission.zip).)
endif
	mkdir -p submission/
	cd example_benchmark; zip -r ../submission/submission.zip ./*

## Creates a submission/submission.zip file from the source code in submission_src
pack-submission:
# Don't overwrite so no work is lost accidentally
ifneq (,$(wildcard ./submission/submission.zip))
	$(error You already have a submission/submission.zip file. Rename or remove that file (e.g., rm submission/submission.zip).)
endif
	mkdir -p submission/
	cd submission_src; zip -r ../submission/submission.zip ./*

repack-submission: clean pack-submission

repack-example: clean pack-example

## Runs container using code from `submission/submission.zip` and data from `data/`
test-submission: _submission_write_perms
# if submission file does not exist
ifeq (,$(wildcard ./submission/submission.zip))
	$(error To test your submission, you must first put a "submission.zip" file in the "submission" folder. \
	  If you want to use the benchmark, you can run `make pack-benchmark` first)
endif
	docker run \
		${TTY_ARGS} \
		--network none \
		--mount type=bind,source="$(shell pwd)"/data,target=/code_execution/data,readonly \
		--mount type=bind,source="$(shell pwd)"/submission,target=/code_execution/submission \
		--shm-size 8g \
		--name ${CONTAINER_NAME} \
		--rm \
		${SUBMISSION_IMAGE}

## Delete temporary Python cache and bytecode files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf ./submission/*
