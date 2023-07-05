.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3 raw_data

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = mlrfproject
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Commands reminder
tldr:
	@printf "\033[1;32m   - %s\033[0m\n      \033[1;31m%s\033[0m\n\n" "Télécharge le jeu de données." "make raw_data"
	@printf "\033[1;32m   - %s\033[0m\n      \033[1;31m%s\033[0m\n\n" "Crée des fichiers csv à partir des données brutes." "make data"
	@printf "\033[1;32m   - %s\033[0m\n      \033[1;31m%s\033[0m\n\n" "Crée des datasets contenant les features extraites." "make {hog|brief}"
	@printf "\033[1;32m   - %s\033[0m\n      \033[1;31m%s\033[0m\n\n" "Entraine les modèles sur les features extraites et affiche les performances sur l'ensemble de test." "make test"

## Install Python Dependencies
requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Download raw data
raw_data:
	@printf "\033[1;34m📥 Downloading cifar-10 dataset...\033[0m\n"
	@wget -P data/raw/ http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
	@printf "\033[1;32m📦 Unpacking cifar-10 dataset...\033[0m\n"
	@tar -xzf data/raw/cifar-10-python.tar.gz -C data/raw/
	@printf "\033[1;31m🗑️ Removing archive...\033[0m\n"
	@rm data/raw/cifar-10-python.tar.gz
	@printf "\033[1;33m🔁 Renaming file...\033[0m\n"
	@mv data/raw/cifar-10-batches-py/test_batch data/raw/cifar-10-batches-py/data_batch_test
	@printf "\033[1;32m✅ Dataset ready.\033[0m\n"

## Delete raw data
clean_raw_data:
	@printf "\033[1;31m🗑️ Removing raw data...\033[0m\n"
	@rm -rf data/raw/cifar-10-batches-py
	@printf "\033[1;32m✅ Done.\033[0m\n"

## Make Dataset
data:
	@printf "\033[1;32m📦 Creating datasets...\033[0m\n"
	$(PYTHON_INTERPRETER) src/data/make_dataset.py data/raw/cifar-10-batches-py data/processed
	@printf "\033[1;33m🔁 Renaming file...\033[0m\n"
	@mv data/processed/data_batch_0.csv data/processed/data_batch_test.csv
	@printf "\033[1;32m✅ Done.\033[0m\n"

## Delete Dataset
clean_data:
	@printf "\033[1;31m🗑️ Removing data...\033[0m\n"
	@rm data/processed/data_batch_*.csv
	@printf "\033[1;32m✅ Done.\033[0m\n"

clean_all_data: clean_raw_data clean_data
	@printf "\033[1;31m🗑️ Removing all data...\033[0m\n"
	@printf "\033[1;32m✅ Done.\033[0m\n"

## Generate datasets with hog descriptors
hog:
	@printf "\033[1;33m🔁 Extracting hog features...\033[0m\n"
	$(PYTHON_INTERPRETER) src/features/hog_features.py
	@printf "\033[1;32m✅ Done.\033[0m\n"

## Generate datasets with brief descriptors
brief:
	@printf "\033[1;33m🔁 Extracting brief features...\033[0m\n"
	$(PYTHON_INTERPRETER) src/features/brief_features.py
	@printf "\033[1;32m✅ Done.\033[0m\n"

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src

## Upload Data to S3
sync_data_to_s3:
ifeq (default,$(PROFILE))
	aws s3 sync data/ s3://$(BUCKET)/data/
else
	aws s3 sync data/ s3://$(BUCKET)/data/ --profile $(PROFILE)
endif

## Download Data from S3
sync_data_from_s3:
ifeq (default,$(PROFILE))
	aws s3 sync s3://$(BUCKET)/data/ data/
else
	aws s3 sync s3://$(BUCKET)/data/ data/ --profile $(PROFILE)
endif

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already installed.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
