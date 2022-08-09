.DEFAULT: help

help:
	@echo "conda-env"
	@echo "        Create conda environment 'tueopt' with dev setup"

.PHONY: conda-env

conda-env:
	@conda env create --file .conda_env.yml
