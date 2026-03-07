.PHONY: validate-amber-cipher pack-amber-cipher train-amber-cipher

validate-amber-cipher:
	python3 thornfield_case_validator.py amber_cipher.json

pack-amber-cipher: validate-amber-cipher
	python3 thornfield/trainer/tools/pack_case.py amber_cipher.json

train-amber-cipher: pack-amber-cipher
	cd thornfield/trainer && PYTHONUNBUFFERED=1 KMP_DUPLICATE_LIB_OK=TRUE KMP_USE_SHM=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python3 tools/train_single_case.py amber_cipher --paths 300 --epochs 1 --proof-paths 200 --proof-max-attempts 2000

train-amber-cipher-fast: pack-amber-cipher
	cd thornfield/trainer && PYTHONUNBUFFERED=1 KMP_DUPLICATE_LIB_OK=TRUE KMP_USE_SHM=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python3 tools/train_single_case.py amber_cipher --paths 100 --epochs 1 --proof-paths 50 --proof-max-attempts 500

colab-install:
	python3 -m pip install -r thornfield/trainer/requirements.txt

train-amber-cipher-colab: pack-amber-cipher colab-install
	cd thornfield/trainer && PYTHONUNBUFFERED=1 KMP_DUPLICATE_LIB_OK=TRUE KMP_USE_SHM=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python3 tools/train_single_case.py amber_cipher --paths 300 --epochs 20 --proof-paths 200 --proof-max-attempts 2000

train-amber-cipher-colab-gpu: pack-amber-cipher colab-install
	cd thornfield/trainer && PYTHONUNBUFFERED=1 KMP_DUPLICATE_LIB_OK=TRUE KMP_USE_SHM=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python3 tools/train_single_case.py amber_cipher --paths 300 --epochs 20 --proof-paths 200 --proof-max-attempts 2000 --device cuda

train-amber-cipher-colab-fastproof: pack-amber-cipher colab-install
	cd thornfield/trainer && PYTHONUNBUFFERED=1 KMP_DUPLICATE_LIB_OK=TRUE KMP_USE_SHM=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python3 tools/train_single_case.py amber_cipher --paths 300 --epochs 20 --proof-paths 50 --proof-max-attempts 500 --device cuda

train-amber-cipher-colab-cpu: pack-amber-cipher colab-install
	cd thornfield/trainer && PYTHONUNBUFFERED=1 KMP_DUPLICATE_LIB_OK=TRUE KMP_USE_SHM=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python3 tools/train_single_case.py amber_cipher --paths 300 --epochs 20 --proof-paths 200 --proof-max-attempts 2000 --device cpu
