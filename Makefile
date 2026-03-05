.PHONY: validate-amber-cipher pack-amber-cipher train-amber-cipher

validate-amber-cipher:
	python3 thornfield_case_validator.py amber_cipher.json

pack-amber-cipher: validate-amber-cipher
	python3 thornfield/trainer/tools/pack_case.py amber_cipher.json

train-amber-cipher: pack-amber-cipher
	PYTHONUNBUFFERED=1 KMP_DUPLICATE_LIB_OK=TRUE KMP_USE_SHM=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=thornfield/trainer python3 thornfield/trainer/tools/train_single_case.py amber_cipher --paths 300 --epochs 1
