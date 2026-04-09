PYTHONPATH := ../thornfield/trainer:..

eval-all:
	PYTHONPATH=$(PYTHONPATH) python3 -m pytest tests/ -v --tb=short

eval-overfitting:
	PYTHONPATH=$(PYTHONPATH) python3 -m pytest tests/test_overfitting.py -v --tb=short

eval-convergence:
	PYTHONPATH=$(PYTHONPATH) python3 -m pytest tests/test_convergence.py -v --tb=short

eval-diversity:
	PYTHONPATH=$(PYTHONPATH) python3 -m pytest tests/test_diversity.py -v --tb=short

eval-chronology:
	PYTHONPATH=$(PYTHONPATH) python3 -m pytest tests/test_chronology.py -v --tb=short
