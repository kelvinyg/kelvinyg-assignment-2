all: install

# Install dependencies from requirements.txt
install:
	pip install -r requirements.txt

# Run the Flask app
run:
	python app.py

# Clean up Python __pycache__ files
clean:
	find . -name "__pycache__" -exec rm -rf {} +

# Phony targets to avoid confusion with actual files
.PHONY: install run clean