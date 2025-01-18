# Base image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy dependencies and install them
COPY requirements.txt /app/
RUN pip install -r requirements.txt

# Copy the application code and model file
COPY . /app/

# Command to run the application
CMD ["python", "app.py"]

