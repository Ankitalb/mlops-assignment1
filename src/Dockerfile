# Step 1: Use a base image with Python
FROM python:3.9-slim

# Step 2: Set working directory in the container
WORKDIR /app

# Step 3: Copy the current directory contents into the container
COPY . /app

# Step 4: Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Expose the port that Flask will run on
EXPOSE 5000

# Step 6: Command to run the Flask app
CMD ["python", "app.py"]
