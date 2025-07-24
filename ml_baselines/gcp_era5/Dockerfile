FROM python:3.11-slim

WORKDIR /app
# Create an output directory
RUN mkdir -p /app/output

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy your script and any needed files
COPY get_gcp_era5.py ./

# Set the default command to run your script
ENTRYPOINT ["python", "get_gcp_era5.py"]
CMD ["MHD", "1978"]