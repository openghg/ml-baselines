FROM python:3.11-slim

WORKDIR /app
# Create an output directory
RUN mkdir -p /app/output

# Install dependencies
COPY docker/gcp_era5/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy your script and any needed files
COPY ml_baselines ./ml_baselines

# Set the default command to run your script
ENTRYPOINT ["python", "-m", "ml_baselines.met_retrieve.get_gcp_era5"]
CMD ["MHD", "1978"]