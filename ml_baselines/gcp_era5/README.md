# Instructions for building and running Docker image on Google Cloud Platform

### Setup
1. ```gcloud``` is installed locally
2. Docker desktop is installed locally
3. ```buildx``` is installed to allow build for different architectures:

```
docker buildx create --use
```

4. Authentication:
   1. ```gcloud auth configure-docker```
5. Enable Cloud Run on GCP
6. Set up a bucket to store the output on GCP
7. Enable Artifact Repository on GCP

### Building and deploying the image

- From within the gcp_era5 folder, build the image for amd64 (or other relevant architecture) and push to Artifact Registry:

```
docker buildx build --platform linux/amd64 -t us-central1-docker.pkg.dev/my-project/my-repo/gcp-era5:latest --push .
```
(replace ```us-central1```, ```my-project``` and ```my-repo``` with desired server, project and Artifact Repository)


### Running

Set up GCP Cloud Run **Job** using appropriate resources (X Gb).

Run using:

gcloud run jobs execute gcp-era5 --region=us-central1 --args='MHD,1978'

Or in a loop:

for year in {1979..2024}; do
   gcloud run jobs execute gcp-era5 --region=us-central1 --args='MHD,'"$year"
done