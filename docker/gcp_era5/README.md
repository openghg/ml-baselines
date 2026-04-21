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

- From **the repository root**, build the image for amd64 (or other relevant architecture) and push to Artifact Registry:

```
docker buildx build --platform linux/amd64 -t us-central1-docker.pkg.dev/my-project/my-repo/gcp-era5:latest --push -f docker/gcp_era5/Dockerfile .
```
(replace ```us-central1```, ```my-project``` and ```my-repo``` with desired server, project and Artifact Repository)


### Running

Set up GCP Cloud Run **Job** using appropriate resources (X Gb).

Run using:

```
gcloud run jobs execute gcp-era5 --region=us-central1 --args='MHD,1978'
```

Or in a loop:

```
for y in {2009..2010}; do gcloud run jobs execute gcp-era5 --region=us-central1 --args="THD,$y"; done
```

It takes a few seconds to provision each job, so you can submit them in parallel, if needed. The ```sleep``` command is there to prevent request limits being reached (although, it still seems easy to reach them, for reasons that I haven't investigated):

```
for y in {2009..2010}; do gcloud run jobs execute gcp-era5 --region=us-central1 --async --args="THD,$y"; sleep 0.5;  done
```

### Downloading the data from the bucket

To download the data from the bucket, you can use:

```gcloud storage rsync gs://<BUCKET_NAME>/ /LOCAL/PATH/```