cd $(dirname ${BASH_SOURCE[0]})
cd ../..
gcloud builds submit $DIRNAME --config=gcloud/preload_instruments/cloudbuild.yaml