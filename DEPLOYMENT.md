# RiceGenixAI Cloud Deployment

This project is ready to deploy as the same Streamlit app, with the same look and features, without keeping the local PC turned on.

## What was prepared

- same Streamlit UI and CSS
- same settings sidebar and weather API usage
- same PDF export flow
- same disease model loading
- same graphs
- added `st.camera_input` for direct mobile camera capture
- Docker deployment file for cloud hosting
- Render deployment manifest

## Recommended hosting options

1. Render
   - Create a new Web Service from this repository.
   - Render will detect [render.yaml](D:\RiceGenixAI\render.yaml) and use the included [Dockerfile](D:\RiceGenixAI\Dockerfile).

2. Any Docker-based cloud
   - Deploy the repo using the included [Dockerfile](D:\RiceGenixAI\Dockerfile).

## Why hosting is not finished in this session

The machine does not have any authenticated cloud CLI or account access available, so I cannot push and publish the app to a cloud provider from here without credentials.
