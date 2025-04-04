# heart-disease-prediction
# Cardio Monitor
A web app to predict heart disease risk.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run pipeline: `python pipeline.py`
3. Start web app: `python webapp/app.py`

## Deployment
- Push to `main` triggers Azure ML deployment via GitHub Actions.
- Set `AZURE_CREDENTIALS` in GitHub Secrets.

########################

Syntax for secrets.AZURE_CREDENTIALS
{
    "clientSecret":  "<>",
    "subscriptionId":  "<>",
    "tenantId":  "<>",
    "clientId":  "<>"
}
#################

folder structure
├── .github
│   └── workflows
│       └── azure_deploy.yml    # GitHub Actions for Azure ML deployment
├── data
│   ├── heart.csv              # Your sample dataset
│   ├── train.csv              # Training data (generated)
│   └── test.csv               # Testing data (generated)
├── models
│   └── cardio_model.pkl       # Trained model (generated)
├── webapp
│   ├── app.py                 # Flask web app
│   ├── static
│   │   └── style.css          # Basic CSS
│   └── templates
│       ├── index.html         # Input form
│       └── result.html        # Prediction result
├── pipeline.py                # Main MLOps pipeline
├── requirements.txt           # Dependencies
├── README.md                  # Project documentation
└── .gitignore                 # Git ignore file