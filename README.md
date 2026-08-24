# Support Ticket Triage

A machine-learning application that predicts the appropriate support queue and priority for customer-support tickets, while using confidence-based human review for uncertain decisions.

## Overview

This project uses natural-language processing to analyse a ticket's subject and description. Two independently trained models predict:

- The support queue responsible for handling the ticket
- The operational priority of the ticket

Predictions below a 55% confidence threshold are added to a human-review queue. Reviewers can approve or override the model's decisions, add notes and store the final routing outcome.

The project includes a Streamlit dashboard, FastAPI service, SQLite database and interactive analytics.

## Application

<img width="1002" height="519" alt="image" src="https://github.com/user-attachments/assets/f1946ec3-8cd3-44a7-bbde-685b3fb0bbd1" />


The dashboard provides three sections:

- **Submit ticket** — analyse and store a new support request
- **Review queue** — inspect and correct low-confidence predictions
- **Analytics** — monitor routing decisions, priorities and review status

### Human Review

<img width="848" height="878" alt="image" src="https://github.com/user-attachments/assets/c3ceae50-c189-4e13-963a-8c816ef62e24" />


Low-confidence predictions display:

- The predicted support queue
- Queue confidence
- Two alternative queues
- Predicted priority
- Priority confidence
- The original ticket text

A reviewer can approve or change both predictions before completing the routing decision.

### Analytics

<img width="854" height="720" alt="image" src="https://github.com/user-attachments/assets/92d541f7-e4a8-4f6e-9176-cb154ab8d31f" />


The analytics dashboard displays:

- Total submitted tickets
- Tickets awaiting review
- Human-reviewed tickets
- Automatically routed tickets
- Ticket distribution by queue
- Ticket distribution by priority
- Recent prediction history

## Dataset

The project uses the [Customer Support Tickets dataset](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) from Hugging Face.

The original dataset contains 61,765 synthetic support tickets in English and German.

During preprocessing, I:

1. Selected English-language tickets.
2. Kept the ten main support queues.
3. Combined each ticket's subject and description.
4. Removed missing and empty ticket text.
5. Removed 4,513 exact duplicate tickets.
6. Selected the `low`, `medium` and `high` priority classes.

The resulting dataset contains 23,747 tickets.

### Supported queues

- Billing and Payments
- Customer Service
- General Inquiry
- Human Resources
- IT Support
- Product Support
- Returns and Exchanges
- Sales and Pre-Sales
- Service Outages and Maintenance
- Technical Support

## Model

The application uses two separate text-classification pipelines:

1. A routing model that predicts the support queue
2. A priority model that predicts ticket priority

Each pipeline uses:

- TF-IDF text features
- Unigrams and bigrams
- English stop-word removal
- A maximum vocabulary of 50,000 features
- Logistic regression
- Balanced class weights

The data was divided into an 80% training split and a 20% held-out test split using a fixed random seed.

## Results

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Queue routing | 0.508 | 0.513 |
| Priority prediction | 0.624 | 0.615 |

Macro F1 is included because the dataset is imbalanced, with substantially more examples for some queues than others.

The routing model performs particularly well for Billing and Payments, while overlapping categories such as Technical Support, IT Support and Product Support are more difficult to distinguish.

## Error Analysis

I analysed both models by examining:

- The most common class confusions
- Per-class precision, recall and F1 scores
- High-confidence incorrect predictions
- Examples where the supplied labels appeared ambiguous

Common routing confusions included:

- Technical Support and IT Support
- Product Support and Technical Support
- Customer Service and Product Support
- Technical Support and Customer Service

Some tickets labelled as Product Support or Technical Support explicitly described service outages. The model often predicted Service Outages and Maintenance for these examples, suggesting that some apparent errors were semantically reasonable.

This analysis influenced the application design: the model acts as a decision-support tool instead of making every routing decision automatically.

## Confidence-Based Review

A ticket requires human review when either its queue confidence or priority confidence is below 55%.

For tickets requiring review, the system stores:

- The original ticket
- Predicted queue and confidence
- Two alternative queue predictions
- Predicted priority and confidence
- Review status
- Final human-selected queue and priority
- Reviewer notes
- Creation and review timestamps

## FastAPI Service

The project exposes its prediction and review functionality through a REST API.

### Endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Check API availability |
| `POST` | `/tickets` | Analyse and save a ticket |
| `GET` | `/tickets` | Retrieve submitted tickets |
| `GET` | `/tickets/{ticket_id}` | Retrieve one ticket |
| `PUT` | `/tickets/{ticket_id}/review` | Complete human review |

Interactive API documentation is available at `/docs` while the API is running.

## Installation

Clone the repository:

```bash
git clone https://github.com/haidencz/Support-Ticket-Triage.git
cd Support-Ticket-Triage
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Running the Dashboard

Start the Streamlit application from the main project directory:

```bash
streamlit run app.py
```

Then open the local address shown in the terminal.

## Running the API

Start the FastAPI development server:

```bash
uvicorn src.api:app --reload
```

Open the interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

## Reproducing the Models

Inspect the original dataset:

```bash
python src/inspect_dataset.py
```

Prepare the filtered dataset:

```bash
python src/prepare_data.py
```

Train and evaluate both models:

```bash
python src/train_models.py
```

Run the error analysis:

```bash
python src/analyse_errors.py
```

Test a single prediction:

```bash
python src/predict.py
```

## Project Structure

```text
Support-Ticket-Triage/
├── assets/
│   ├── analytics-dashboard.png
│   ├── review-queue.png
│   └── submit-ticket.png
├── data/
│   └── tickets.db
├── models/
│   ├── priority_model.joblib
│   └── queue_model.joblib
├── src/
│   ├── analyse_errors.py
│   ├── api.py
│   ├── database.py
│   ├── inspect_dataset.py
│   ├── predict.py
│   ├── prepare_data.py
│   └── train_models.py
├── app.py
├── requirements.txt
└── README.md
```

The downloaded dataset, processed dataset and local SQLite database are excluded from version control.

## Limitations

- The dataset contains synthetic rather than genuine customer tickets.
- Several support queues have overlapping responsibilities.
- Some supplied labels appear ambiguous or inconsistent.
- The models only support English-language tickets.
- The current priority model only predicts low, medium and high.
- Confidence scores should not be interpreted as guaranteed probabilities.
- The 55% review threshold is an initial operational rule rather than a formally optimised threshold.
- SQLite data stored on Streamlit Cloud may be reset when the application restarts or is redeployed.
- This project is intended as a portfolio demonstration and not as a production customer-support system.

## Dataset Attribution

Customer Support Tickets dataset by Tobi Bueck:

[https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets)

Dataset licence: CC BY-NC 4.0.
