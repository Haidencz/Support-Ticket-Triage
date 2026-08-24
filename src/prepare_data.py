from pathlib import Path

from datasets import load_dataset


DATASET_NAME = "Tobi-Bueck/customer-support-tickets"
OUTPUT_PATH = Path("data/processed_tickets.csv")

CORE_QUEUES = [
    "Technical Support",
    "Product Support",
    "Customer Service",
    "IT Support",
    "Billing and Payments",
    "Returns and Exchanges",
    "Service Outages and Maintenance",
    "Sales and Pre-Sales",
    "Human Resources",
    "General Inquiry",
]


print("Loading dataset...")

dataset = load_dataset(
    DATASET_NAME,
    split="train"
)

tickets = dataset.to_pandas()

print(f"Original rows: {len(tickets)}")

# Keep English-language tickets only.
tickets = tickets[tickets["language"] == "en"].copy()

print(f"English rows: {len(tickets)}")

# These fields are essential for our two prediction tasks.
tickets = tickets.dropna(
    subset=["body", "queue", "priority"]
)

# Keep the ten main support departments.
tickets = tickets[
    tickets["queue"].isin(CORE_QUEUES)
].copy()

print(f"Rows in core queues: {len(tickets)}")

# Some tickets do not have a subject, but nearly all have a body.
tickets["subject"] = tickets["subject"].fillna("")

# Combine the subject and body into one model input.
tickets["text"] = (
    tickets["subject"].str.strip()
    + "\n\n"
    + tickets["body"].str.strip()
).str.strip()

# Remove empty and exactly duplicated ticket text.
tickets = tickets[tickets["text"].str.len() > 0].copy()

duplicate_count = tickets.duplicated(
    subset=["text"]
).sum()

print(f"Exact duplicate tickets found: {duplicate_count}")

tickets = tickets.drop_duplicates(
    subset=["text"]
).copy()

# Keep only fields required by the models and application.
tickets = tickets[
    ["text", "queue", "priority"]
].reset_index(drop=True)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

tickets.to_csv(
    OUTPUT_PATH,
    index=False
)

print(f"\nProcessed rows: {len(tickets)}")
print(f"Saved dataset to: {OUTPUT_PATH}")

print("\nQueue distribution:")
print(tickets["queue"].value_counts())

print("\nPriority distribution:")
print(tickets["priority"].value_counts())