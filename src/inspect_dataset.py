from datasets import load_dataset


DATASET_NAME = "Tobi-Bueck/customer-support-tickets"

print("Downloading dataset...")

dataset = load_dataset(
    DATASET_NAME,
    split="train"
)

tickets = dataset.to_pandas()

print("\nDataset shape:")
print(tickets.shape)

print("\nColumns:")
for column in tickets.columns:
    print(f"- {column}")

print("\nMissing values:")
print(tickets.isna().sum())

print("\nLanguage distribution:")
print(tickets["language"].value_counts())

print("\nTicket type distribution:")
print(tickets["type"].value_counts())

print("\nPriority distribution:")
print(tickets["priority"].value_counts())

print("\nTop 15 routing queues:")
print(tickets["queue"].value_counts().head(15))

print("\nExample ticket:")
example = tickets.iloc[0]

print("Subject:", example["subject"])
print("Body:", example["body"][:500])
print("Type:", example["type"])
print("Queue:", example["queue"])
print("Priority:", example["priority"])
print("Language:", example["language"])