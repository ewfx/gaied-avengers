from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import joblib

# Custom Dataset Class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }

# Train the Model
def train_model_from_file(file):
    try:
        # Step 1: Read the Excel file into a Pandas DataFrame
        df = pd.read_excel(file)
        df.columns = df.columns.str.strip()
        print("Columns in the DataFrame:", df.columns.tolist())

        # Extract input and output columns
        texts = (df["Email Text"] + " " + df["Reasoning"]).tolist()  # Replace with the actual column name
        request_types = df["Request Type"].tolist()  # Replace with the actual column name
        sub_request_types = df["Sub Request Type"].tolist()  # Replace with the actual column name

        # Combine Request Type and Sub-Request Type into a single label
        combined_labels = [f"{req_type}::{sub_req_type}" for req_type, sub_req_type in zip(request_types, sub_request_types)]

        print(combined_labels)
        # Encode labels
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(combined_labels)       

        # Split the data into training and testing sets
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, encoded_labels, test_size=0.2, random_state=42
        )

        # Load BERT tokenizer and model
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_encoder.classes_))

        # tokenizer = BertTokenizer.from_pretrained("distilbert-base-uncased")
        # model = BertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(label_encoder.classes_))

        # Prepare datasets and dataloaders
        train_dataset = TextDataset(train_texts, train_labels, tokenizer)
        test_dataset = TextDataset(test_texts, test_labels, tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=8)

        # Set up optimizer and loss function
        optimizer = optim.AdamW(model.parameters(), lr=5e-5)
        criterion = nn.CrossEntropyLoss()

        # Train the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.train()

        for epoch in range(3):  # Train for 3 epochs
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

        # Evaluate the model
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        print(f"Accuracy: {accuracy:.2f}")
        print(correct)
        print(total)


        # Save the label encoder and model for later use
        torch.save(model, "bert_model_full.pth")
        # Save the label encoder
        joblib.dump(label_encoder, "label_encoder.pkl")

        return accuracy, predictions

    except Exception as e:
        print(f"Error: {e}")
        return None, None, None