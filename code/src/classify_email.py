import sys
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def load_model_and_label_encoder():    
    # Load the label encoder using joblib
    label_encoder = joblib.load("label_encoder.pkl")    
    # # Load the trained model
    model = torch.load("bert_model_full.pth", weights_only=False)
    model.eval()

    return model, label_encoder

def predict_email(email_text, model, label_encoder):
    try:
        # Load the tokenizer
        tokenizer = BertTokenizer.from_pretrained("distilbert-base-uncased")

        # Tokenize the input email text
        encoding = tokenizer(
            email_text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        # Make predictions
        with torch.no_grad():
            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]
            outputs = model(input_ids, attention_mask=attention_mask)
            prediction = torch.argmax(outputs.logits, dim=-1).item()

        # Decode the prediction
        decoded_label = label_encoder.inverse_transform([prediction])[0]
        request_type, sub_request_type = decoded_label.split("::")
        return {"request_type": request_type, "sub_request_type": sub_request_type}

    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

def main():
    # Get the input passed as a command-line argument
    if len(sys.argv) > 1 and sys.argv[1].strip():
        user_input = sys.argv[1]
    else:
        # Fallback: Read content from a .txt file in a folder
        folder_path = os.path.join(os.path.dirname(__file__), "output")
        file_name = "content.txt" 
        file_path = os.path.join(folder_path, file_name)
        # Check if outputattachment.txt exists and append its content to user_input
        attachment_file_path = os.path.join(os.path.dirname(__file__), "attachments", "outputattachment.txt")
    
        try:            
            with open(file_path, 'r', encoding='utf-8') as file:
                user_input = file.read().strip()
                # print(f"Read input from file: {file_path}")
            if os.path.exists(attachment_file_path):
                try:
                    with open(attachment_file_path, 'r', encoding='utf-8') as attachment_file:
                        attachment_content = attachment_file.read().strip()
                        user_input += f"\n{attachment_content}"  # Append the content of the attachment
                        # print(f"Appended content from {attachment_file_path} to user input.")
                except Exception as e:
                    print(f"Error reading attachment file: {str(e)}")
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return
            

    # print(f"Received input: {user_input}")
    # Load the model and label encoder
    model, label_encoder = load_model_and_label_encoder()

    # Predict the email classification
    result = predict_email(user_input, model, label_encoder)
    if result:
        print(f"Prediction: {result}")
    else:
        print("Failed to classify the email.")

if __name__ == '__main__':
    main()