import os
import email
from email import policy
from email.parser import BytesParser
import base64
from io import BytesIO
from PIL import Image
from PyPDF2 import PdfReader
import openai
from transformers import pipeline
import textwrap
from docx import Document

# Define the folder path
folder_path = os.path.join(os.path.dirname(__file__), 'emails')
output_file_path = os.path.join(os.path.dirname(__file__), 'output', 'content.txt')
attachment_folder = os.path.join(os.path.dirname(__file__), 'attachments')
# Ensure the attachment folder exists
os.makedirs(attachment_folder, exist_ok=True)

# Function to generate a summary using OpenAI API
# Example content to summarize
# Function to read content from the output file
def read_output_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    print({content})
    return content
    
def generate_summary(content):
    summarizer = pipeline("summarization")
    
    # Generate summary for the content
    summary = summarizer(content, max_length=150, min_length=50, do_sample=False)
    
    return summary[0]['summary_text']

# Function to chunk long content and summarize
def summarize_long_content(content, chunk_size=1024):
    # Split the content into chunks of the specified size (in tokens, 1024 characters is a rough estimate)
    chunks = textwrap.wrap(content, chunk_size)
    
    summaries = []
    for chunk in chunks:
        summary = generate_summary(chunk)
        summaries.append(summary)
    
    # Combine all the summaries
    final_summary = " ".join(summaries)
    
    return final_summary




# Function to process the .eml file and append its content to the output file
def process_eml(file_path, output_file):
    # Open and parse the .eml file
    with open(file_path, 'rb') as f:
        msg = BytesParser(policy=policy.default).parse(f)

    # Write the email headers to the output file
    output_file.write(f"Subject: {msg['subject']}\n")
    output_file.write(f"From: {msg['from']}\n")
    output_file.write(f"To: {msg['to']}\n")
    output_file.write(f"Date: {msg['date']}\n\n")

    # Extract the body of the email (text or HTML)
    if msg.is_multipart():
        for part in msg.iter_parts():
            content_type = part.get_content_type()
            print(f"part = {content_type}")

            # Handle multipart/alternative
            if content_type == 'multipart/alternative':
                # Extract the text/plain part from multipart/alternative
                for subpart in part.iter_parts():
                    if subpart.get_content_type() == 'text/plain':
                        charset = subpart.get_content_charset() or 'utf-8'
                        print('body')
                        output_file.write("Text Body :\n")
                        output_file.write(subpart.get_payload(decode=True).decode(charset) + "\n\n")
                        break  # Stop after extracting the plain text part
                
                    elif subpart.get_content_type() == 'text/html':
                        charset = subpart.get_content_charset() or 'utf-8'
                        print('html')
                        output_file.write("HTML Body:\n")
                        output_file.write(subpart.get_payload(decode=True).decode(charset) + "\n\n")
                        break
                    
            # Handle attachments
            elif 'attachment' in part.get('Content-Disposition', ''):
                file_name = part.get_filename()

                if file_name:
                    attachment_path = os.path.join(attachment_folder, file_name)
                    output_file.write(f"Attachment: {file_name}\n")
                    attachment_data = part.get_payload(decode=True)  # Decode the binary data

                    # Save the attachment as a file
                    with open(attachment_path, "wb") as attachment_file:
                        attachment_file.write(attachment_data)

                    # Attempt to display the content of the attachment based on its type
                    if file_name.lower().endswith('.txt'):
                        try:
                            output_file.write("Attachment Content (Text):\n")
                            output_file.write(attachment_data.decode('utf-8') + "\n\n")
                        except UnicodeDecodeError as e:
                            output_file.write(f"Error decoding text attachment: {str(e)}\n\n")
                    elif file_name.lower().endswith('.docx'):
                        try:
                            # Open the .docx file and extract its text
                            doc = Document(attachment_path)
                            doc_text = "\n".join([paragraph.text for paragraph in doc.paragraphs])

                            # Write the extracted text to a separate .txt file
                            txt_file_name = "outputattachment.txt"  # Replace .docx with .txt
                            txt_file_path = os.path.join(attachment_folder, txt_file_name)
                            with open(txt_file_path, "w", encoding="utf-8") as txt_file:
                                txt_file.write(doc_text)

                            # Log the saved .txt file in the main output file
                            output_file.write(f"Attachment Content (DOCX saved as TXT): {txt_file_name}\n")
                            output_file.write(f"Text content saved to: {txt_file_path}\n\n")
                        except Exception as e:
                            output_file.write(f"Error reading DOCX file: {str(e)}\n\n")
                    elif file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        output_file.write("Attachment Content (Image):\n")
                        try:
                            image = Image.open(BytesIO(attachment_data))
                            output_file.write(f"Image: {file_name} saved as {attachment_path}\n")
                        except Exception as e:
                            output_file.write(f"Error opening image: {str(e)}\n\n")
                    elif file_name.lower().endswith('.pdf'):
                        output_file.write("Attachment Content (PDF):\n")
                        try:
                            pdf_reader = PdfReader(BytesIO(attachment_data))
                            text = ""
                            for page in pdf_reader.pages:
                                text += page.extract_text()
                            output_file.write(text + "\n\n")
                        except Exception as e:
                            output_file.write(f"Error reading PDF: {str(e)}\n\n")
                    else:
                        # For other file types (binary, encrypted, etc.), output the Base64 encoded data
                        output_file.write("Attachment Content (Binary):\n")
                        base64_content = base64.b64encode(attachment_data).decode('utf-8')
                        output_file.write(base64_content + "\n\n")
       
    else:
        # Non-multipart email (plain text)
        charset = msg.get_content_charset() or 'utf-8'
        print(f"plain text: {charset}")
        output_file.write(f"Body: {msg.get_payload(decode=True).decode(charset)}\n\n")

# Open the output file in append mode
with open(output_file_path, 'a', encoding='utf-8') as output_file:
    # Loop through all .eml files in the folder and process them
    for filename in os.listdir(folder_path):
        if filename.endswith('.eml'):
            file_path = os.path.join(folder_path, filename)
            output_file.write(f"Processing file: {filename}\n")
            email_body = process_eml(file_path, output_file)
            output_file.write("\n" + "="*80 + "\n\n") 
            print(f"Content and attachments have been appended to {output_file_path}") # Add separator between emails
    content = read_output_file(output_file_path)
    final_summary = summarize_long_content(content, chunk_size=1024)  # You can adjust the chunk_size
    
    print("Final Summary:", final_summary)


            

