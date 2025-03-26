# 🚀 Intelligent Content Classification and Extraction (ICCE)

## 📌 Table of Contents
- [Introduction](#introduction)
- [Demo](#demo)
- [Inspiration](#inspiration)
- [What It Does](#what-it-does)
- [How We Built It](#how-we-built-it)
- [Challenges We Faced](#challenges-we-faced)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)
- [Team](#team)

---

## 🎯 Introduction
In todays fast-paced digital world, managing and categorizing incoming emails efficiently is crucial for business. This project aims to automate the extraction of request types and sub-request types from email, especially those with attachments, and is also capable of extracting required data from the inbound email based on configured values. This streamlines the workflow, improves efficiency, and reduced turnaround time. We have identified a pre-trained generative AI model, which will be downloaded and fine-tuned within our environment for this classification.

## 🎥 Demo
🔗 [Live Demo](#) (if applicable)  
📹 [Video Demo](#) (if applicable)  
🖼️ Screenshots:

![Screenshot 1](link-to-image)

## 💡 Inspiration
The inspiration for this project came from the need to handle significant volumes of servicing requests through emails by commercial bank service departments. Manually triaging, sorting, categorizing and extracting key attributes from these emails is time-consuming and prone to errors. Automating this process can bring effficiency, accuracy, save time and reduce human error.

## ⚙️ What It Does
The system processes incoming email request with or without attachments, summarizes the content of both emails and attachments into a single content and uses fine-tuned generative AI models (which is already trained with our input dataset) to classify the request type and sub-request type.It also provides a confidence score and given a try to extracts the required data from the inbound content based on preconfigured data fields based on their priority level settings against each fields. This process has been exposed as an API with different endpoints, making it flexible for easy consumption and integeration from any platform.

## 🛠️ How We Built It
We have built this sytem with different endpoints as described below:

A. Orchestrate Email Endpoint

This layer is mainly focused on classifying and extracting the required information from emails, which will accept *.eml files (with or without attachments). once the email is received, the process will filter the email, separate the attachments and start reading them. If the attachment contains extensive details, it will be split into different chunks and we use the Hugging face summarizer pipeline to generate a summary of the content. Based on this generated summary, a fine-tuned model named "bert-base-uncased" will be called for request and subrequest classification along with a confidence score. once classified , we extract the required fields using our pre-configured master template data which uploaded by the end user, and the response will be provided to the end user.

B. TrainModel Endpoint

We have exposed the trainmodel end point to provide the flexibility for developers to upload training data at any point in time, which will be used for fine-tuning. This data is captured in an Excel file, which will be stored in a directory and used for training our model

C. MasterFeed Endpoint

We had a plan to expose another end point call MasterFeed to provide additional flexibility for developers to upload the extraction field along with their priority. This this will be based on the request type and the sub type they are looking for in the email.
## 🚧 Challenges We Faced
Email Parsing: Handling different attachments and parsing them into a stream content.

Scalability: Making the system scalable to handle a large volume of emails efficiently

Model identification: Identifying a suitable free pre-trained model that does not expose any data to the outside world, and is capable of being fine tune to provide the exact classification, we need.

Test Data: Preparing test data was quite challenging Ask me did not have enough information about the current system and how it classifies the email.

## 🏃 How to Run
1. Clone the repository

   `https://github.com/ewfx/gaied-avengers.git`

2. Install dependencies
   
   `pip install flask pandas openai scikit-learn PyPDF2 Pillow transformers torch joblib docx`

3. Run the project
   
   ```
   #Navigate to project folder
   python app.py
   ```
   

## 🏗️ Tech Stack
- 🔹 Frontend: Python
- 🔹 Other:  Hugging face (summarization) , Gen AI [bert-base-uncased]

## 👥 Team

Ramkumar Palraj

R, Yokambika

Sekar,Karthikeyan

Sp, Kishorekumar

Velayutham g, Shunmuga

