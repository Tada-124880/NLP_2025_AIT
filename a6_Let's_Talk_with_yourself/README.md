# RAG-Based Chatbot with FastChat-T5-3B

## Project Overview
This project involves developing a RAG (Retrieval-Augmented Generation) chatbot that answers questions related to personal information. The chatbot is designed to provide informative and well-structured responses while referencing relevant documents or sources. The chatbot is deployed as a web application with a user-friendly interface.

## Tasks and Implementation

### Task 1: Source Discovery
#### 1. Reference Documents
- The chatbot utilizes the following sources for retrieving personal information:
  - Personal data provided in the uploaded documents.
  - The personal data is about me.

#### 2. Prompt Design
- The chatbot uses a structured prompt template to ensure responses are informative and polite. The prompt is designed to handle questions regarding personal background, work experience, academic pursuits, and perspectives on technology and culture.

#### 3. Model Exploration
- The chatbot was tested with various text-generation models, including OpenAI's models and Groq’s LLaMA3-70B. However, due to resource constraints, `fastchat-t5-3b-v1.0` was chosen for deployment as it provides faster responses.

### Task 2: Analysis and Problem Solving
#### 1. List of Retriever and Generator Models Used
- **Retriever Model:** FAISS (Facebook AI Similarity Search)
- **Generator Model:** `fastchat-t5-3b-v1.0`
- **Alternative Model:** `llama-3.2-3B` (provided better responses but was resource-intensive)

#### 2. Issues Encountered
- Some responses contained unrelated information due to document retrieval mismatches.
- The generator model sometimes generated vague or overly general responses.
- Using `llama-3.2-3B` led to incomplete responses due to resource limitations but the answer form this model is better.

### Task 3: Chatbot Development (Web Application)
#### Features
- Interactive chat interface with an input box for user queries.
- Responses generated based on relevant retrieved documents.
- Answers include citations or references where applicable.

#### Example Questions the Chatbot Can Answer
1. How old are you?
2. What is your highest level of education?
3. What major or field of study did you pursue during your education?
4. How many years of work experience do you have?
5. What type of work or industry have you been involved in?
6. Can you describe your current role or job responsibilities?
7. What are your core beliefs regarding the role of technology in shaping society?
8. How do you think cultural values should influence technological advancements?
9. As a master’s student, what is the most challenging aspect of your studies so far?
10. What specific research interests or academic goals do you hope to achieve during your time as a master’s student?

#### JSON Submission Format
The chatbot is designed to generate responses in the following JSON format for submission:
```json
[
  {
    "question": "How old are you?",
    "answer": "24"
  },
  {
    "question": "What is your highest level of education?",
    "answer": "Bachelor’s Degree: Engineering, Kasetsart University"
  },
  {
    "question": "What major or field of study did you pursue during your education?",
    "answer": "Bachelor’s Degree: Engineering"
  },
  {
    "question": "How many years of work experience do you have?",
    "answer": ""
  },
  {
    "question": "What type of work or industry have you been involved in?",
    "answer": "Work Experience Total Experience: 5 years Workplace: Home / Café (Freelance Tutor) Role: Math & Physics Tutor (Secondary Education) Key Responsibilities: o Teaching Mathematics and Physics to high school students preparing for university entrance exams. o Helping students improve their grades and problem-solving skills. Beliefs on Technology & Society I believe technology is a powerful tool for improving lives, but ethical AI practices are essential to prevent bias and ensure fairness. Challenges in Master’s Studies Transitioning from engineering to data science has been challenging but exciting, as it requires"
  },
  {
    "question": "Can you describe your current role or job responsibilities?",
    "answer": "Oops, I don’t know that one!"
  },
  {
    "question": "What are your core beliefs regarding the role of technology in shaping society?",
    "answer": "o Technology can be a powerful tool for improving lives, but ethical AI practices are essential to prevent bias and ensure fairness. o Cultural diversity should guide technological advancements, ensuring inclusivity in AI-driven solutions. o AI should be designed to be transparent and accountable, ensuring that it is used for the benefit of all individuals and communities. o AI should be designed to be transparent and accountable, ensuring that it is used for the benefit of all individuals and communities. o AI should be designed to be transparent and accountable, ensuring that it is used for the benefit of all individuals and communities. o AI should be designed to"
  },
  {
    "question": "How do you think cultural values should influence technological advancements?",
    "answer": "Cultural diversity should guide technological advancements, ensuring inclusivity in AI-driven solutions. Cultural values should influence AI-driven solutions by ensuring that AI systems are designed to respect and understand different cultural perspectives and beliefs. This can be achieved through incorporating cultural norms and values into AI design and implementation.   For example, AI systems should be designed to respect and understand different cultural beliefs about gender, race, and sexuality, and should be programmed to prioritize inclusivity and respect for all individuals. Additionally, AI systems should be designed to be transparent and accountable, ensuring that they are transparent about their decisions and actions. This can help to prevent bias and ensure that AI systems "
  },
  {
    "question": "As a master’s student, what is the most challenging aspect of your studies so far?",
    "answer": "Transitioning from engineering to data science has been the most challenging aspect of my master’s studies so far."
  },
  {
    "question": "What specific research interests or academic goals do you hope to achieve during your time as a master’s student?",
    "answer": "Long-term academic and research goals as a master’s student are to contribute to NLP research, focusing on real-world AI applications, and continue learning and exploring the intersection of AI and ethics. I also aim to gain hands-on experience in industrial process and mechanical optimization. I also aim to continue learning and exploring the intersection of AI and ethics in the future."
  }
]
```