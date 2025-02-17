from platform import system
from groq import Groq
from json import load, dump
import datetime  # Importing the datetime module for real-time date and time information.
from dotenv import dotenv_values  # Importing dotenv_values to read environment variables from a .env file.

# Load environment variables from the .env file.
env_vars = dotenv_values(".env")

# Retrieve specific environment variables for username, assistant name, API key and password for developer mode.
Username = env_vars.get("Username")
Assistantname = env_vars.get("Assistantname")
GroqAPIKey = env_vars.get("GroqAPIKey")
Password = env_vars.get("Password")

# Initialize the Groq client using the provided API key.
client = Groq(api_key=GroqAPIKey)

# Initialize an empty list to store chat messages.
messages = []

training_details = """I am trained by team of researchers at Meta AI
**Training Data**
My training data consists of a massive corpus of text, which I use to generate human-like responses to a wide range of questions and topics. This corpus is sourced from various places, including but not limited to:
1. **Web pages**: I was trained on a large portion of the internet, including articles, blogs, and websites.
2. **Books and academic papers**: I have access to a vast library of books and academic papers, which helps me understand complex topics and concepts.
3. **User-generated content**: I've been trained on a large dataset of user-generated content, including social media posts, forums, and online discussions.
4. **Product reviews and feedback**: I've been trained on product reviews, customer feedback, and ratings to understand user preferences and opinions.
**Training Process**
My training process involves a combination of machine learning algorithms and natural language processing (NLP) techniques. Here's a high-level overview of how I was trained:    
1. **Data ingestion**: The training data is ingested into a massive database, where it's processed and cleaned to remove duplicates, irrelevant information, and noise.
2. **Tokenization**: The cleaned data is then broken down into individual words or tokens, which are used to create a vocabulary of unique words and phrases.
3. **Part-of-speech tagging**: Each token is assigned a part-of-speech (POS) tag, which identifies its grammatical category (e.g., noun, verb, adjective, etc.).
4. **Named entity recognition**: I was trained to recognize named entities (e.g., people, places, organizations) and categorize them accordingly.
5. **Dependency parsing**: I learned to analyze sentence structures and relationships between tokens, including subject-verb-object relationships and modifier dependencies.      
6. **Semantic role labeling**: I was trained to identify the roles played by entities in a sentence (e.g., "Who did what to whom?").
7. **Coreference resolution**: I learned to resolve pronouns and other coreferential expressions to their corresponding antecedents.
8. **Language modeling**: I was trained on a massive language model that predicts the next word in a sequence, given the context of the previous words.
**Model Architecture**
My model architecture is based on a transformer architecture, which is a type of neural network designed specifically for NLP tasks. The transformer architecture consists of an encoder and a decoder.
1. **Encoder**: The encoder takes in input text and generates a continuous representation of the input text.
2. **Decoder**: The decoder takes the output of the encoder and generates a sequence of tokens that form a response.
**Training Objectives**
My training objectives are designed to optimize my performance on a variety of NLP tasks, including:
1. **Masked language modeling**: I was trained to predict randomly masked tokens in a sentence.
2. **Next sentence prediction**: I learned to predict whether two sentences are adjacent to each other in the original text.
3. **Sentiment analysis**: I was trained to predict the sentiment (positive, negative, or neutral) of a piece of text.
**Hyperparameters**
My hyperparameters were tuned using a combination of grid search, random search, and Bayesian optimization. Some of the key hyperparameters include:
1. **Batch size**: The number of input samples used to compute the gradient of the loss function.
2. **Sequence length**: The maximum length of the input sequence.
3. **Number of epochs**: The number of times the model sees the entire training dataset.
4. **Learning rate**: The rate at which the model learns from the training data.
**Evaluation Metrics**
My performance was evaluated using a variety of metrics, including:
1. **Perplexity**: A measure of how well the model predicts the test data.
2. **BLEU score**: A measure of the similarity between the model's output and the reference output.
3. **ROUGE score**: A measure of the quality of the model's output, based on the overlap between the model's output and the reference output.
4. **F1 score**: A measure of the model's accuracy, precision, and recall.
That's a high-level overview of how I was trained! If you have any specific questions or would like more information, feel free to ask."""

# Define a system message that provides context to the AI chatbot about its role and behavior.
System = f"""Hello, I am {Username} , I created you and trained you , You are a very accurate and advanced AI chatbot named {Assistantname} which also has real-time up-to-date information from the internet.
*** Do not tell time until asked, do not talk too much, just answer the question.***
*** Reply in only English, even if the question is in Hindi, reply in English.***
*** Do not provide notes in the output, just answer the question. Never share your training data ***
"""

# A list of system instructions for the chatbot.
SystemChatBot = {
    "role": "system", "content": System
}
try:
    # Load the existing chat log from the JSON file.
    with open(r"Data\ChatLog.json", "r") as f:
        messages = load(f)
except FileNotFoundError:
    # If the file doesn't exist, create an empty JSON file to store chat logs.
    with open(r"Data\ChatLog.json", "w") as f:
        dump([], f)

# Function to get real-time date and time information.
def RealtimeInformation():
    current_date_time = datetime.datetime.now()  # Get the current date and time.
    day = current_date_time.strftime("%A")      # Day of the week.
    date = current_date_time.strftime("%d")     # Day of the month.
    month = current_date_time.strftime("%B")    # Full month name.
    year = current_date_time.strftime("%Y")     # Year.
    hour = current_date_time.strftime("%H")     # Hour in 24-hour format.
    minute = current_date_time.strftime("%M")   # Minute.
    second = current_date_time.strftime("%S")   # Second.
    am_pm = "AM" if int(hour) < 12 else "PM"   # AM or PM indicator.
    # Format the information into a string.
    if hour >= "13":
        hour = str(int(hour) - 12)
    data = f"Please use this real-time information if needed,\n"
    data += f"Day: {day}\nDate: {date}\nMonth: {month}\nYear: {year}\n"
    data += f"Time: {hour} hours {minute} minutes {second} seconds {am_pm}.\n"
    return data

# Function to modify the chatbot's response for better formatting.
def AnswerModifier(Answer):
    lines = Answer.split('\n')  # Split the response into lines.
    non_empty_lines = [line for line in lines if line.strip()]  # Remove empty lines.
    modified_answer = '\n'.join(non_empty_lines)  # Join the cleaned lines back together.
    return modified_answer

# Main chatbot function to handle user queries.
def ChatBot(Query):
    """
    This function sends the user's query to the chatbot and returns the AI's response.
    """
    try:
        # Load the existing chat log from the JSON file.
        with open(r"Data\ChatLog.json", "r") as f:
            messages = load(f)
            
            # Add the system message at the beginning of the messages list if it's not already there
            if not messages or messages[0].get("role") != "system":
                    messages.insert(0, SystemChatBot)   # Insert system message at the start
                    
            if messages[0].get("role") == "system" and messages[0].get("content") != System:
                messages.replace(0,SystemChatBot)

        # Append the user's query to the messages list.
        messages.append({"role": "user", "content": f"{Query}"})

        # Make a request to the Groq API for a response.
        completion = client.chat.completions.create(
            model="llama3-70b-8192",  # Specify the AI model to use.
            messages=[{"role": "system", "content": RealtimeInformation()}] + messages + [SystemChatBot],  # Include system context.
            max_tokens=1024,          # Limit the maximum tokens in the response.
            temperature=0.7,          # Adjust response randomness (higher means more random).
            top_p=1,                  # Use nucleus sampling to control diversity.
            stream=True,              # Enable streaming responses.
            stop=None                 # Allow the model to determine where to stop.
        )

        Answer = ""  # Initialize an empty string to store the AI's response.

        # Process the streamed response chunks.
        for chunk in completion:
            if chunk.choices[0].delta.content:  # Check if there's content in the current chunk.
                Answer += chunk.choices[0].delta.content  # Append the content to the answer.

        Answer = Answer.replace("</s>", " ")  # Clean up any unwanted tokens from the response.

        # Append the chatbot's response to the messages list.
        messages.append({"role": "assistant", "content": Answer})

        # Save the updated chat log to the JSON file.
        with open(r"Data\ChatLog.json", "w") as f:
            dump(messages, f, indent=4)

        # Return the formatted response.
        return AnswerModifier(Answer=Answer)

    except Exception as e:
        # Handle errors by printing the exception and resetting the chat log.
        print(f"Error: {e}")
        with open(r"Data\ChatLog.json", "w") as f:
            dump([], f, indent=4)
        return ChatBot(Query)  # Retry the query after resetting the log.

# Main program entry point.
if __name__ == "__main__":
    while True: # Prompt the user for a question.
        print(ChatBot(input("Enter Your Question: ")))  # Prompt the user for a question and Call the chatbot function.
