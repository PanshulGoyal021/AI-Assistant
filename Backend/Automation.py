# Import required libraries
from AppOpener import close, open as appopen #Import functions to open and close apps.
from webbrowser import open as webopen # Import web browser functionality.
from pywhatkit import search, playonyt # Import functions for Google search and YouTube playback.
from dotenv import dotenv_values # Import dotenv to manage environment variables.
from bs4 import BeautifulSoup # Import BeautifulSoup for parsing HTML content.
from rich import print # Import rich for styled console output.
from groq import Groq # Import Groq for AI chat functionalities.
import webbrowser # Import webbrowser for opening URLS.
import subprocess # Import subprocess for interacting with the system. 
import requests # Import requests for making HTTP requests.
import keyboard # Import keyboard for keyboard-related actions.
import asyncio # Import asyncio for asynchronous programming.
import os # Import os for operating system functionalities.

# Load environment variables from the .env file.
env_vars = dotenv_values(".env")
GroqAPIKey = env_vars.get("GroqAPIKey")
# Retrieve the Groq API key.

# Define CSS classes for parsing specific elements in HTML content.
classes = [
    "zCubwf", "hgKELC", "LTKOO SY7ric", "ZOLcW", "gsrt vk_bk FzvWSb YwPhnf", "pclqee",
    "tw-Data-text tw-text-small tw-ta", "IZ6rdc", "05uR6d LTKOO", "vlzY6d", 
    "webanswers-webanswers_table_webanswers-table", "dDoNo ikb4Bb gsrt", 
    "sXLa0e", "LWkfke", "VQF4g", "qv3Wpe", "kno-rdesc", "SPZz6b"
]

# Define a user-agent for making web requests.
useragent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36'

# Initialize the Groq client with the API key.
client = Groq(api_key=GroqAPIKey)

# Predefined professional responses for user interactions.
professional_responses = [
    "Your satisfaction is my top priority; feel free to reach out if there's anything else I can help you with.",
    "I'm at your service for any additional questions or support you may need—don't hesitate to ask."
]

# List to store chatbot messages.
messages = []

# System message to provide context to the chatbot.
SystemChatBot = [
    {"role": "system", "content": f"Hello, I am {os.environ['Username']}, You're a content writer. You have to write content like letters, codes, applications, essays, notes, songs, poems etc."}
]

# Dictionary to store file extensions for different programming languages.
file_extensions = {
    "python": ".py",
    "javascript": ".js",
    "java": ".java",
    "html": ".html",
    "typescript": ".ts",
    "css": ".css",
    "php": ".php",
    "swift": ".swift",
    "kotlin": ".kt",
    "ruby": ".rb",
    "go": ".go",
    "rust": ".rs",
    "dart": ".dart",
    "r": ".r",
    "perl": ".pl",
    "lua": ".lua",
    "shell script": ".sh",
    "objective-c": ".m",
    "matlab": ".m",
    "sql": ".sql",
    "scala": ".scala",
    "haskell": ".hs",
    "pascal": ".pas",
    "vb.net": ".vb",
    "f#": ".fs",
    "groovy": ".groovy",
    "assembly": ".asm",
    "c" : ".c",
    "c++": ".cpp",
    "c#": ".cs",
}

# Function to perform a Google search.
def GoogleSearch(Topic):
    search(Topic)  # Use pywhatkit's search function to perform a Google search.
    return True  # Indicate success.

# Function to generate content using AI and save it to a file.
def Content(Topic):
    # Nested function to open a file in Notepad.
    def OpenNotepad(File):
        default_text_editor = 'notepad.exe'  # Default text editor.
        subprocess.Popen([default_text_editor, File])  # Open the file in Notepad.

    def ContentWriterAI(prompt):
        messages.append({"role": "user", "content": f"{prompt}"})

        completion = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=SystemChatBot + messages,
            max_tokens=2048,
            temperature=0.7,
            top_p=1,  # Use nucleus sampling for response diversity.
            stream=True,  # Enable streaming response.
            stop=None  # Allow the model to determine stopping conditions.
        )

        Answer = ""  # Initialize an empty string for the response.
        # Process streamed response chunks.
        for chunk in completion:
            if chunk.choices[0].delta.content:
                # Check for content in the current chunk.
                Answer += chunk.choices[0].delta.content  # Append the content to the answer.

        Answer = Answer.replace("</s>", "")  # Remove unwanted tokens from the response.
        messages.append({"role": "assistant", "content": Answer})  # Add the AI's response to messages.
        return Answer

    Topic = Topic.replace("Content", "")  # Remove "Content" from the topic.
    ContentByAI = ContentWriterAI(Topic)  # Generate content using AI.

    # Save the generated content to a file.
    if "code" in Topic:
        Topic = Topic.split(" ")
        language = next((lang for lang in file_extensions if lang in Topic), "python")
        extension = file_extensions[language]  # Retrieve the file extension based on the language.
        Topic = " ".join(Topic)
        file = rf"data\{Topic.lower().replace(' ', '_')}{extension}"
        ContentByAI = ContentByAI.split("\n")
        
        if f"```{language}" in ContentByAI:
            Starting = int(ContentByAI.index(f"```{language}") + 1)
        if "```" in ContentByAI:
            Ending = int(ContentByAI.index("```"))
        ContentByAI  = ContentByAI[Starting:Ending]  
        if ContentByAI:
            ContentByAI = "\n".join(ContentByAI)
        else:
            print("ContentByAI not found")
        
    else:
        file = rf"Data\{Topic.lower().replace(' ', '_')}.txt"  # Construct the file name.
        
    with open(file, "w", encoding="utf-8") as f:
        f.write(ContentByAI)  # Write the content to the file.
        f.close()


    OpenNotepad(rf"Data\{Topic.lower().replace(' ', '')}.txt")  # Open the file in Notepad.
    return True  # Indicate success.

# Function to search for a topic on YouTube.
def YouTubeSearch(Topic):
    Url4Search = f"https://www.youtube.com/results?search_query={Topic}"  # Construct the YouTube search URL.
    webbrowser.open(Url4Search)  # Open the search URL in a web browser.
    return True  # Indicate success.

# Function to play a video on YouTube.
def PlayYoutube(query):
    playonyt(query)  # Use pywhatkit's playonyt function to play the video.
    return True  # Indicate success.

# Function to open an application or a relevant webpage.
def OpenApp(app, sess=requests.session()):
    try:
        appopen(app, match_closest=True, output=True, throw_error=True)  # Attempt to open the app.
        return True  # Indicate success.
    except:
        # Nested function to extract links from HTML content.
        def extract_links(html):
            if html is None:
                return []
            soup = BeautifulSoup(html, 'html.parser')  # Parse the HTML content.
            links = soup.find_all('a', {'jsname': 'UWckNb'})  # Find relevant links.
            return [link.get('href') for link in links]  # Return the links.

        # Nested function to perform a Google search and retrieve HTML.
        def search_google(query):
            url = f"https://www.google.com/search?q={query}"  # Construct the Google search URL.
            headers = {"User-Agent": useragent}  # Use the predefined user-agent.
            response = sess.get(url, headers=headers)  # Perform the GET request.

            if response.status_code == 200:
                return response.text  # Return the HTML content.
            else:
                print("Failed to retrieve search results.")  # Print an error message.
                return None

        html = search_google(app)  # Perform the Google search.
        if html:
            link = extract_links(html)[0]  # Extract the first link from the search results.
            webopen(link)  # Open the link in a web browser.
        return True  # Indicate success.

# Function to close an application.
def CloseApp(app):
    if "chrome" in app:
        pass  # Skip if the app is Chrome.
    else:
        close(app, match_closest=True, output=True, throw_error=True)  # Attempt to close the app.
    return True  # Indicate success.

# Function to execute system-level commands.
def System(command):
    # Nested function to mute the system volume.
    def mute():
        keyboard.press_and_release("volume mute")

    # Nested function to unmute the system volume.
    def unmute():
        keyboard.press_and_release("volume mute")

    # Nested function to increase the system volume.
    def volume_up():
        keyboard.press_and_release("volume up")

    # Nested function to decrease the system volume.
    def volume_down():
        keyboard.press_and_release("volume down")
        
    def pause():
        keyboard.press_and_release("playplay/pause media")

    # Execute the appropriate command.
    if command == "mute":
        mute()
    elif command == "unmute":
        unmute()
    elif command == "volume up":
        volume_up()
    elif command == "volume down":
        volume_down()
    elif command == "pause" or command == "play":
        pause()

    return True  # Indicate success.


# Asynchronous function to translate and execute user commands.
async def TranslateAndExecute(commands: list[str]):
    funcs = []  # List to store asynchronous tasks.

    for command in commands:
        if command.startswith("open "):  # Handle "open" commands.
            if "open it" in command:  # Ignore "open it" commands.
                pass
            elif "open file" == command:  # Ignore "open file" commands.
                pass
            else:
                fun = asyncio.to_thread(OpenApp, command.removeprefix("open "))
                funcs.append(fun)

        elif command.startswith("general "):  # Placeholder for general commands.
            pass

        elif command.startswith("realtime "):  # Placeholder for real-time commands.
            pass

        elif command.startswith("close "):  # Handle "close" commands.
            fun = asyncio.to_thread(CloseApp, command.removeprefix("close "))
            funcs.append(fun)

        elif command.startswith("play "):  # Handle "play" commands.
            fun = asyncio.to_thread(PlayYoutube, command.removeprefix("play "))
            funcs.append(fun)

        elif command.startswith("content "):  # Handle "content" commands.
            fun = asyncio.to_thread(Content, command.removeprefix("content "))
            funcs.append(fun)

        elif command.startswith("google search "):  # Handle Google search commands.
            fun = asyncio.to_thread(GoogleSearch, command.removeprefix("google search "))
            funcs.append(fun)

        elif command.startswith("youtube search "):  # Handle YouTube search commands.
            fun = asyncio.to_thread(YouTubeSearch, command.removeprefix("youtube search "))
            funcs.append(fun)

        elif command.startswith("system "):  # Handle system commands.
            fun = asyncio.to_thread(System, command.removeprefix("system "))
            funcs.append(fun)

        else:
            print(f"No Function Found for {command}")  # Print an error for unrecognized commands.

    results = await asyncio.gather(*funcs)  # Execute all tasks concurrently.

    for result in results:
        if isinstance(result, str):
            yield result
        else:
            yield result


# Asynchronous function for automation.
async def Automation(commands: list[str]):
    # Check if the command list is empty.
        async for result in TranslateAndExecute(commands):
            pass

        return True  # Indicate success
    
if __name__ == "__main__":
    # List of commands
    asyncio.run(Automation([input(">>>")]))