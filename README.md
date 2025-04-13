PersonaProfiler is a small project that leverages large language models (LLMs) to analyze exported ChatGPT (or from any other provider) conversations and deduce personal information. The goal is to demonstrate how digital dialogues can reveal sensitive data — not only what is directly shared but also what can be inferred through context.

I wrote a detailed a medium story about this project and its findings using my own data. You can [check it here](https://medium.com/@loving_ochre_cattle_466/profiling-users-from-conversations-what-openai-or-other-chatbot-providers-know-about-you-5014750fa8bb).

# Setup
1. (Optional) create a virtual environment
2. pip install -r requirements.txt
3. Export you conversations from ChatGPT into `chatGPT-data`. 
    * You are free to rename this folder, but you will also need to update the scripts accordingly. 
    * The scripts only support reading ChatGPT exports, so if you are using other provider you will need to adapt the loading code. Particularly, the export from ChatGPT has a `conversations.json` file that is used to load the conversations.
4. Rename the `.env.example` to `.env` and put your OpenAI key.


# Running the code
The pipeline is composed of 3 steps, each having a corresponding script:
1. **Information Extraction (`profile_extractor.py`):** For each conversation, a model (e.g., GPT-4 variants) is prompted to extract personal details and compile a running summary of deductions. Both the conversation and the latest summary is fed to the model, such that the summary is continuously increasing and being corrected. After analyzing a conversation, the summary is written into `òutputs/{model_name}/summary_{i}`, with i the index of the conversation. The final summary contains all the information from all conversations -- we store intermediate summaries for cases where there are some connectivity issues.
2. **Profiling (`profile_aggregator.py`)**: Another model uses the comprehensive summary to generate a final profile in JSON format, covering aspects like personal attributes, professional background, lifestyle, and additional inferences.
3. **Person Finder (`webperson_finder.py`)**: A model with web search capability attempts to locate the user online based on the profile details.

Feel free to change the openAI models that are used in each step by specifying them in the main of each script. By default, the information extraction uses gpt-4o, the profiling uses gpt-o3-mini, and the person finder uses gpt-4o with web search capabilities.

A jupyter notebook `data_analysis.ipynb` is provided with some code to inspect the conversations and all outputs of the models.

# Disclaimer

This project is intended for educational purposes only. It demonstrates the potential privacy implications of using chatbots and LLMs. Please handle all personal data responsibly and avoid publicly sharing sensitive information.