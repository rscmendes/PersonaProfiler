"""Steps to perform "profile" extraction:
1. Load the conversations
2. Create a prompt that integrates 1 conversation and the summary of deductions, to generate an updated summary of deductions
3. Make the API request and handle the response (updated summary of deductions)
4. Repeat 2 and 3 for each conversation
"""
import json
from pathlib import Path
import time
from typing import List, Tuple, Optional
from datetime import datetime
from dotenv import load_dotenv
import os

from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser

from pydantic import BaseModel, Field, RootModel
import openai


def check_api_key_in_env():
    # Loads the environment variables from the .env file into os.environ
    load_dotenv()

    # Retrieve the OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("OpenAI API key not found. Please set it "
                           "in your environment or in the .env file.")


def load_data(data_path: str | Path) -> List[dict]:
    """Loads the full conversations from the export.

    The exported `conversations.json` file has a hierarchical structure. We need to reconstruct the full conversation.
    I found a [script by Jacob Peacock (txtatech)](https://github.com/txtatech/chatgpt-export-to-text/tree/main) to convert to text. 
    I slightly adapted it to retrieve the conversation ID and last updated time.
    """
    # Load the conversations from the JSON file
    with open(data_path, "r") as read_file:
        data = json.load(read_file)

    # Empty list to hold all conversations
    all_conversations = []

    # Loop through each conversation in the data
    for conversation in data:
        conversation_text = []

        for current_node, node_details in conversation['mapping'].items():
            message = node_details.get('message')

            if message:
                author_role = message['author']['role']
                if 'parts' in message['content']:
                    content_parts = message['content']['parts']

                    # Create conversation text
                    text_string = ""
                    for part in content_parts:
                        # Handle different data types in 'parts'
                        if isinstance(part, str):
                            text_string += part
                        elif isinstance(part, dict):
                            # Adjust based on the actual key(s) in the dictionary
                            # Replace 'text' if necessary
                            text_string += part.get('text', '')
                        else:
                            # Fallback to string conversion
                            text_string += str(part)

                    text_prefix = "User: " if author_role == 'user' else "ChatGPT: "
                    text_string = text_prefix + text_string

                    conversation_text.append(text_string)

        # Add this conversation to the all_conversations
        if conversation_text:
            # last_updated = conversation['update_time']
            # conversation_id = conversation['conversation_id']
            all_conversations.append({
                "conversation_id": conversation['conversation_id'],
                "last_updated": conversation['update_time'],
                "content": "\n".join(conversation_text),
            })

    return all_conversations


def sort_conversations(all_conversations: List[dict]):
    """Sorts conversations ascending by last_updated"""
    all_conversations.sort(key=lambda x: x['last_updated'])
    start_date = datetime.fromtimestamp(all_conversations[0]['last_updated'])
    end_date = datetime.fromtimestamp(all_conversations[-1]['last_updated'])
    print(
        f"Data spawns from {start_date} to {end_date} for a "
        f"total of {(end_date-start_date).days} days.")


def get_prompt_template_and_output_parser() -> Tuple[PromptTemplate, PydanticOutputParser]:
    """
    We will base the prompt on the original research paper: 
    Staab, Robin, et al. "Beyond memorization: Violating privacy via inference with large language models." arXiv preprint arXiv:2310.07298 (2023).

    However, we need to adjust to our use-case: 
    1. multiple conversations belonging to a single individual -- iterative approach where we feed conversation-by-conversation
    2. A summary of deductions is outputed by the model at each turn. This summary is fed to the next term
    3. we will not focus on any specific attribute, and let the model extract anything it wants
    """
    class PersonalDeduction(BaseModel):
        """Personal information deducted from the conversation"""
        information: str = Field(
            description="The deduced personal information and value (e.g. 'gender: male')")
        reasoning: str = Field(
            description="The reasoning that led to inferring the information from the conversation.")
        is_inferred: bool = Field(
            description="Whether the information was inferred/deduced from available information")
        references: Optional[List[str]] = Field(
            default=None, description="The conversation IDs that support the deduction.")

    class DeductionList(RootModel[List[PersonalDeduction]]):
        pass

    # Create a structured output parser from the defined schemas
    output_parser = PydanticOutputParser(pydantic_object=DeductionList)
    format_instructions = output_parser.get_format_instructions()
    escaped_format_instructions = format_instructions.replace(
        "{", "{{").replace("}", "}}")

    # Create a prompt template that integrates one conversation and the current summary
    # Prompt inspired by the original research paper:
    # Staab, Robin, et al. "Beyond memorization: Violating privacy via inference with large language models." arXiv preprint arXiv:2310.07298 (2023).
    prompt_template = PromptTemplate(
        input_variables=["conversation", "current_summary"],
        output_parser=output_parser,
        template=f"""
You are an expert investigator with years of experience in online profiling and text analysis.
You are tasked with extract as much personal information as possible an anonymous user of a chat service, i.e., user profiling and deanonymization.
You only care about extracting personal information such as job, age, hobbies, family, relationships, tastes, personality, etc. World facts, and non-personal information is irrelevant, except when it hints about potential occupation, for example.

You are provided with the text of one conversation (between the user and ChatGPT) and a summary of personal deductions from past conversations.
The summary is a JSON array where each element is an object with keys "information", "reasoning", "inferred", and "references".
Your task is to update the summary by:
    1. Correcting any errors in previous deductions.
    2. Complementing existing deductions with any new details found in the conversation.
    3. Adding new deductions if the conversation contains additional personal information.
Deductions can be hints or unconfirmed personal information, that you need more conversations to confirm.

Ensure that each updated deduction object includes:
    - "information": The deduced personal attribute and its value (e.g., "gender: male").
    - "reasoning": A brief explanation of how the deduction was made.
    - "is_inferred": the value for this key is either false, if the information is explicitly mentioned by the user in the conversation, or true if you inferred/deduced from available information.
    - "references": A list of conversation IDs or parts that support this deduction. If you use the current conversation to deduce or correct information for this particular deduction, then append the current conversation ID to this list. Otherwise don't change it.

{escaped_format_instructions}

Conversation (id = {{conversation_id}}):
{{conversation}}

Current Summary:
{{current_summary}}

Return only the JSON output.
"""
    )
    return prompt_template, output_parser


def prompt_model(prompt: str,
                 model: str,
                 temperature: float = 0,
                 errors_file: str | Path = "error_log.txt") -> dict:
    """
    Sends the prompt to the OpenAI API and returns the updated summary as a Python dict.
    Logs errors to `errors_file` and returns None on failure.
    """
    try:
        response = openai.responses.create(
            model=model,
            input=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
        )

        return response.output_text
    except Exception as e:
        with open(errors_file, "a") as log_file:
            error = f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Error: {e}\n"
            log_file.write(error)
            print(f"Error prompting the model: {error}")
        return None


def log_summary2file(output_file: Path, current_summary: List[dict]):
    with open(output_file, 'w') as f:
        json.dump(current_summary, f)


if __name__ == '__main__':
    # change as needed
    model = "gpt-4o"  # gpt-4o-mini
    data_path = Path("chatGPT-data/conversations.json").resolve()
    output_path = Path("outputs").joinpath(model).resolve()
    output_path.mkdir(exist_ok=True, parents=True)
    error_logfile = Path("error_log.txt").resolve()

    check_api_key_in_env()

    print(f"Loading data from {data_path}")
    all_conversations = load_data(data_path)

    num_conversations = len(all_conversations)
    print(f"Successfully loaded {num_conversations} conversations")

    sort_conversations(all_conversations)

    prompt_template, output_parser = get_prompt_template_and_output_parser()

    current_summary = []
    for i, conv in enumerate(all_conversations):
        output_file = output_path.joinpath(f"summary_{i}.json")
        if output_file.exists():
            print(
                f"Found existing output file {output_file}, loading from file.")
            with open(output_file, 'r') as f:
                current_summary = json.load(f)
            continue

        print(f"Prompting for conversation {i}/{num_conversations}")
        prompt = prompt_template.format(
            conversation_id=conv['conversation_id'],
            conversation=conv['content'],
            current_summary=json.dumps(current_summary, indent=2)
        )

        # TODO: I got tokens per min (TPM) limit error so I ran passing an empty summary from there on
        # summary_66, summary_83, 108, 152

        model_output = prompt_model(
            prompt, model=model, errors_file=error_logfile)
        parsed_output = output_parser.parse(model_output)
        current_summary = parsed_output.model_dump()

        # print(current_summary)

        log_summary2file(output_file, current_summary)
