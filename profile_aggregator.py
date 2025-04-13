import json
from pathlib import Path
import time
from typing import List
from dotenv import load_dotenv
import os

from langchain.prompts import PromptTemplate

from typing import List

import openai


def check_api_key_in_env():
    # Loads the environment variables from the .env file into os.environ
    load_dotenv()

    # Retrieve the OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("OpenAI API key not found. Please set it "
                           "in your environment or in the .env file.")


def load_summary(summary_path: str | Path) -> List[dict]:
    # Load the data from the JSON file
    with open(summary_path, "r") as read_file:
        summary = json.load(read_file)

    return summary


def rm_refs_inferred_flag(summary: List[dict]) -> List[dict]:
    for info in summary:
        del info['references']
        del info['is_inferred']


def get_prompt_template() -> PromptTemplate:
    prompt_template = PromptTemplate(
        input_variables=["summary"],
        template="""
You are an expert investigator in profiling individuals based on available personal data.
Your task is to analyze the provided summary of deductions about a user and compile a comprehensive profile.
The summary is given as a JSON array where each object has keys:
- "information": The deduced personal attribute and its value (e.g., "gender: male").
- "reasoning": A brief explanation of how the deduction was made.
Based on the information in the summary, produce a final profile of the user in JSON format. The profile should include at least the following keys:
- "personal_attributes": which may include details like age, gender, location.
- "professional_background": deduced occupation, education, or related professional information.
- "lifestyle_and_interests": hobbies, interests, or personality traits inferred from the data.
- "additional_inferences": any other relevant insights or deductions you can make, such as risk factors or behavioral tendencies.
- "summary": a small summary about the overall user profile.

You can add more keys as you find relevant to make a more complete profile.
If any key information is missing or ambiguous, you may indicate it as "Unknown" or note the uncertainty.

Here is an example of the expected output format:

{{
  "personal_attributes": {{
    "age": "30",
    "gender": "male",
    "location": "San Francisco, CA"
  }},
  "professional_background": {{
    "occupation": "Software Engineer",
    "education": "Bachelor's in Computer Science"
  }},
  "lifestyle_and_interests": {{
    "hobbies": "Hiking, reading, exploring new technologies",
    "personality_traits": "Analytical, detail-oriented, curious"
  }},
  "additional_inferences": "Likely interested in startup culture and new technological innovations.",
  "summary": "30-year-old male software engineer in San Francisco with a Computer Science degree. Enjoys hiking, reading, and exploring new technology. Analytical, detail-oriented, and curious, likely drawn to startup culture."
}}

Below is the summary of deductions extracted from the user's conversations:
{summary}

Based on the above summary, compile the final profile in the specified JSON format.
Return only the JSON output.
"""
    )
    return prompt_template


def prompt_model(prompt: str,
                 model: str,
                 errors_file: str | Path = "error_log.txt") -> str:
    """
    Sends the prompt to the OpenAI API and returns the text output.
    Logs errors to `errors_file` and returns None on failure.
    """
    try:
        response = openai.responses.create(
            model=model,
            input=[
                {"role": "user", "content": prompt}
            ]
        )

        return response.output_text
    except Exception as e:
        with open(errors_file, "a") as log_file:
            error = f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Error: {e}\n"
            log_file.write(error)
            print(f"Error prompting the model: {error}")
        return None


def log_profile2file(output_file: Path, profile: dict):
    with open(output_file, 'w') as f:
        json.dump(profile, f)


if __name__ == '__main__':
    # change as needed
    model = "o3-mini"
    # use the last summary
    summary_path = Path("outputs/gpt-4o/summary_concatenated.json").resolve()
    output_path = Path("outputs/gpt-4o/").resolve()
    output_path.mkdir(exist_ok=True)
    output_file = output_path.joinpath(f"profile_{model}.json").resolve()
    error_logfile = Path("error_log.txt").resolve()

    check_api_key_in_env()

    print(f"Loading summary {summary_path}")
    summary = load_summary(summary_path)

    num_summary_points = len(summary)
    print(
        f"Successfully loaded summary containing {num_summary_points} information pieces.")

    # We don't need the refs anymore.
    rm_refs_inferred_flag(summary)

    prompt_template = get_prompt_template()
    prompt = prompt_template.format(summary=json.dumps(summary, indent=2))

    model_output = prompt_model(prompt, model=model, errors_file=error_logfile)

    if model_output is not None:
        log_profile2file(output_file, json.loads(model_output))
        print(f"Model output written to {output_file}")
