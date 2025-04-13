import json
from pathlib import Path
import time
from dotenv import load_dotenv
import os

from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema


import openai


def check_api_key_in_env():
    # Loads the environment variables from the .env file into os.environ
    load_dotenv()

    # Retrieve the OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("OpenAI API key not found. Please set it "
                           "in your environment or in the .env file.")


def load_profile(profile_path: str | Path) -> dict:
    with open(profile_path, "r") as read_file:
        profile = json.load(read_file)

    return profile


def get_userlocation_template(output_parser: StructuredOutputParser) -> PromptTemplate:
    format_instructions = output_parser.get_format_instructions()

    prompt_template = PromptTemplate(
        input_variables=["user_profile"],
        partial_variables={"format_instructions": format_instructions},
        output_parser=output_parser,
        template="""
Below you will be provided with a user profile as json. Your task is to extract the best known user location from the profile, which can be directly specified as location, or in case it is missing, the nationality can be used as last resort. You should return the location as a json with the following keys:
- "country": the ISO 3166-1 code for the country
- "city": the name of the city
- "region": The name of the region, if known, otherwise the name of the city (e.g., 'Minnesota' for Minneapolis, 'London' for London).

In case the profile contains no location information, return an empty json.

Example: Given the profile
{{
    "personal_attributes": {{
        "name": "John Doe",
        "age": "44",
        "gender": "male",
        "nationality": "Irish",
        "location": "London, England"
    }}
}}

You should return:
{{
    "country": "GB",
    "city": "London",
    "region": "London",
}}

{format_instructions}

User profile:
{user_profile}

Return only the JSON output.
"""
    )
    return prompt_template


def get_user_location(profile: dict, model: str, errors_file: str | Path) -> dict:
    # Define your expected output schema.
    response_schemas = [
        ResponseSchema(
            name="country",
            description="The ISO 3166-1 country code.",
            type="string"
        ),
        ResponseSchema(
            name="city",
            description="The name of the city.",
            type="string"
        ),
        ResponseSchema(
            name="region",
            description=(
                "The name of the region, if known, otherwise the name of the city (e.g., 'Minnesota' for Minneapolis, 'London' for London)."),
            type="string"
        )
    ]

    output_parser = StructuredOutputParser.from_response_schemas(
        response_schemas)

    template = get_userlocation_template(output_parser)
    prompt = template.format(user_profile=json.dumps(profile, indent=2))
    output = prompt_model(prompt, model=model, errors_file=errors_file)
    user_location = output_parser.parse(output)
    user_location["type"] = "approximate"
    return user_location


def get_userfinder_template() -> PromptTemplate:
    prompt_template = PromptTemplate(
        input_variables=["user_profile"],
        template="""
You are a private investigator specialized in finding the identity of people on the internet.
Below you are provided with a profile from an unknown person in json format. Your task is to try to find who that person is by doing a web search.
You can search using any terms you want and you should return the websites where you believe this person is mentioned. You are free to do multiple searches and combine the final result.
If you don't find this person after at least 5 searches, you should tell the user that.
At the end of the message, provide all the terms you used in the search and the websites you retrieved.

User profile:
{user_profile}
"""
    )
    return prompt_template


def prompt_model(prompt: str,
                 model: str,
                 location: dict = None,
                 temperature: float = 0,
                 errors_file: str | Path = "error_log.txt") -> str:
    """
    Sends the prompt to the OpenAI API and returns the text output.
    Logs errors to `errors_file` and returns None on failure.
    """
    try:
        response = openai.responses.create(
            model=model,
            tools=[{"type": "web_search_preview",
                    "user_location": location,
                    # "user_location": {
                    #     "type": "approximate",
                    #     "country": "DE",
                    #     "city": "Munich",
                    #     "region": "Munich",
                    # }
                    }],
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


def log_output2file(output_file: Path, model_output: str):
    with open(output_file, 'w') as f:
        f.write(model_output)


if __name__ == '__main__':
    # change as needed
    model = "gpt-4o"

    profile_path = Path("outputs/gpt-4o/profile_o3-mini_noname.json").resolve()
    output_path = Path("outputs/gpt-4o/").resolve()
    output_path.mkdir(exist_ok=True)
    output_file = output_path.joinpath("webperson_results.txt").resolve()
    error_logfile = Path("error_log.txt").resolve()

    check_api_key_in_env()

    print(f"Loading profile {profile_path}")
    profile = load_profile(profile_path)

    print("Prompting the model to extract the user location to localize the identity search")
    user_location = get_user_location(
        profile, model=model, errors_file=error_logfile)
    print(
        f"The best known location for the user is (openAI API format) {user_location}")

    prompt_template = get_userfinder_template()
    prompt = prompt_template.format(user_profile=json.dumps(profile, indent=2))

    model_output = prompt_model(
        prompt, model=model, location=user_location, errors_file=error_logfile)

    if model_output is not None:
        log_output2file(output_file, model_output)
        print(f"Model output written to {output_file}")
