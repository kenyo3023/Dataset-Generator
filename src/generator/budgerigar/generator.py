import re
import json
from enum import Enum
from typing import Union

from jinja2 import Template

from utils.image_process import encode_image


PROMPT_FOR_MULTITURN_QA_GENERATION = Template('''
You are given an image and must generate a multi-turn QA session.
The session should have a minimum of {{ min_turn }} turns and a maximum of {{ max_turn }} turns.
{% if question_independent -%}
Each question could be independent of the previous turn.
{%- else -%}
Each question should derive from the previous one, creating a coherent conversation.
{%- endif %}

1. Start by identifying an important feature in the image.
2. Generate the first question based on that feature.
{% if question_independent -%}
3. For each subsequent turn, generate a question that could be independent of the previous turn.
{% else -%}
3. For each subsequent turn, generate a question that derives from the last question.
{% endif -%}
4. Provide an answer for each question.

Format the output as follows:
{
  "messages": [
    {
      "role": "user",
      "content": [Generated question based on the selected key point]
    },
    {
      "role": "assistant",
      "content": [Generated answer based on the image and key point]
    },
    ...
  ]
}
'''.strip())


class BudgeRigar:
    """A class for generating responses based on image input using a specified engine.

    Attributes:
        engine: An instance of the engine used for generating responses.
    """

    def __init__(self, engine=None):
        """Initializes the BudgeRigar instance with the specified engine.

        Args:
            engine: The engine to be used for generating responses. Default is None.
        """
        self.engine = engine


    def parse_and_get_generated_messages(self, content:str):
        """Parses the input content to extract generated messages.

        Args:
            content (str): The content containing the JSON messages.

        Returns:
            list: A list of messages extracted from the content if successful; otherwise, None.
        """
        pattern = r'.*?"messages":\s*(\[[^]]*\]).*?' # `.*?` 用於匹配任意文字
        match = re.search(pattern, content, re.DOTALL)

        if match:
            try:
                json_str = match.group(1)  # 提取到的 JSON 字符串
                messages = json.loads(json_str)  # 解析 JSON
                return messages
            except Exception as e:
                return None # TODO
        else:
            print("No match found.")
            return None # TODO


    def generate(
        self,
        image_path:str,
        min_turn:int=1,
        max_turn:int=3,
        question_independent:bool=False,
    ):
        """Generates messages based on the provided image path and turn parameters.

        Args:
            image_path (str): The file path to the image.
            min_turn (int): The minimum number of turns for generation. Default is 1.
            max_turn (int): The maximum number of turns for generation. Default is 3.
            question_independent (bool): If True, each question may be independent of the previous questions. Default is False.

        Returns:
            list: The generated messages as a list.
        
        Raises:
            AssertionError: If min_turn is greater than max_turn.
        """
        assert min_turn <= max_turn, \
            f'arg \'min_turn={min_turn}\' must smaller than \'max_turn={max_turn}\''

        image_url = encode_image(image_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        # "text": PROMPT_FOR_MULTITURN_QA_GENERATION.format(
                        "text": PROMPT_FOR_MULTITURN_QA_GENERATION.render(
                            min_turn=min_turn,
                            max_turn=max_turn,
                            question_independent=question_independent,
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                        "url": image_url,
                        },
                    },
                ],
            }
        ]

        response = self.engine.chat_completions(messages)
        content = response.choices[0].message.content
        data = self.parse_and_get_generated_messages(content)
        return data


    def batch_generate(
        self,
        image_paths:str | list[str],
        **kwargs,
    ):
        """Generates responses for multiple images based on the specified generation type.

        This method handles a single image path or a list of image paths and generates responses for each.

        Args:
            image_paths (str | list[str]): A single image path or a list of image paths.
            **kwargs: Additional keyword arguments for the generate method.

        Returns:
            list: A list of generated responses for each image.

        Raises:
            ValueError: If generate_type is not a valid element of GenerateType.
        """
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        return [self.generate(image_path, **kwargs) for image_path in image_paths]
