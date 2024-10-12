import asyncio

from .generator import BudgeRigar, PROMPT_FOR_MULTITURN_QA_GENERATION
from utils.image_process import encode_image


class AsyncBudgeRigar(BudgeRigar):

    async def generate(
        self,
        image_path:str,
        min_turn:int=1,
        max_turn:int=3,
        question_independent:bool=False,
    ):
        """Asynchronously generates messages based on the provided image path and turn parameters.

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

        response = await self.engine.chat_completions(messages)
        content = response.choices[0].message.content
        data = self.parse_and_get_generated_messages(content)
        return data


    async def batch_generate(
        self,
        image_paths:str | list[str],
        **kwargs,
    ):
        """Asynchronously generates responses for multiple images based on the specified generation type.

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

        tasks = [self.generate(image_path, **kwargs) for image_path in image_paths]
        responses = await asyncio.gather(*tasks)
        return responses
