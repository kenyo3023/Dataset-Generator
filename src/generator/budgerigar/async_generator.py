import asyncio
from typing import Union, List
from .generator import BudgeRigar


class AsyncBudgeRigar(BudgeRigar):

    async def generate(
        self,
        image_path:str,
        **kwargs
    ):
        """Asynchronously generates messages based on the provided image path and turn parameters.

        Args:
            image_path (str): The file path to the image.
            **kwargs: Additional keyword arguments for message preparation
                such as min_turn, max_turn, and question_independent.

        Returns:
            list: The generated messages as a list.

        Raises:
            AssertionError: If min_turn is greater than max_turn.
        """
        messages = self._prepare_messages(image_path, **kwargs)

        response = await self.engine.chat_completions(messages)
        content = response.choices[0].message.content
        data = self.parse_and_get_generated_messages(content)
        return data


    async def batch_generate(
        self,
        image_paths:Union[str, List[str]],
        **kwargs,
    ):
        """Asynchronously generates responses for multiple images based on the specified generation type.

        This method handles a single image path or a list of image paths and generates responses for each.

        Args:
            image_paths (Union[str, List[str]]): A single image path or a list of image paths.
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
