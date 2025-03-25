'''This file contains functions to initialize the OpenAI client, call assistant and delete threads.
'''
import os
from typing import Tuple

from openai import OpenAI


def init_openai_client() -> OpenAI:
    '''Initialize the OpenAI API client.

    Args:
        None

    Returns:
        OpenAI: The OpenAI API client.

    Raises:
        ValueError: If the OpenAI API key is missing.
    '''
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        raise ValueError('OpenAI API key is missing.')
    client = OpenAI(api_key=openai_api_key)
    return client


def delete_assistant_thread(openai_client: OpenAI, thread_id: str) -> None:
    '''Delete an assistant thread.

    Args:
        openai_client: The OpenAI API client.
        thread_id: The thread ID.
    '''
    openai_client.beta.threads.delete(thread_id)


def call_assistant(openai_client: OpenAI, user_message: str, assistant_id: str, existing_thread_id: str = None) -> Tuple[str, str]:
    '''Call an assistant.

    Args:
        openai_client: The OpenAI API client.
        user_message: The user message.
        assistant_id: The assistant ID.
        existing_thread_id (optional): The existing thread ID. Defaults to None.

    Returns:
        Tuple[str, str]: The assistant response and the thread ID.
    '''
    if not existing_thread_id:
        thread = openai_client.beta.threads.create()
        existing_thread_id = thread.id
    else:
        thread = openai_client.beta.threads.retrieve(existing_thread_id)

    openai_client.beta.threads.messages.create(
        thread_id=existing_thread_id,
        role='user',
        content=user_message
    )
    run = openai_client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant_id,
    )
    if run.status == 'completed':
        messages = openai_client.beta.threads.messages.list(
            thread_id=existing_thread_id
        )
        list_of_messages = list(messages)
        try:
            return list_of_messages[0].content[0].text.value, existing_thread_id
        except Exception as e:
            print('Exception:', e)
            return None, existing_thread_id
    else:
        print(run.status)
        return None, existing_thread_id
