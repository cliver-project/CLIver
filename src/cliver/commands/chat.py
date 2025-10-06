import asyncio
import time
from typing import List, Optional

import click

from cliver.cli import Cliver, pass_cliver
from cliver.llm import TaskExecutor
from cliver.media_handler import MultimediaResponseHandler
from cliver.model_capabilities import ModelCapability


@click.command(name="chat", help="Chat with LLM models")
@click.option(
    "--model",
    "-m",
    type=str,
    required=False,
    help="Which LLM model to use",
)
@click.option(
    "--stream",
    "-s",
    is_flag=True,
    default=False,
    help="Stream the response",
)
@click.option(
    "--image",
    "-img",
    multiple=True,
    help="Image files to send with the message",
)
@click.option(
    "--audio",
    "-aud",
    multiple=True,
    help="Audio files to send with the message",
)
@click.option(
    "--video",
    "-vid",
    multiple=True,
    help="Video files to send with the message",
)
@click.option(
    "--file",
    "-f",
    multiple=True,
    help="General files to upload for tools like code interpreter",
)
@click.option(
    "--save-media",
    "-sm",
    is_flag=True,
    default=False,
    help="Automatically save media content from responses to files",
)
@click.option(
    "--media-dir",
    "-md",
    type=str,
    default=None,
    help="Directory to save media files (default: current directory)",
)
@click.option(
    "--skill-set",
    "-ss",
    multiple=True,
    help="Skill sets to apply to the chat session",
)
@click.option(
    "--template",
    "-t",
    type=str,
    help="Template to use for the prompt",
)
@click.option(
    "--param",
    "-p",
    multiple=True,
    type=(str, str),
    help="Parameters for skill sets and templates (key=value)",
)
@click.argument("query", nargs=-1)
@pass_cliver
def chat(
    cliver: Cliver,
    model: Optional[str],
    stream: bool,
    image: List[str],
    audio: List[str],
    video: List[str],
    file: List[str],
    skill_set: List[str],
    template: Optional[str],
    param: List[tuple],
    save_media: bool,
    media_dir: Optional[str],
    query: str,
):
    """
    Chat with LLM models.
    """
    try:
        task_executor = cliver.task_executor
        llm_engine = task_executor.get_llm_engine(model)
        if not llm_engine:
            # print message that model is not found
            click.echo(f"Model '{model}' not found.")
            return 1

        # Check capabilities before making calls
        if len(file) > 0 and not llm_engine.supports_capability(
            ModelCapability.FILE_UPLOAD
        ):
            click.echo(f"Model '{model}' does not support file uploads.")
            return 1

        if len(image) > 0 and not llm_engine.supports_capability(
            ModelCapability.IMAGE_TO_TEXT
        ):
            click.echo(f"Model '{model}' does not support image processing.")
            return 1

        if len(audio) > 0 and not llm_engine.supports_capability(
            ModelCapability.AUDIO_TO_TEXT
        ):
            click.echo(f"Model '{model}' does not support audio processing.")
            return 1

        if len(video) > 0 and not llm_engine.supports_capability(
            ModelCapability.VIDEO_TO_TEXT
        ):
            click.echo(f"Model '{model}' does not support video processing.")
            return 1

        sentence = " ".join(query)
        # Convert param tuples to dictionary
        params = dict(param)
        return _async_chat(
            task_executor,
            sentence,
            model,
            stream,
            image,
            audio,
            video,
            file,
            skill_set,
            template,
            params,
            save_media,
            media_dir,
        )
    except Exception as e:
        click.echo(f"Error: {e}")
        return 1


def _async_chat(
    task_executor: TaskExecutor,
    user_input: str,
    model: str,
    stream: bool = False,
    images: List[str] = None,
    audio_files: List[str] = None,
    video_files: List[str] = None,
    files: List[str] = None,
    skill_sets: List[str] = None,
    template: Optional[str] = None,
    params: dict = None,
    save_media: bool = False,
    media_dir: Optional[str] = None,
):
    # Create multimedia response handler
    response_handler = MultimediaResponseHandler(media_dir)

    try:
        if stream:
            # For streaming, we need to run the async generator
            return asyncio.run(
                _stream_chat(
                    task_executor,
                    user_input,
                    images,
                    audio_files,
                    video_files,
                    files,
                    model,
                    skill_sets,
                    template,
                    params,
                    save_media,
                    media_dir,
                )
            )
        else:
            response = task_executor.process_user_input_sync(
                user_input=user_input,
                images=images,
                audio_files=audio_files,
                video_files=video_files,
                files=files,
                model=model,
                skill_sets=skill_sets,
                template=template,
                params=params,
            )
            if response:
                # Get the LLM engine used for this response
                llm_engine = task_executor.get_llm_engine(model)

                # Process response with multimedia handler
                multimedia_response = response_handler.process_response(
                    response, llm_engine=llm_engine, auto_save_media=save_media
                )

                # Display text content
                if multimedia_response.has_text():
                    click.echo(multimedia_response.text_content)

                # Display media information
                if multimedia_response.has_media():
                    media_count = len(multimedia_response.media_content)
                    click.echo(f"\n[Media Content: {media_count} items]")
                    for i, media in enumerate(multimedia_response.media_content):
                        info = f"  {i + 1}. {media.type.value}"
                        if media.filename:
                            info += f" ({media.filename})"
                        if media.mime_type:
                            info += f" [{media.mime_type}]"
                        click.echo(info)
    except ValueError as e:
        if "File upload is not supported" in str(e):
            click.echo(f"Error: {e}")
            click.echo(
                "Please configure your model with --supports-file-upload flag "
                "or remove the --file option."
            )
        else:
            raise
        return 1
    except Exception as e:
        click.echo(f"Error: {e}")
        return 1
    return 0


async def _stream_chat(
    task_executor: TaskExecutor,
    user_input: str,
    images: List[str] = None,
    audio_files: List[str] = None,
    video_files: List[str] = None,
    files: List[str] = None,
    model: str = None,
    skill_sets: List[str] = None,
    template: Optional[str] = None,
    params: dict = None,
    save_media: bool = False,
    media_dir: Optional[str] = None,
):
    """Stream the chat response character by character."""
    # Create multimedia response handler
    response_handler = MultimediaResponseHandler(media_dir)

    try:
        accumulated_content = ""
        async for chunk in task_executor.stream_user_input(
            user_input=user_input,
            images=images,
            audio_files=audio_files,
            video_files=video_files,
            files=files,
            model=model,
            skill_sets=skill_sets,
            template=template,
            params=params,
        ):
            if hasattr(chunk, "content") and chunk.content:
                # For streaming, we accumulate content and display it as it comes
                accumulated_content += str(chunk.content)
                # Print each character with a small delay to simulate streaming
                import sys

                for char in str(chunk.content):
                    sys.stdout.write(char)
                    sys.stdout.flush()
                    time.sleep(0.01)

        # After streaming is complete, process the accumulated content
        if accumulated_content:
            # Create a simple AIMessage for processing
            from langchain_core.messages import AIMessage

            simple_response = AIMessage(content=accumulated_content)

            # Get the LLM engine used for this response
            llm_engine = task_executor.get_llm_engine(model)

            multimedia_response = response_handler.process_response(
                simple_response, llm_engine=llm_engine, auto_save_media=save_media
            )

            # Display media information if any
            if multimedia_response.has_media():
                print()  # New line
                media_count = len(multimedia_response.media_content)
                print(f"\n[Media Content: {media_count} items]")
                for i, media in enumerate(multimedia_response.media_content):
                    info = f"  {i + 1}. {media.type.value}"
                    if media.filename:
                        info += f" ({media.filename})"
                    if media.mime_type:
                        info += f" [{media.mime_type}]"
                    print(info)

        print()  # New line at the end
        return 0
    except ValueError as e:
        if "File upload is not supported" in str(e):
            click.echo(f"Error: {e}")
            click.echo(
                "Please configure your model with --supports-file-upload flag "
                "or remove the --file option."
            )
        else:
            raise
        return 1
    except Exception as e:
        click.echo(f"Error: {e}")
        return 1
