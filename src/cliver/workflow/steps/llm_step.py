"""
LLM step implementation.
"""
import logging
import time
from pathlib import Path
from typing import Optional
from langchain_core.messages import AIMessage

from cliver import MultimediaResponseHandler
from cliver.workflow.steps.base import StepExecutor
from cliver.workflow.workflow_models import LLMStep, ExecutionContext, ExecutionResult
from cliver.llm import TaskExecutor

logger = logging.getLogger(__name__)


class LLMStepExecutor(StepExecutor):
    """Executor for LLM steps."""

    def __init__(self, step: LLMStep, task_executor: TaskExecutor, cache_dir: Optional[str] = None):
        super().__init__(step)
        self.step = step
        self.task_executor = task_executor
        self.cache_dir = cache_dir

    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute an LLM call.

        Args:
            context: Execution context containing inputs

        Returns:
            ExecutionResult with outputs
        """
        start_time = time.time()

        try:
            # Resolve variables in the prompt
            resolved_prompt = self.resolve_variable(self.step.prompt, context)

            # Prepare LLM call parameters, resolving variables in all fields
            llm_params = {
                "user_input": resolved_prompt,
                "model": self.resolve_variable(self.step.model, context) if self.step.model else None,
                "images": [self.resolve_variable(img, context) for img in self.step.images] if self.step.images else [],
                "audio_files": [self.resolve_variable(audio, context) for audio in self.step.audio_files] if self.step.audio_files else [],
                "video_files": [self.resolve_variable(video, context) for video in self.step.video_files] if self.step.video_files else [],
                "files": [self.resolve_variable(file, context) for file in self.step.files] if self.step.files else [],
                "skill_sets": [self.resolve_variable(skill, context) for skill in self.step.skill_sets] if self.step.skill_sets else [],
                "template": self.resolve_variable(self.step.template, context) if self.step.template else None,
                "params": {k: self.resolve_variable(v, context) for k, v in self.step.params.items()} if self.step.params else {}
            }

            # Execute the LLM call
            if self.step.stream:
                # For streaming, we need to accumulate the response
                accumulated_content = ""
                async for chunk in self.task_executor.stream_user_input(**llm_params):
                    if hasattr(chunk, "content") and chunk.content:
                        accumulated_content += str(chunk.content)
                response = AIMessage(content=accumulated_content)
            else:
                # For non-streaming, get the complete response
                response = await self.task_executor.process_user_input(**llm_params)

            # Now we get the final response whether in streaming mode or not
            # Get the LLM engine used for this response
            llm_engine = self.task_executor.get_llm_engine(self.step.model)

            # Process response with multimedia handler
            # For now, we'll create a handler without auto-saving, and handle media saving separately
            response_handler = MultimediaResponseHandler()
            multimedia_response = response_handler.process_response(
                response, llm_engine=llm_engine, auto_save_media=False
            )

            result_content = response.content if hasattr(response, 'content') else str(response)

            # Handle multimedia content caching if cache directory is provided
            media_references = {}
            if self.cache_dir and multimedia_response.has_media():
                # Create step-specific media directory with proper structure
                # {cache_dir}/{workflow_name}/{execution_id}/{step_id}/
                step_media_dir = Path(self.cache_dir) / self.step.id
                step_media_dir.mkdir(parents=True, exist_ok=True)

                # Save media content and create references, organized by media type
                for media in multimedia_response.media_content:
                    media_type = media.type.value
                    # Create subdirectory for media type (images, audios, videos, files)
                    media_type_dir = step_media_dir / media_type
                    media_type_dir.mkdir(parents=True, exist_ok=True)

                    # Save the media file in the appropriate subdirectory
                    try:
                        file_path = media_type_dir / (media.filename or f"{media_type}_{len(media_references.get(media_type, []))}")
                        media.save(file_path)
                        if media_type not in media_references:
                            media_references[media_type] = []
                        media_references[media_type].append(str(file_path))
                    except Exception as e:
                        logger.warning(f"Error saving media {media.filename}: {e}")

            execution_time = time.time() - start_time

            # Prepare outputs
            outputs = await self.extract_outputs(result_content)

            # Add media references to outputs if any
            if media_references:
                outputs["media"] = media_references

            return ExecutionResult(
                step_id=self.step.id,
                outputs=outputs,
                success=True,
                error=None,
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error executing LLM step {self.step.id}: {str(e)}")
            return ExecutionResult(
                step_id=self.step.id,
                success=False,
                error=str(e),
                execution_time=execution_time
            )
