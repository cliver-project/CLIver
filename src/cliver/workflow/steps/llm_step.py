"""
LLM step implementation.
"""

import logging
import time
from pathlib import Path
from typing import Optional

from langchain_core.messages import AIMessage

from cliver import MultimediaResponseHandler
from cliver.llm import TaskExecutor
from cliver.workflow.steps.base import StepExecutor
from cliver.workflow.workflow_models import ExecutionContext, ExecutionResult, LLMStep

logger = logging.getLogger(__name__)


class LLMStepExecutor(StepExecutor):
    """Executor for LLM steps."""

    def __init__(self, step: LLMStep, task_executor: TaskExecutor, cache_dir: Optional[str] = None):
        super().__init__(step)
        self.step = step
        self.task_executor = task_executor
        self.cache_dir = cache_dir

    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        start_time = time.time()

        try:
            # Resolve variables in all LLM parameters
            llm_params = {
                "user_input": self.resolve_variable(self.step.prompt, context),
                "model": self.resolve_variable(self.step.model, context) if self.step.model else None,
                "images": [self.resolve_variable(img, context) for img in self.step.images] if self.step.images else [],
                "audio_files": [self.resolve_variable(a, context) for a in self.step.audio_files]
                if self.step.audio_files
                else [],
                "video_files": [self.resolve_variable(v, context) for v in self.step.video_files]
                if self.step.video_files
                else [],
                "files": [self.resolve_variable(f, context) for f in self.step.files] if self.step.files else [],
                "template": self.resolve_variable(self.step.template, context) if self.step.template else None,
                "params": {k: self.resolve_variable(v, context) for k, v in self.step.params.items()}
                if self.step.params
                else {},
            }

            # Execute LLM call
            if self.step.stream:
                accumulated_content = ""
                async for chunk in self.task_executor.stream_user_input(**llm_params):
                    if hasattr(chunk, "content") and chunk.content:
                        accumulated_content += str(chunk.content)
                response = AIMessage(content=accumulated_content)
            else:
                response = await self.task_executor.process_user_input(**llm_params)

            result_content = response.content if hasattr(response, "content") else str(response)

            # Handle multimedia caching
            media_references = {}
            if self.cache_dir:
                llm_engine = self.task_executor.get_llm_engine(self.step.model)
                response_handler = MultimediaResponseHandler()
                multimedia_response = response_handler.process_response(
                    response, llm_engine=llm_engine, auto_save_media=False
                )
                if multimedia_response.has_media():
                    step_media_dir = Path(self.cache_dir) / self.step.id
                    step_media_dir.mkdir(parents=True, exist_ok=True)
                    for media in multimedia_response.media_content:
                        media_type = media.type.value
                        media_type_dir = step_media_dir / media_type
                        media_type_dir.mkdir(parents=True, exist_ok=True)
                        try:
                            file_path = media_type_dir / (
                                media.filename or f"{media_type}_{len(media_references.get(media_type, []))}"
                            )
                            media.save(file_path)
                            media_references.setdefault(media_type, []).append(str(file_path))
                        except Exception as e:
                            logger.warning(f"Error saving media {media.filename}: {e}")

            outputs = await self.extract_outputs(result_content)
            if media_references:
                outputs["media"] = media_references

            return ExecutionResult(
                step_id=self.step.id,
                outputs=outputs,
                success=True,
                execution_time=time.time() - start_time,
            )

        except Exception as e:
            logger.error(f"Error executing LLM step {self.step.id}: {e}")
            return ExecutionResult(
                step_id=self.step.id,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time,
            )
