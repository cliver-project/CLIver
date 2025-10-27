import asyncio

from cliver import TaskExecutor
from cliver.workflow.workflow_executor import WorkflowExecutor
from cliver.workflow.workflow_manager_local import LocalDirectoryWorkflowManager

task_executor = TaskExecutor(llm_models={}, mcp_servers={}, default_model="qwen")

# Create workflow manager and executor (requires a TaskExecutor instance)
workflow_manager = LocalDirectoryWorkflowManager()
workflow_executor = WorkflowExecutor(task_executor, workflow_manager)


async def run_workflow():
    result = await workflow_executor.execute_workflow(
        workflow_name="workflow_name",
        inputs={"document_path": "/path/to/doc.txt", "author": "CLIver Team"},
    )
    print(result)


asyncio.run(run_workflow())
