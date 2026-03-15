import asyncio

from cliver import TaskExecutor
from cliver.workflow.workflow_executor import WorkflowExecutor
from cliver.workflow.workflow_manager_local import LocalDirectoryWorkflowManager

task_executor = TaskExecutor(llm_models={}, mcp_servers={}, default_model="qwen")

# Create workflow manager and executor
workflow_manager = LocalDirectoryWorkflowManager()
workflow_executor = WorkflowExecutor(task_executor, workflow_manager)


async def run_workflow():
    result = await workflow_executor.execute_workflow(
        workflow_name="research_analysis",
        inputs={"topic": "Quantum computing"},
    )
    print(f"Status: {result.status}")
    # Access step outputs
    if result.status == "completed":
        report = result.context.steps["generate_report"]["outputs"]["report"]
        print(f"Report: {report}")


asyncio.run(run_workflow())
