import { useCallback, useEffect, useMemo, useState } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  addEdge,
  useNodesState,
  useEdgesState,
  MarkerType,
  type OnConnect,
  type Edge,
} from "@xyflow/react";
import dagre from "@dagrejs/dagre";
import "@xyflow/react/dist/style.css";

import { WorkflowNode, type WorkflowNodeData, type WorkflowNodeType } from "./workflow-node";
import { NodeDetailPanel } from "./node-detail-panel";
import { WorkflowToolbar } from "./workflow-toolbar";

const nodeTypes = { workflowNode: WorkflowNode };

interface Step {
  id: string;
  type: "llm" | "python";
  prompt?: string;
  model?: string;
  role?: string;
  output_format?: string;
  file?: string;
  depends_on?: string[];
  condition?: string;
  tools?: string[];
}

interface WorkflowCanvasProps {
  name: string;
  steps: Step[];
  models: string[];
  onSave: (steps: Step[]) => void;
  onRun: () => void;
  saving?: boolean;
}

function layoutNodes(steps: Step[]): { nodes: WorkflowNodeType[]; edges: Edge[] } {
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({ rankdir: "TB", nodesep: 60, ranksep: 80 });

  for (const step of steps) {
    g.setNode(step.id, { width: 160, height: 70 });
  }
  for (const step of steps) {
    for (const dep of step.depends_on ?? []) {
      g.setEdge(dep, step.id);
    }
  }
  dagre.layout(g);

  const nodes: WorkflowNodeType[] = steps.map((step) => {
    const pos = g.node(step.id);
    return {
      id: step.id,
      type: "workflowNode" as const,
      position: { x: pos.x - 80, y: pos.y - 35 },
      data: {
        stepId: step.id,
        type: step.type,
        model: step.model,
        role: step.role,
        outputFormat: step.output_format ?? "json",
        prompt: step.prompt,
        file: step.file,
        status: "pending",
      } satisfies WorkflowNodeData,
    };
  });

  const edges: Edge[] = [];
  for (const step of steps) {
    for (const dep of step.depends_on ?? []) {
      edges.push({
        id: `${dep}->${step.id}`,
        source: dep,
        target: step.id,
        markerEnd: { type: MarkerType.ArrowClosed, width: 16, height: 16 },
        style: { strokeWidth: 1.5 },
      });
    }
  }

  return { nodes, edges };
}

export function WorkflowCanvas({
  name,
  steps: initialSteps,
  models,
  onSave,
  onRun,
  saving,
}: WorkflowCanvasProps) {
  const [steps, setSteps] = useState<Step[]>(initialSteps);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);

  const { nodes: layoutedNodes, edges: layoutedEdges } = useMemo(
    () => layoutNodes(steps),
    [steps],
  );

  const [nodes, setNodes, onNodesChange] = useNodesState(layoutedNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(layoutedEdges);

  useEffect(() => {
    setNodes(layoutedNodes);
    setEdges(layoutedEdges);
  }, [layoutedNodes, layoutedEdges, setNodes, setEdges]);

  const onConnect: OnConnect = useCallback(
    (params) => {
      setEdges((eds) =>
        addEdge(
          {
            ...params,
            markerEnd: { type: MarkerType.ArrowClosed, width: 16, height: 16 },
            style: { strokeWidth: 1.5 },
          },
          eds,
        ),
      );
      if (params.source && params.target) {
        const targetId = params.target;
        const sourceId = params.source;
        setSteps((prev) =>
          prev.map((s) =>
            s.id === targetId
              ? {
                  ...s,
                  depends_on: [
                    ...(s.depends_on ?? []),
                    sourceId,
                  ].filter((v, i, a) => a.indexOf(v) === i),
                }
              : s,
          ),
        );
      }
    },
    [setEdges],
  );

  const onNodeClick = useCallback((_: React.MouseEvent, node: WorkflowNodeType) => {
    setSelectedNodeId(node.id);
  }, []);

  const selectedStep = steps.find((s) => s.id === selectedNodeId);
  const selectedNode = nodes.find((n) => n.id === selectedNodeId);

  function handleNodeDataChange(updates: Partial<WorkflowNodeData>) {
    if (!selectedNodeId) return;
    setSteps((prev) =>
      prev.map((s) => {
        if (s.id !== selectedNodeId) return s;
        const patched = { ...s };
        if (updates.model !== undefined) patched.model = updates.model;
        if (updates.role !== undefined) patched.role = updates.role;
        if (updates.outputFormat !== undefined) patched.output_format = updates.outputFormat;
        if (updates.prompt !== undefined) patched.prompt = updates.prompt;
        if (updates.file !== undefined) patched.file = updates.file;
        return patched;
      }),
    );
  }

  function handleAddNode(type: "llm" | "python") {
    const id = `new_step_${Date.now()}`;
    const newStep: Step =
      type === "llm"
        ? { id, type: "llm", prompt: "", output_format: "json", depends_on: [] }
        : { id, type: "python", file: "", depends_on: [] };
    setSteps((prev) => [...prev, newStep]);
    setSelectedNodeId(id);
  }

  function handleSave() {
    onSave(steps);
  }

  return (
    <div className="flex h-[calc(100vh-3.5rem)]">
      <div className="flex-1 flex flex-col">
        <WorkflowToolbar
          name={name}
          stepCount={steps.length}
          onAddNode={handleAddNode}
          onRun={onRun}
          onSave={handleSave}
          saving={saving}
        />
        <div className="flex-1">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeClick={onNodeClick}
            nodeTypes={nodeTypes}
            fitView
            proOptions={{ hideAttribution: true }}
            className="bg-background"
          >
            <Background gap={20} size={1} />
            <Controls />
            <MiniMap className="!bg-card !border-border" />
          </ReactFlow>
        </div>
      </div>

      {selectedStep && selectedNode && (
        <NodeDetailPanel
          data={selectedNode.data as WorkflowNodeData}
          dependsOn={selectedStep.depends_on ?? []}
          models={models}
          onChange={handleNodeDataChange}
          onClose={() => setSelectedNodeId(null)}
        />
      )}
    </div>
  );
}
