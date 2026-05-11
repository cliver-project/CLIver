import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  Controls,
  MiniMap,
  addEdge,
  useNodesState,
  useEdgesState,
  useReactFlow,
  reconnectEdge,
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

const PYTHON_SKELETON = `def run(inputs: dict, state: dict) -> dict:
    # inputs: upstream step results + workflow inputs
    # state: full workflow state (outputs_dir, workflow_id, etc.)
    # Add your code here
    return {"result": ""}
`;

interface Step {
  id: string;
  type: "llm" | "python";
  prompt?: string;
  agent?: string;
  output_format?: string;
  file?: string;
  code?: string;
  depends_on?: string[];
  condition?: string;
  tools?: string[];
}

interface LayoutData {
  nodes?: Record<string, { x: number; y: number }>;
  edges?: Record<string, { sourceHandle?: string; targetHandle?: string }>;
}

interface WorkflowCanvasProps {
  name: string;
  steps: Step[];
  agents: string[];
  layout?: LayoutData | null;
  stepStatuses?: Record<string, string>;
  stepOutputs?: Record<string, Record<string, unknown>>;
  outputsDir?: string;
  onSave: (steps: Step[], layout: LayoutData) => void;
  onRun: () => void;
  onRunStep?: (stepId: string) => void;
  onResumeFromStep?: (stepId: string) => void;
  onDelete?: () => void;
  saving?: boolean;
  saved?: boolean;
  running?: boolean;
  executions?: Array<Record<string, unknown>>;
  selectedExecutionId?: string | null;
  onSelectExecution?: (id: string) => void;
}

function buildNodeData(step: Step, onRename?: (o: string, n: string) => void): WorkflowNodeData {
  return {
    stepId: step.id,
    type: step.type,
    agent: step.agent,
    outputFormat: step.output_format ?? "json",
    prompt: step.prompt,
    file: step.file,
    code: step.code,
    status: "pending",
    onRename,
  };
}

function initialLayout(
  steps: Step[],
  savedLayout?: LayoutData | null,
): { nodes: WorkflowNodeType[]; edges: Edge[] } {
  let positions: Record<string, { x: number; y: number }>;
  const savedEdgeHandles = savedLayout?.edges ?? {};

  if (savedLayout?.nodes && Object.keys(savedLayout.nodes).length > 0) {
    positions = savedLayout.nodes;
  } else {
    const g = new dagre.graphlib.Graph();
    g.setDefaultEdgeLabel(() => ({}));
    g.setGraph({ rankdir: "TB", nodesep: 60, ranksep: 80 });
    for (const step of steps) g.setNode(step.id, { width: 160, height: 70 });
    for (const step of steps) {
      for (const dep of step.depends_on ?? []) g.setEdge(dep, step.id);
    }
    dagre.layout(g);
    positions = {};
    for (const step of steps) {
      const pos = g.node(step.id);
      positions[step.id] = { x: pos.x - 80, y: pos.y - 35 };
    }
  }

  const nodes: WorkflowNodeType[] = steps.map((step) => ({
    id: step.id,
    type: "workflowNode" as const,
    position: positions[step.id] ?? { x: 200, y: 100 },
    data: buildNodeData(step),
  }));

  const edges: Edge[] = [];
  for (const step of steps) {
    for (const dep of step.depends_on ?? []) {
      const edgeKey = `${dep}->${step.id}`;
      const savedHandles = savedEdgeHandles[edgeKey];
      edges.push({
        id: edgeKey,
        source: dep,
        sourceHandle: savedHandles?.sourceHandle ?? "bottom-source",
        target: step.id,
        targetHandle: savedHandles?.targetHandle ?? "top-target",
        markerEnd: { type: MarkerType.ArrowClosed, width: 16, height: 16 },
        style: { strokeWidth: 1.5 },
      });
    }
  }

  return { nodes, edges };
}

export function WorkflowCanvas(props: WorkflowCanvasProps) {
  return (
    <ReactFlowProvider>
      <WorkflowCanvasInner {...props} />
    </ReactFlowProvider>
  );
}

interface DropMenu { x: number; y: number; sourceNodeId: string; sourceHandleId: string | null }

function WorkflowCanvasInner({
  name,
  steps: initialSteps,
  agents,
  layout: savedLayout,
  stepStatuses,
  stepOutputs,
  outputsDir,
  onSave,
  onRun,
  onRunStep,
  onResumeFromStep,
  onDelete,
  saving,
  saved,
  running,
  executions,
  selectedExecutionId,
  onSelectExecution,
}: WorkflowCanvasProps) {
  const [steps, setSteps] = useState<Step[]>(initialSteps);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [dropMenu, setDropMenu] = useState<DropMenu | null>(null);
  const connectingRef = useRef<{ nodeId: string; handleId: string | null }>({ nodeId: "", handleId: null });
  const dropMenuSetAt = useRef(0);
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const { screenToFlowPosition } = useReactFlow();

  // Layout: use saved positions if available, otherwise auto-layout via dagre
  const { nodes: initNodes, edges: initEdges } = useMemo(
    () => initialLayout(initialSteps, savedLayout),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );

  const [nodes, setNodes, onNodesChange] = useNodesState(initNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initEdges);

  // Update node statuses when stepStatuses changes (WITHOUT re-layout)
  useEffect(() => {
    if (!stepStatuses) return;
    setNodes((nds) =>
      nds.map((n) => ({
        ...n,
        data: { ...n.data, status: stepStatuses[n.id] ?? "pending" },
      })),
    );
  }, [stepStatuses, setNodes]);

  // --- Rename (updates steps + nodes + edges in-place) ---

  const handleRename = useCallback((oldId: string, newId: string) => {
    setSteps((prev) => {
      if (prev.some((s) => s.id === newId)) return prev;
      return prev.map((s) => {
        const updated = s.id === oldId ? { ...s, id: newId } : { ...s };
        if (updated.depends_on) {
          updated.depends_on = updated.depends_on.map((d) => (d === oldId ? newId : d));
        }
        return updated;
      });
    });
    setNodes((nds) =>
      nds.map((n) =>
        n.id === oldId
          ? { ...n, id: newId, data: { ...n.data, stepId: newId } }
          : n,
      ),
    );
    setEdges((eds) =>
      eds.map((e) => ({
        ...e,
        id: e.id.replace(oldId, newId),
        source: e.source === oldId ? newId : e.source,
        target: e.target === oldId ? newId : e.target,
      })),
    );
    if (selectedNodeId === oldId) setSelectedNodeId(newId);
  }, [selectedNodeId, setNodes, setEdges]);

  // Inject onRename callback into node data once
  useEffect(() => {
    setNodes((nds) =>
      nds.map((n) => ({
        ...n,
        data: { ...n.data, onRename: handleRename },
      })),
    );
  }, [handleRename, setNodes]);

  // --- Connect (only one edge per node pair) ---

  const onConnect: OnConnect = useCallback(
    (params) => {
      if (!params.source || !params.target) return;
      if (params.source === params.target) return;

      // Prevent duplicate edges between same node pair
      const exists = edges.some(
        (e) =>
          (e.source === params.source && e.target === params.target) ||
          (e.source === params.target && e.target === params.source),
      );
      if (exists) return;

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
    },
    [setEdges, edges],
  );

  // --- Reconnect: drag edge endpoint to a different handle ---

  const onReconnect = useCallback(
    (oldEdge: Edge, newConnection: { source: string; target: string; sourceHandle: string | null; targetHandle: string | null }) => {
      // Update the edge visually
      setEdges((eds) => reconnectEdge(oldEdge, newConnection, eds));

      // Sync depends_on if the source or target node changed
      if (oldEdge.source !== newConnection.source || oldEdge.target !== newConnection.target) {
        setSteps((prev) =>
          prev.map((s) => {
            let deps = [...(s.depends_on ?? [])];
            if (s.id === oldEdge.target) {
              deps = deps.filter((d) => d !== oldEdge.source);
            }
            if (s.id === newConnection.target && !deps.includes(newConnection.source)) {
              deps.push(newConnection.source);
            }
            return { ...s, depends_on: deps };
          }),
        );
      }
    },
    [setEdges],
  );

  // --- Delete edges: sync back to steps.depends_on ---

  const handleEdgesChange = useCallback(
    (changes: Parameters<typeof onEdgesChange>[0]) => {
      const removals = changes.filter((c) => c.type === "remove").map((c) => c.id);
      if (removals.length > 0) {
        const removedEdges = edges.filter((e) => removals.includes(e.id));
        setSteps((prev) =>
          prev.map((s) => {
            const toRemove = removedEdges
              .filter((e) => e.target === s.id)
              .map((e) => e.source);
            if (toRemove.length === 0) return s;
            return {
              ...s,
              depends_on: (s.depends_on ?? []).filter((d) => !toRemove.includes(d)),
            };
          }),
        );
      }
      onEdgesChange(changes);
    },
    [onEdgesChange, edges],
  );

  // --- Delete nodes: sync back to steps ---

  const handleNodesChange = useCallback(
    (changes: Parameters<typeof onNodesChange>[0]) => {
      const removals = changes.filter((c) => c.type === "remove").map((c) => c.id);
      if (removals.length > 0) {
        setSteps((prev) =>
          prev
            .filter((s) => !removals.includes(s.id))
            .map((s) => ({
              ...s,
              depends_on: (s.depends_on ?? []).filter((d) => !removals.includes(d)),
            })),
        );
        if (selectedNodeId && removals.includes(selectedNodeId)) {
          setSelectedNodeId(null);
        }
      }
      onNodesChange(changes);
    },
    [onNodesChange, selectedNodeId],
  );

  // --- Connect start/end for drop-to-create ---

  const onConnectStart = useCallback((_: React.MouseEvent | React.TouchEvent, params: { nodeId: string | null; handleId: string | null }) => {
    connectingRef.current = { nodeId: params.nodeId ?? "", handleId: params.handleId };
  }, []);

  const onConnectEnd = useCallback((event: MouseEvent | TouchEvent) => {
    const target = event.target as HTMLElement;
    if (target.closest(".react-flow__handle") || target.closest(".react-flow__node")) return;
    if (!connectingRef.current.nodeId) return;

    const clientX = "changedTouches" in event ? event.changedTouches[0].clientX : event.clientX;
    const clientY = "changedTouches" in event ? event.changedTouches[0].clientY : event.clientY;

    dropMenuSetAt.current = Date.now();
    setDropMenu({
      x: clientX,
      y: clientY,
      sourceNodeId: connectingRef.current.nodeId,
      sourceHandleId: connectingRef.current.handleId,
    });
  }, []);

  const handleDropCreate = useCallback((type: "llm" | "python") => {
    if (!dropMenu) return;
    const position = screenToFlowPosition({ x: dropMenu.x, y: dropMenu.y });
    const id = `step_${Date.now()}`;
    const newStep: Step =
      type === "llm"
        ? { id, type: "llm", prompt: "", output_format: "json", depends_on: [dropMenu.sourceNodeId] }
        : { id, type: "python", code: PYTHON_SKELETON, depends_on: [dropMenu.sourceNodeId] };
    setSteps((prev) => [...prev, newStep]);
    setSelectedNodeId(id);

    setNodes((nds) => [
      ...nds,
      {
        id,
        type: "workflowNode" as const,
        position,
        data: buildNodeData(newStep, handleRename),
      },
    ]);

    setEdges((eds) =>
      addEdge(
        {
          source: dropMenu.sourceNodeId,
          sourceHandle: dropMenu.sourceHandleId,
          target: id,
          targetHandle: null,
          markerEnd: { type: MarkerType.ArrowClosed, width: 16, height: 16 },
          style: { strokeWidth: 1.5 },
        },
        eds,
      ),
    );

    setDropMenu(null);
  }, [dropMenu, screenToFlowPosition, handleRename, setNodes, setEdges]);

  // --- Node click / data change ---

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
        if (updates.agent !== undefined) patched.agent = updates.agent;
        if (updates.outputFormat !== undefined) patched.output_format = updates.outputFormat;
        if (updates.prompt !== undefined) patched.prompt = updates.prompt;
        if (updates.file !== undefined) patched.file = updates.file;
        if (updates.code !== undefined) patched.code = updates.code;
        return patched;
      }),
    );
    // Update node data in-place (no re-layout)
    setNodes((nds) =>
      nds.map((n) =>
        n.id === selectedNodeId
          ? { ...n, data: { ...n.data, ...updates } }
          : n,
      ),
    );
  }

  function handleAddNode(type: "llm" | "python") {
    const id = `step_${Date.now()}`;
    const newStep: Step =
      type === "llm"
        ? { id, type: "llm", prompt: "", output_format: "json", depends_on: [] }
        : { id, type: "python", code: PYTHON_SKELETON, depends_on: [] };
    setSteps((prev) => [...prev, newStep]);
    setSelectedNodeId(id);

    // Place new node below the last node
    const lastNode = nodes[nodes.length - 1];
    const position = lastNode
      ? { x: lastNode.position.x, y: lastNode.position.y + 120 }
      : { x: 200, y: 100 };

    setNodes((nds) => [
      ...nds,
      {
        id,
        type: "workflowNode" as const,
        position,
        data: buildNodeData(newStep, handleRename),
      },
    ]);
  }

  function handleSave() {
    // Sync depends_on from current edges
    const edgeMap = new Map<string, string[]>();
    for (const e of edges) {
      const deps = edgeMap.get(e.target) ?? [];
      deps.push(e.source);
      edgeMap.set(e.target, deps);
    }
    const synced = steps.map((s) => ({
      ...s,
      depends_on: edgeMap.get(s.id) ?? [],
    }));

    // Collect current node positions
    const nodePositions: Record<string, { x: number; y: number }> = {};
    for (const n of nodes) {
      nodePositions[n.id] = { x: Math.round(n.position.x), y: Math.round(n.position.y) };
    }

    // Collect edge handle info — keyed by "source->target" for stable lookup
    const edgeHandles: Record<string, { sourceHandle?: string; targetHandle?: string }> = {};
    for (const e of edges) {
      const key = `${e.source}->${e.target}`;
      edgeHandles[key] = {
        sourceHandle: e.sourceHandle ?? undefined,
        targetHandle: e.targetHandle ?? undefined,
      };
    }

    onSave(synced, { nodes: nodePositions, edges: edgeHandles });
  }

  return (
    <div className="flex h-[calc(100vh-3.5rem)]">
      <div className="flex-1 flex flex-col">
        <WorkflowToolbar
          name={name}
          stepCount={steps.length}
          outputsDir={outputsDir}
          onAddNode={handleAddNode}
          onRun={onRun}
          onSave={handleSave}
          onDelete={onDelete}
          saving={saving}
          saved={saved}
          running={running}
          executions={executions}
          selectedExecutionId={selectedExecutionId}
          onSelectExecution={onSelectExecution}
        />
        <div className="flex-1 relative" ref={reactFlowWrapper}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={handleNodesChange}
            onEdgesChange={handleEdgesChange}
            onConnect={onConnect}
            onReconnect={onReconnect}
            onConnectStart={onConnectStart}
            onConnectEnd={onConnectEnd}
            onNodeClick={onNodeClick}
            onPaneClick={() => { if (Date.now() - dropMenuSetAt.current > 200) setDropMenu(null); }}
            nodeTypes={nodeTypes}
            edgesReconnectable
            deleteKeyCode={["Backspace", "Delete"]}
            fitView
            proOptions={{ hideAttribution: true }}
            className="bg-background"
          >
            <Background gap={20} size={1} />
            <Controls />
            <MiniMap className="!bg-card !border-border" />
          </ReactFlow>

          {dropMenu && (
            <div
              className="absolute z-50 bg-card border border-border rounded-lg shadow-lg py-1 min-w-[140px]"
              style={{ left: dropMenu.x - (reactFlowWrapper.current?.getBoundingClientRect().left ?? 0), top: dropMenu.y - (reactFlowWrapper.current?.getBoundingClientRect().top ?? 0) }}
            >
              <button
                className="w-full text-left px-3 py-1.5 text-sm hover:bg-accent flex items-center gap-2"
                onClick={() => handleDropCreate("llm")}
              >
                <span className="w-2 h-2 rounded-full" style={{ background: "#818cf8" }} />
                LLM Step
              </button>
              <button
                className="w-full text-left px-3 py-1.5 text-sm hover:bg-accent flex items-center gap-2"
                onClick={() => handleDropCreate("python")}
              >
                <span className="w-2 h-2 rounded-full" style={{ background: "#34d399" }} />
                Python Step
              </button>
            </div>
          )}
        </div>
      </div>

      {selectedStep && selectedNode && (
        <NodeDetailPanel
          data={selectedNode.data as WorkflowNodeData}
          dependsOn={selectedStep.depends_on ?? []}
          agents={agents}
          stepOutput={stepOutputs?.[selectedNodeId!] as Record<string, unknown> | undefined}
          onChange={handleNodeDataChange}
          onRename={handleRename}
          onRunStep={onRunStep ? () => onRunStep(selectedNodeId!) : undefined}
          onResumeFromStep={onResumeFromStep ? () => onResumeFromStep(selectedNodeId!) : undefined}
          onClose={() => setSelectedNodeId(null)}
        />
      )}
    </div>
  );
}
