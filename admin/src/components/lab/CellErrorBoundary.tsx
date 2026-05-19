import { Component, type ReactNode } from "react";
import { AlertTriangle } from "lucide-react";

interface Props {
  cellTitle?: string;
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class CellErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="rounded-lg bg-red-50 border border-red-200 p-4 m-4">
          <div className="flex items-center gap-2 text-red-700 mb-1">
            <AlertTriangle className="w-4 h-4" />
            <span className="text-sm font-medium">
              {this.props.cellTitle ? `Error in "${this.props.cellTitle}"` : "Cell error"}
            </span>
          </div>
          <div className="text-xs text-red-600 font-mono">
            {this.state.error?.message || "Unknown error"}
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
