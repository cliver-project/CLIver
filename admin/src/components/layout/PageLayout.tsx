import { Breadcrumb } from "./Breadcrumb";

interface PageLayoutProps {
  title: string;
  breadcrumb?: { label: string; href?: string }[];
  actions?: React.ReactNode;
  children: React.ReactNode;
}

export function PageLayout({ title, breadcrumb, actions, children }: PageLayoutProps) {
  return (
    <div>
      {breadcrumb && breadcrumb.length > 1 && (
        <Breadcrumb items={breadcrumb} />
      )}
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-xl font-semibold text-foreground">{title}</h1>
        {actions && <div className="flex items-center gap-2">{actions}</div>}
      </div>
      {children}
    </div>
  );
}
