export function CliverLogo({ size = 32 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 100 100"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className="text-primary"
    >

      <line x1="18" y1="22" x2="48" y2="36" stroke="currentColor" strokeWidth="0.8" opacity="0.12" />
      <line x1="18" y1="30" x2="60" y2="50" stroke="currentColor" strokeWidth="0.8" opacity="0.1" />
      <line x1="18" y1="70" x2="60" y2="50" stroke="currentColor" strokeWidth="0.8" opacity="0.1" />
      <line x1="18" y1="78" x2="48" y2="64" stroke="currentColor" strokeWidth="0.8" opacity="0.12" />

      <line x1="18" y1="18" x2="34" y2="28" stroke="currentColor" strokeWidth="1.8" opacity="0.35" />
      <line x1="34" y1="28" x2="48" y2="38" stroke="currentColor" strokeWidth="2" opacity="0.5" />
      <line x1="48" y1="38" x2="60" y2="50" stroke="currentColor" strokeWidth="2.2" opacity="0.7" />
      <line x1="60" y1="50" x2="48" y2="62" stroke="currentColor" strokeWidth="2.2" opacity="0.7" />
      <line x1="48" y1="62" x2="34" y2="72" stroke="currentColor" strokeWidth="2" opacity="0.5" />
      <line x1="34" y1="72" x2="18" y2="82" stroke="currentColor" strokeWidth="1.8" opacity="0.35" />

      <line x1="34" y1="28" x2="34" y2="72" stroke="currentColor" strokeWidth="0.8" opacity="0.15" />
      <line x1="48" y1="38" x2="48" y2="62" stroke="currentColor" strokeWidth="1" opacity="0.2" />

      <circle cx="18" cy="30" r="2" fill="currentColor" opacity="0.25" />
      <circle cx="18" cy="70" r="2" fill="currentColor" opacity="0.25" />

      <circle cx="18" cy="18" r="3.5" fill="currentColor" opacity="0.5" />
      <circle cx="34" cy="28" r="4" fill="currentColor" opacity="0.65" />
      <circle cx="48" cy="38" r="4.5" fill="currentColor" opacity="0.8" />
      <circle cx="48" cy="62" r="4.5" fill="currentColor" opacity="0.8" />
      <circle cx="34" cy="72" r="4" fill="currentColor" opacity="0.65" />
      <circle cx="18" cy="82" r="3.5" fill="currentColor" opacity="0.5" />

      <circle cx="60" cy="50" r="7" fill="currentColor" />
      <circle cx="60" cy="50" r="3.5" className="fill-sidebar-background" />
      <circle cx="60" cy="50" r="1.5" fill="currentColor" opacity="0.5" />

      <rect x="72" y="47.5" width="12" height="3.5" rx="1.5" fill="currentColor" opacity="0.8">
        <animate attributeName="opacity" values="0.8;0.15;0.8" dur="1.2s" repeatCount="indefinite" />
      </rect>
    </svg>
  );
}
