export default function Logo({ size = 34 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 40 40"
      fill="none"
      aria-hidden="true"
      className="logo-mark"
    >
      <defs>
        <linearGradient id="logo-bg" x1="0" y1="0" x2="40" y2="40">
          <stop offset="0%" stopColor="#4f7cff" />
          <stop offset="100%" stopColor="#22d3ee" />
        </linearGradient>
      </defs>
      <rect width="40" height="40" rx="11" fill="url(#logo-bg)" />
      {/* An efficient-frontier curve rising to the right, with the optimal point marked. */}
      <path
        d="M9 29.5C13.5 29.5 17.5 25.5 20 21C22.5 16.5 26.5 12.5 31 12.5"
        stroke="white"
        strokeWidth="2.4"
        strokeLinecap="round"
        opacity="0.95"
      />
      <circle cx="20" cy="21" r="2.9" fill="white" />
      <circle cx="31" cy="12.5" r="2" fill="white" opacity="0.75" />
      <circle cx="9" cy="29.5" r="2" fill="white" opacity="0.75" />
    </svg>
  );
}
