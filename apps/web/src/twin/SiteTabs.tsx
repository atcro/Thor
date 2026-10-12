import type { SiteTab } from "../api/types";

/** Greyed-out site tabs: Packaging is live; Molding and Batching are the scalability story. */
const DEFAULT_TABS: SiteTab[] = [
  { name: "Packaging", available: true, note: "" },
  { name: "Molding", available: false, note: "site tab only — not built" },
  { name: "Batching", available: false, note: "site tab only — not built" },
];

export function SiteTabs({ tabs }: { tabs?: SiteTab[] | null }) {
  const list = tabs && tabs.length ? tabs : DEFAULT_TABS;
  return (
    <div className="twin-sites" role="tablist" aria-label="Sites">
      {list.map((t) => (
        <button
          key={t.name}
          type="button"
          role="tab"
          className={`twin-site${t.available ? " on" : " off"}`}
          aria-selected={t.available}
          aria-disabled={!t.available}
          disabled={!t.available}
          title={t.available ? t.name : t.note || "not built"}
        >
          {t.name}
        </button>
      ))}
    </div>
  );
}
