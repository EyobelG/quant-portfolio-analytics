import { useEffect, useRef, useState } from "react";
import type { AdvisorMessage } from "../types";

const PERSONA_LABELS: Record<string, string> = {
  financial_coordinator: "Coordinator",
  data_analyst_agent: "Data Analyst",
  growth_strategist_agent: "Growth Strategist",
  macro_risk_assessor_agent: "Macro Risk Assessor",
  value_investor_agent: "Value Investor",
};

function userIdKey() {
  const key = "advisor_user_id";
  let id = localStorage.getItem(key);
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem(key, id);
  }
  return id;
}

/** Parses one SSE frame's `data:` lines into an ADK Event, if present. */
function parseEventFrame(frame: string): { author?: string; text?: string } | null {
  const dataLines = frame
    .split("\n")
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trim());
  if (dataLines.length === 0) return null;
  try {
    const event = JSON.parse(dataLines.join("\n"));
    const parts = event?.content?.parts ?? [];
    const text = parts
      .map((p: { text?: string }) => p.text ?? "")
      .join("")
      .trim();
    if (!text) return null;
    return { author: event.author, text };
  } catch {
    return null;
  }
}

export default function AdvisorPanel({ apiBase }: { apiBase: string }) {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sessionError, setSessionError] = useState<string | null>(null);
  const [messages, setMessages] = useState<AdvisorMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [streaming, setStreaming] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const userId = useRef(userIdKey());

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(`${apiBase}/api/advisor/session`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ user_id: userId.current }),
        });
        if (!res.ok) throw new Error(`Request failed (${res.status})`);
        const body = await res.json();
        if (!cancelled) setSessionId(body.id);
      } catch (e) {
        if (!cancelled) {
          setSessionError(
            e instanceof Error ? e.message : "Could not reach the AI Portfolio Strategist"
          );
        }
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [apiBase]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  const send = async () => {
    const text = draft.trim();
    if (!text || !sessionId || streaming) return;
    setDraft("");
    setMessages((m) => [...m, { id: crypto.randomUUID(), author: "user", text }]);
    setStreaming(true);

    try {
      const res = await fetch(`${apiBase}/api/advisor/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: userId.current, session_id: sessionId, message: text }),
      });
      if (!res.ok || !res.body) throw new Error(`Request failed (${res.status})`);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        let sep;
        while ((sep = buffer.indexOf("\n\n")) !== -1) {
          const frame = buffer.slice(0, sep);
          buffer = buffer.slice(sep + 2);
          const parsed = parseEventFrame(frame);
          if (parsed?.text) {
            setMessages((m) => [
              ...m,
              { id: crypto.randomUUID(), author: parsed.author ?? "advisor", text: parsed.text! },
            ]);
          }
        }
      }
    } catch (e) {
      setMessages((m) => [
        ...m,
        {
          id: crypto.randomUUID(),
          author: "error",
          text: e instanceof Error ? e.message : "Something went wrong",
        },
      ]);
    } finally {
      setStreaming(false);
    }
  };

  return (
    <div className="panel advisor-panel">
      <h2>AI Portfolio Strategist</h2>
      <p className="panel-sub">
        Ask about one ticker. A data analyst pulls QuantFolio's own metrics for it, then a growth
        strategist, macro risk assessor, and value investor each argue their case from the same
        numbers.
      </p>

      {sessionError && <p className="advisor-error">{sessionError}</p>}

      <div className="advisor-thread" ref={scrollRef}>
        {messages.length === 0 && !sessionError && (
          <p className="advisor-empty">
            {sessionId ? "Try: “Analyze AAPL for a moderate, medium-term investor.”" : "Connecting…"}
          </p>
        )}
        {messages.map((m) => (
          <div key={m.id} className={`advisor-msg advisor-msg-${m.author === "user" ? "user" : "agent"}`}>
            {m.author !== "user" && (
              <div className="advisor-author">{PERSONA_LABELS[m.author] ?? m.author}</div>
            )}
            <div className="advisor-text">{m.text}</div>
          </div>
        ))}
        {streaming && <div className="advisor-typing">Thinking…</div>}
      </div>

      <div className="advisor-input-row">
        <textarea
          className="advisor-input"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
          placeholder="Ask about a ticker…"
          disabled={!sessionId || streaming}
          rows={2}
        />
        <button
          className="primary-btn"
          onClick={send}
          disabled={!sessionId || streaming || !draft.trim()}
        >
          {streaming ? "Sending…" : "Send"}
        </button>
      </div>

      <p className="advisor-disclaimer">
        Educational and informational purposes only. Not financial advice.
      </p>
    </div>
  );
}
