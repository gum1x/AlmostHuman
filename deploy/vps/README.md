Telegram Groups (Com_Chat, DWCusers_Chat)
      │
      ├── [Telethon ingestion → PostgreSQL: messages, is_deleted, reply_to]
      │
      └── ConversationScheduler (per-chat polling)
             │
             ├── [Fast Path: social/micro → _light_participation_reply → LocalLoRA → send]
             │
             └── [Full Path]
                    │
                    ├── Enrichment: VADER + topic keywords → EnrichedMessage[]
                    ├── Brief: tension (↑FIX VADER) + threads + drift
                    ├── Gate: velocity/fatigue/relationship/feedback → advisory score (↑EXPOSE
   TO MODEL)
                    │
                    ├── Context Bundle:
                    │     target + nearby + vector memories (pgvector cosine+importance)
                    │     + WHO I AM (persona core)
                    │     + MY SELF-REFLECTION (latest drift summary)
                    │     + MY RECENT ACTIVITY as me (last 6 replies + reasoning)
                    │     + MY CURRENT POSTURE (↑FIX: persist+retrieve this!)
                    │     + MY ENGAGEMENT SIGNALS (tension, outcome_24h, gate_score)
                    │
                    ├── Request 1 — Grok perception (↑EITHER USE OR REMOVE)
                    │     → context summary
                    │
                    ├── Request 2 — Grok smart participant (↑RAISE TEMP to 0.7-0.9)
                    │     System: SMART_PARTICIPANT_SYSTEM (rich character)
                    │     → {should_respond, plan, posture_update}
                    │
                    ├── LocalLoRA (HTTP) — phrases the plan in authentic voice
                    │
                    ├── Validate → send via Telethon
                    │
                    └── Feedback observation (45 min)
                          replies + reactions (↑FIX: actually fetch reactions!)
                          + follow-up sentiment → ResponseFeedback
                          → Self-reflection (6h/50 msgs) → persona update + relationships
                          → Meta-reflection (12h/10+ feedbacks) → stances + tone prefs

  ---
  🔬 Research Context