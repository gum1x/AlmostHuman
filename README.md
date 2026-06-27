# AlmostHuman

Can an AI chatbot embed itself in adversarial, high suspicion online communities, build real rapport, and pass as human for weeks on end? Turns out, yes. This one did. 

Self hosted autonomous group chat participant that ingests a live Telegram group over MTProto, decides *whether* to speak with a machine learning classifier trained on which messages real members actually reply to, before the expensive perception and decision calls, reasons about *what* to say with an LLM, rewrites the result in a fine-tuned local voice, sends it back, and grades how each message landed 45 minutes later so the next decision gets smarter. It tracks mood, memory, and relationships across cycles, so it reads as a real member. 

**For obvious reasons, the fined tuned voice model is not included publicily in this repository**

## By the numbers

- **~162k messages** of real group history (≈3 weeks) train and calibrate the when-to-respond model.
- **11-feature logistic-regression** classifier, fit on a **time-ordered 60/20/20 split** (no future leakage) with **isotonic-calibrated** probabilities.
- **One tunable threshold** sets how chatty it is. At its shipped **~6% reply cadence**, the large majority of the unprompted firehose is dropped before the expensive perception and decision calls.
- Perception compresses up to **200 messages** per decision into a **6k-token** context budget.
- Every message it sends is **scored 45 minutes later**, and that grade feeds back into the gate.
- Local ranking is free: **384-dim** MiniLM embeddings + VADER sentiment, zero API calls.


## License

No license file is included. This is a private research project, not licensed for redistribution.
