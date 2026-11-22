# Sharing your models with guests

Instinct can let people anywhere use the models on your Mac: Gemma 4 and the
Qwen3.5 sub-agents, with the full agent harness. Guests never get anything
else. Their files and chats stay on their own computer, and you can cut anyone
off at any time.

## Giving someone access (for example, an interviewer)

1. **Once:** install [Tailscale](https://tailscale.com/download) and sign in.
2. In Instinct, open **Settings › Guests** and turn on **Share on the
   internet**. The first time, Tailscale opens a page asking you to allow
   HTTPS certificates and Funnel for your tailnet. Approve it, then turn
   sharing off and on again. The pane shows your public address,
   `https://<your-mac>.<tailnet>.ts.net`.
3. Under **Guests**, type a name ("Interviewer"), pick how long the invite lasts
   and how many requests it may run at once, and click **Create invite**.
   Copy the link: it's shown once.
4. Send the link. The guest opens it in Chrome, Edge, Arc or Brave and clicks
   **Choose a folder** to pick a folder on their computer. They can then ask
   the agents to write code there, solve math, draft writing, and so on. They
   choose per message whether the agents may only read, must ask before each
   change, or may change files freely.
5. When you're done, click **Revoke**. Their browser is signed out at once,
   and anything they had running stops.

A guest who has the Instinct app can paste the same link into **Settings ›
Use another Mac** instead. Their app then keeps its own projects, terminal and
sandboxed commands, and only borrows your models.

## What guests can and can't reach

| | Guests | You |
| --- | --- | --- |
| Gemma 4 and the Qwen3.5 sub-agents | ✓, within their invite's limits | ✓ |
| Other models (the abliterated Gemma) | | ✓ |
| Your workspaces, chats, terminal, files | | ✓ |
| The `/v1` API and API keys | | ✓, on this Mac only |
| Their own folder and chats | ✓, on their computer | |

**How the separation works:**

- **Your app.** Its server listens only on 127.0.0.1 and accepts nothing but
  the Mac app's one-time sign-in.
- **The guest listener.** Sharing starts a second listener, also on
  127.0.0.1. It's a separate server with its own routes and cookies, and it
  serves only the guest page and model calls. Tailscale Funnel publishes that
  one port with HTTPS. When sharing is off, nothing on your Mac is reachable
  from outside.
- **Browser guests.** The agents run on your Mac, but each file operation
  (list, read, search, write, edit, move, delete) is sent to the guest's
  browser and carried out there on their folder. The run's journal is kept in
  memory and dropped when the run ends. Nothing a guest sends is logged or
  written to disk here.
- **Mac app guests.** Their whole harness runs on their Mac. Only model calls
  come to yours.
- **What your Mac keeps.** Each invite's name, limits, expiry and token count.
  The invite token itself is stored only as a hash.

## Invites

| Setting | Default | Range |
| --- | --- | --- |
| Lasts | 1 day | up to 90 days |
| Requests at once | 2 | 1–16 |
| Tokens a day | 2 million | 10 thousand – 1 billion |
| Context | 65,536 tokens | 4,096–262,144 |
| Reply length | 8,192 tokens | 256–32,768 |

**Link and storage.** The invite sits after the `#` in the link, which
browsers never send to a server, and the guest page removes it from the
address bar after signing in. On your Mac, invites live in
`~/Library/Application Support/Instinct/guest-invites.json` (mode 0600).

**Revoking or removing.** Revoking keeps the record so you can see past use.
The × removes the record entirely.

## Many guests at once

**Batching.** Guests share the same model workers as your own chats. Each
model step batches every active request, so guests don't queue behind each
other one at a time. Each worker takes up to 16 requests at once.

**Measured on an M3 Max.** 16 simultaneous users on Gemma, each holding a
three-turn chat over a 4,000-token document:

| Measure | Result |
| --- | --- |
| Total throughput | 78 tokens/s |
| Prompt reused from cache on later turns | 99% |
| Later turns' first token (p50) | 0.4 s |
| Gemma worker's peak memory | 26 GB |

**Limits on load.**
- Each invite's requests-at-once limit caps a single guest.
- The host runs at most 24 guest runs at once.
- Model workers together are held under 105 GB of memory (see
  [models.md](models.md#memory-ceiling)).

## Troubleshooting

- **"Tailscale isn't installed"**: install it from tailscale.com/download and
  sign in, then turn sharing off and on.
- **The pane shows an error from Tailscale**: usually a link to enable HTTPS
  or Funnel for your tailnet. Open it, approve, and toggle sharing again.
  Funnel needs Tailscale 1.52 or later.
- **The guest sees "This browser can't open folders"**: they're on Safari or
  Firefox. Chatting works, but file work needs a Chromium browser or the
  Instinct app.
- **"This invite has used its tokens for today"**: make a new invite with a
  larger daily limit, or wait for the day to roll over (UTC).
