# Hosting a Website from This Mac

This folder (`web/`) contains a simple static website that demonstrates hosting directly from your Mac.

## Quick Start (Local Network)

```bash
# From inside this directory
cd ~/Research/web

# Option 1: Python (easiest)
python3 -m http.server 8080

# Option 2: With Docker + Nginx (more production-like)
docker run --rm -p 8080:80 -v "$(pwd)":/usr/share/nginx/html:ro nginx:alpine
```

Then visit:

- From this Mac: http://localhost:8080
- From other devices on your Wi-Fi: http://<LAN_IP>:8080

## Making it Publicly Accessible

### Recommended: ngrok (fastest)

1. `brew install ngrok`
2. `ngrok http 8080`
3. Copy the `https://*.ngrok.io` URL and share it.

Works instantly, handles HTTPS, and no router configuration.

### Using Tailscale (you already have it)

Your Tailscale IP is `<TAILSCALE_IP>`.

Anyone on your Tailscale network can visit:
http://<TAILSCALE_IP>:8080

Very secure (no public exposure).

### More Permanent Option: Cloudflare Tunnel

```bash
brew install cloudflare/cloudflare/cloudflared
cloudflared tunnel --url http://localhost:8080
```

Gives you a stable `*.trycloudflare.com` address.

## Connecting to Your Telegram CI API

The site tries to call your local API at `http://localhost:8000/health`.

To make the API available:

```bash
# In the project root
cd ~/Research
docker compose up -d
```

Then refresh the page — the status indicator should turn green.

## Keeping the Mac Awake

To prevent the Mac from sleeping while serving:

```bash
caffeinate -s
```

Or install `caffeinate` wrapper / use Amphetamine app.

## Limitations

- Your Mac must stay on and connected to the internet.
- Home internet upload speeds are usually slow.
- Dynamic IP (use Tailscale/ngrok/Cloudflare Tunnel instead of port forwarding).
- Not recommended for high-traffic production sites (get a cheap VPS or use Vercel/Netlify/Cloudflare Pages for static).

## Next Steps

Want a real dashboard for your Telegram messages?

I can help you build a proper frontend that:
- Lists recent messages from the `/messages/{chat_id}` endpoint
- Shows bot decisions
- Has filters, search, etc.

Just say the word.

---

