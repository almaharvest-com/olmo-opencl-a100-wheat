#!/usr/bin/env bash
# Connect to the Maahr VPN from WSL, checking the gateway certificate fresh every time.
# Run this in its own terminal and leave it open while you work. Ctrl+C disconnects.
HOST=maahr.fortiddns.com
PORT=10443
LAST="$HOME/.maahr_last_cert"

CERT=$(echo | openssl s_client -connect "$HOST:$PORT" -servername "$HOST" 2>/dev/null \
  | openssl x509 -noout -fingerprint -sha256 2>/dev/null \
  | cut -d= -f2 | tr -d ':' | tr 'A-F' 'a-f')

if [ -z "$CERT" ]; then
  echo "Could not read the gateway certificate. Check your internet connection." >&2
  exit 1
fi

echo "Gateway certificate fingerprint (SHA-256):"
echo "  $CERT"

if [ -f "$LAST" ] && [ "$(cat "$LAST")" = "$CERT" ]; then
  echo "Same as last time."
else
  [ -f "$LAST" ] && echo "WARNING: this certificate is different from the one you used last time."
  read -r -p "Trust this gateway certificate for this session? [y/N] " ans
  case "$ans" in
    y|Y) echo "$CERT" > "$LAST" ;;
    *) echo "Aborted."; exit 1 ;;
  esac
fi

# Asks for your sudo password, then the VPN account password.
exec sudo openfortivpn -c "$HOME/maahr.conf" --trusted-cert "$CERT"
