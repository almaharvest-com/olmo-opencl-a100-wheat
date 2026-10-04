# Maahr cluster VPN from WSL: connect, use, disconnect

Written 2026-10-02, **updated 2026-10-04** (no more pinned certificate: the connect script checks the gateway's current certificate every time). The VPN to the Maahr GPU cluster runs **inside WSL Ubuntu** with `openfortivpn`, instead of FortiClient on Windows. Windows keeps its normal network and only WSL talks to the cluster.

Verified working on 2026-10-02: the tunnel came up, and `10.172.0.233` (g13) answered on SSH port 22.

---

## 0. What changed on 2026-10-04 (and why)

The gateway (`maahr.fortiddns.com`) renews its self-signed certificate often. The old config pinned one fixed fingerprint (`trusted-cert = ...`), so every renewal made the connection fail with `Gateway certificate validation failed`.

Now:
- `~/maahr.conf` has **no `trusted-cert` line**.
- A small script, `~/maahr-vpn.sh`, reads the gateway's **current** certificate fingerprint at connect time, shows it, and passes it to `openfortivpn` for that one session only.
- You still type the **VPN password at every connection** (it is never stored in any file). The username is in the config.

Trade-off: the fingerprint is no longer pinned permanently, so the script trusts whatever certificate the gateway presents right now. To keep some protection, the script remembers the last fingerprint it saw and **asks you to confirm whenever it changes**. A change is normal after a renewal, but if you are on an unknown or public network, or the admins did not mention a renewal, answer `n` and ask them.

---

## 1. How it is set up (one time)

| Piece | Details |
|---|---|
| Client | `openfortivpn` 1.21.0 (+ `ppp`), installed with `sudo apt install openfortivpn` |
| Kernel | WSL2 kernel 6.6.87 already has PPP built in (`/dev/ppp`, `/dev/net/tun` exist), so no extra drivers |
| Gateway | `maahr.fortiddns.com`, port `10443` (same as the FortiClient setup in the Maahr guide) |
| Config file | `~/maahr.conf` (permissions `600`, **no password and no certificate stored**) |
| Connect script | `~/maahr-vpn.sh` (executable) |
| apt mirror | `https://mirrors.edge.kernel.org/ubuntu/` (your ISP serves a stale cache for `archive.ubuntu.com`) |

### 1a. Update the config (remove the pinned certificate)

Run once in Ubuntu:

```bash
sed -i '/^trusted-cert/d' ~/maahr.conf
cat ~/maahr.conf
```

The file should now contain exactly:

```
host = maahr.fortiddns.com
port = 10443
username = almauser1
set-dns = 0
```

- `set-dns = 0` stops the VPN from rewriting WSL's DNS settings. We connect by IP address.
- Do **not** add a `password =` line. Keep typing it at the prompt.

### 1b. Create the connect script

Run once in Ubuntu (this writes the file, then makes it executable):

```bash
cat > ~/maahr-vpn.sh <<'EOF'
#!/usr/bin/env bash
# Connect to the Maahr VPN, checking the gateway certificate fresh every time.
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
EOF
chmod +x ~/maahr-vpn.sh
```

---

## 2. Connect (every time)

**Before you start:** make sure **FortiClient on Windows is disconnected**. Two VPN connections with the same account can conflict.

1. Open an Ubuntu terminal. This will be your **VPN terminal**, and it has to stay open while you work.
2. Start the tunnel:

```bash
~/maahr-vpn.sh
```

3. The script prints the gateway's current fingerprint. On the very first run, or after a certificate renewal, it asks `Trust this gateway certificate for this session? [y/N]`. Answer `y` if you expect it (see section 0).
4. It then asks for **two passwords** in this order: first your Ubuntu `sudo` password, then `VPN account password:`, which is your Maahr VPN password (the same one you use for SSH).
5. Wait for these lines (repeated "Negotiation complete" lines are normal):

```
INFO:   Interface ppp0 is UP.
INFO:   Setting new routes...
INFO:   Tunnel is up and running.
```

6. Leave that terminal alone. It shows the connection while it lives.

---

## 3. Use it

Open a **second** Ubuntu terminal for your work.

Check that the cluster answers:

```bash
timeout 5 bash -c '</dev/tcp/10.172.0.233/22' && echo "g13 ssh port reachable"
```

Log in to the GPU node g13 (the admin said to use this one directly):

```bash
ssh almauser1@10.172.0.233
```

The login node is `ssh almauser1@10.172.2.37` (mgt1), but the admin asked us not to work there.

Copy files to the cluster (run from WSL; your Windows drives are under `/mnt/c` and `/mnt/d`):

```bash
scp /mnt/d/work/alma/somefile.tgz almauser1@10.172.0.233:~/
```

Notes:
- **Run `ssh`/`scp` from WSL, not from PowerShell.** Windows is not on the VPN, so PowerShell can no longer reach `10.172.x.x`.
- Use **IP addresses**, not host names. The cluster's own DNS is not applied in WSL (`set-dns = 0`).

---

## 4. Disconnect

**Normal way:** go to the VPN terminal and press **Ctrl+C**. You should see it clean up and end with a message like `Closed connection to gateway`. Then the terminal returns to a prompt.

**Check that it's really off** (from any Ubuntu terminal):

```bash
pgrep -a openfortivpn || echo "openfortivpn not running"
```

```bash
ip -br addr show ppp0 2>&1 | head -1
```

The second command should say `Device "ppp0" does not exist`.

**If you lost the VPN terminal** (closed the tab, or it hangs), stop it from another terminal:

```bash
sudo pkill -INT openfortivpn
```

If `ppp0` still exists afterwards:

```bash
sudo pkill pppd
```

**Last resort:** in Windows PowerShell, `wsl --shutdown` kills everything in WSL, including the tunnel. Docker Desktop restarts WSL by itself afterwards.

Always disconnect when you are done for the day. The tunnel also drops when the laptop sleeps, so reconnect with section 2 afterwards.

---

## 5. What goes through the tunnel (observed)

While connected, `ip route` shows the gateway sends **only specific cluster addresses** through `ppp0` (the `10.172.0.x`, `10.172.1.x`, `10.172.2.x` and `10.177.0.x` machines, including g13 `10.172.0.233` and mgt1 `10.172.2.37`). The default route stays on the normal connection, so:

- Normal internet in WSL keeps working while the VPN is up (`pypi.org` and `github.com` both returned HTTP 200 during the test).
- Windows is completely unaffected.
- A cluster machine whose address is **not** in that route list cannot be reached. Check with `ip route get <ip>`: `dev ppp0` means it goes through the VPN, `dev eth0` means it does not. Ask the admins to add routes if you need other machines.

You can look at the routes yourself:

```bash
ip route | grep ppp0 | head
```

---

## 6. Troubleshooting

| What you see | What it means / what to do |
|---|---|
| Script asks `Trust this gateway certificate...` with a **WARNING: different from last time** | The gateway probably renewed its certificate (it does this often). Fine on your home or office network. On public or unknown Wi-Fi, or if you doubt it, answer `n` and ask the admins. |
| `Could not read the gateway certificate` | No internet, or `maahr.fortiddns.com:10443` is blocked or down. Check that your normal internet works, then retry later or from another network. |
| `Gateway certificate validation failed` | You ran `openfortivpn` directly instead of the script. Use `~/maahr-vpn.sh`. |
| `Could not authenticate to gateway` / `Authentication failed` | Wrong VPN password or username. Retry. The username is in `~/maahr.conf` (change `username =` if yours differs). Do not put the password in the file. |
| `Could not connect` / timeouts to `maahr.fortiddns.com` | No internet on Windows or a firewall in between. Try again later or from another network. |
| `pppd: This system lacks kernel support for PPP` | Should not happen (the WSL kernel has it). Run `wsl --shutdown` in PowerShell and open Ubuntu again; if it persists, tell me. |
| `ppp0` already exists / second start fails | A previous run is still active. Run `pgrep -a openfortivpn`, then the stop commands in section 4. |
| Tunnel is up but `ssh` hangs or times out | The target IP may not be routed through the VPN (see `ip route get <ip>`), the node may be down, or you are in PowerShell instead of WSL. |
| `ssh: Could not resolve hostname g13` | Host names are not resolved. Use the IP, `10.172.0.233`. |
| `WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED` | The node was reinstalled or the IP is now another machine. Confirm with the admins before running `ssh-keygen -R <ip>`. |
| `sudo: openfortivpn: command not found` | The package got removed. Reinstall: `sudo apt install openfortivpn` (use the kernel.org mirror in section 1 if apt fails). |
| `openssl: command not found` | Install it: `sudo apt install openssl`. |

To debug a failed connection, run the client verbosely with the fingerprint the script printed (never share the password):

```bash
sudo openfortivpn -c ~/maahr.conf --trusted-cert <fingerprint> -v
```

---

## 7. Optional shortcuts

Two aliases so that you type `maahr-vpn` to connect and `maahr-vpn-off` to disconnect:

```bash
echo "alias maahr-vpn='~/maahr-vpn.sh'" >> ~/.bashrc
```

```bash
echo "alias maahr-vpn-off='sudo pkill -INT openfortivpn'" >> ~/.bashrc
```

Open a new terminal afterwards so the aliases load. If you created the old `maahr-vpn` alias earlier (it called `openfortivpn` directly), delete that old line from `~/.bashrc` first, or the old one may win:

```bash
sed -i "/^alias maahr-vpn=/d" ~/.bashrc
```

Do **not** put the VPN password into `~/maahr.conf`, into the script, or into this document. A file would hold it in plain text, and it is the same password you use to log in to the servers.

---

## 8. Cheat sheet

| Task | Command |
|---|---|
| Connect | `~/maahr-vpn.sh` (leave terminal open; asks sudo password, then VPN password) |
| Connected? | `ip -br addr show ppp0` (shows `192.168.70.x` when up) |
| Log in to GPU node | `ssh almauser1@10.172.0.233` |
| Disconnect | Ctrl+C in the VPN terminal |
| Force-stop | `sudo pkill -INT openfortivpn` |
| Remove completely | `sudo apt remove --purge openfortivpn ppp`, then `rm ~/maahr.conf ~/maahr-vpn.sh ~/.maahr_last_cert` |
