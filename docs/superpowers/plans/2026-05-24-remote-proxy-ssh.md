# SSH Remote Proxy Forwarding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route network traffic from the remote server (`121.48.164.50`, user `zxsun`) through the local Clash proxy via SSH Remote Port Forwarding.

**Architecture:** 
1. Map remote port `7890` to local port `7890` inside the Windows SSH config under host `EMI_zxsun` using `RemoteForward`.
2. Add proxy toggling helpers (`proxy_on`, `proxy_off`) on the remote server's `~/.bashrc` to control the proxy variables dynamically.

**Tech Stack:** SSH config, Bash shell scripting.

---

### Task 1: Update Local SSH Configuration

**Files:**
- Modify: `C:\Users\admin\.ssh\config`

- [ ] **Step 1: Check existing host config in C:\Users\admin\.ssh\config**
Verify the structure of `EMI_zxsun` configuration.

- [ ] **Step 2: Append RemoteForward option to EMI_zxsun**
Modify `C:\Users\admin\.ssh\config` to add `RemoteForward 7890 127.0.0.1:7890` at the end of the `EMI_zxsun` section.

Target replacement content for lines 1-7:
```ssh
Host EMI_zxsun
  HostName 121.48.164.50
  User zxsun
  Port 22
  IdentityFile ~/.ssh/id_ed25519
  PreferredAuthentications publickey,password
  IdentitiesOnly yes
  RemoteForward 7890 127.0.0.1:7890
```

- [ ] **Step 3: Verify local config has been written**
View `C:\Users\admin\.ssh\config` to verify the entry is correct.

---

### Task 2: Configure Remote Shell Helper Functions

**Files:**
- Modify: `~/.bashrc` on the remote server (via SSH connection)

- [ ] **Step 1: Append helper functions to remote ~/.bashrc**
Run a command to SSH into `EMI_zxsun` and append the proxy helper functions to the end of the `~/.bashrc` file.
Command:
```powershell
ssh -o StrictHostKeyChecking=accept-new EMI_zxsun "cat << 'EOF' >> ~/.bashrc

# Proxy helper functions
function proxy_on() {
    export http_proxy=\"http://127.0.0.1:7890\"
    export https_proxy=\"http://127.0.0.1:7890\"
    export all_proxy=\"socks5://127.0.0.1:7890\"
    echo \"Proxy environment variables set (port 7890).\"
}

function proxy_off() {
    unset http_proxy
    unset https_proxy
    unset all_proxy
    echo \"Proxy environment variables cleared.\"
}
EOF"
```

- [ ] **Step 2: Verify helper functions were appended successfully**
Run:
```powershell
ssh EMI_zxsun "tail -n 15 ~/.bashrc"
```
Expected: The printed output should show the `proxy_on` and `proxy_off` functions matching the code above.

---

### Task 3: Verification of Reverse Tunnel and Proxy Functionality

**Files:**
- None (Verification only)

- [ ] **Step 1: Test connection and port listener on remote server**
Run:
```powershell
ssh EMI_zxsun "ss -ant | grep 7890"
```
Expected: Output showing port `7890` in `LISTEN` state on the loopback address `127.0.0.1`.

- [ ] **Step 2: Test proxy functionality with Google query**
Run:
```powershell
ssh EMI_zxsun "source ~/.bashrc && proxy_on && curl -I -s --connect-timeout 5 https://www.google.com"
```
Expected: HTTP status code `200 OK` (or redirection/response from Google), indicating successful routing through the local Clash proxy.

- [ ] **Step 3: Test disabling the proxy**
Run:
```powershell
ssh EMI_zxsun "source ~/.bashrc && proxy_off"
```
Expected: Output "Proxy environment variables cleared."
