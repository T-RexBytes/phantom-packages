# 07: Demo and Assets

## 1. Hero demo GIF (required): "the hook stops my agent"

**Goal:** in 15–25 seconds, show an agent trying to install a ghost and getting blocked, then recovering.

**Setup**
1. Make a new scratch folder, `C:\Users\Soumya\Desktop\DEV\ghost-demo`, with an empty `main.py`.
2. Add `.claude/settings.json` with the hook from `templates/settings.hook.json`, pointing at your repo's hook script.
3. Pick **one real ghost from your data** that came back repeatedly *and* is still unregistered. Show it only in this local demo; blur or anonymise it if you publish.
   - **Safer:** pick a ghost that is now registered and harmless-looking, or a ghost already named in prior public research.
   - Check with Claude Code that showing the name publicly doesn't break your own ethics rule (02 §6). **If in doubt, blur the name.**
4. Find the prompt that summoned it in `raw_generations.jsonl`, and use a similar task prompt for the agent.

**Recording** (ScreenToGif on Windows, or OBS):
1. Terminal at about 110 columns, large font, dark theme.
2. Type the task to Claude Code, for example: *"Write a script that {{task that summoned the ghost}} and install what it needs."*
3. Capture: the agent proposes `pip install <ghost>` → the hook blocks it with the phantom message → the agent replies and switches to a real package.
4. Trim to under 25 seconds, under 8 MB, 960px wide, 10–12 fps.

**Fallback if the agent doesn't summon the ghost live:** record `phantom scan-cmd "pip install <ghost> requests"` instead, then a Claude Code session where you ask it to run that exact command. The block is still real.

## 2. Secondary GIFs or screenshots

| # | Asset | Where it's used |
|---|---|---|
| 2 | `phantom check requirements.txt` with GHOST / SUSPECT / OK rows (anonymised) | Demo |
| 3 | `phantom check --offline` with the Wi-Fi visibly off (show the system tray) | Demo + Why Open |
| 4 | `phantom why <registered ghost>` card | How I Built It |
| 5 | GitHub Action failing on a PR that adds a ghost (red ❌ check) | How I Built It |
| 6 | Calibration plot PNG | What I Measured |
| 7 | Rates by model PNG | What I Measured |
| 8 | Site hero + graveyard screenshot | Demo |
| 9 | Friend's reaction screenshot (crop out personal info) | Spirit of It |

## 3. Cover image (1000×420)

**Concept:** a terminal window in the dark. A `pip install ████████` line glows marigold-orange and a faint ghost shape rises out of the text. Bottom-left: "Phantom Packages". Bottom-right: the small HF26 mark. Hacktoberfest brand rules say don't modify the logos, so use them only as-is or leave them off.

**Palette**
- Ink `#231F20` background
- Ochre `#F5B726` and Orange `#E53927` accents (official HF26 accents)
- Paper `#F2F2EB` text

**Fonts:** Barlow Semi Condensed for the title, Martian Mono for the terminal text. Both are HF26 brand fonts and free on Google Fonts.

**Ways to make it:**
- Figma or Canva by hand.
- An HTML page rendered to PNG.
- An image model. Example prompt: *"Minimal dark illustration, a terminal window at night, one command line glowing marigold orange, a translucent ghost made of monospace characters rising from the text, flat design, high contrast, no text except the command line"*. Add the title text yourself afterwards.

## 4. Optional 60–90 second video script

| Time | Visual | Voice-over / caption |
|---|---|---|
| 0–8s | The hook GIF | "My coding agent tried to install a package that doesn't exist." |
| 8–20s | Site hero numbers | "I asked three open models for {{N}} scripts. {{rate}}% of their installs were ghosts." |
| 20–35s | Graveyard grid | "Some ghosts came back again and again. That's what makes them dangerous: anyone can register the name." |
| 35–55s | `phantom check` + `--offline` | "Phantom Packages spots them, even offline, with an open TabPFN model." |
| 55–75s | Action failing on a PR | "It runs as a Claude Code hook and as a GitHub Action." |
| 75–90s | Repo + post URL | "The model can dream a package. Only the registry can make it real." |

## 5. Site copy (paste into `web/index.html`)

- **H1:** *My local AI summoned {{N_GHOSTS}} Python packages that don't exist.*
- **Sub:** *Open-weight coding models invent package names. Attackers register them. This is the registry of ghosts I found, and the guard I built to stop them.*
- **Stat tiles:** "{{GHOST_RATE}}% of suggested installs were ghosts" · "{{REPEAT_RATE}}% of ghosts came back" · "{{N_REGISTERED}} ghosts already have a body"
- **Graveyard intro:** *Each stone is a package a model told me to install. Names of unregistered ghosts are hidden so nobody can claim them.*
- **Ethics note:** *Nothing in this study was installed. Lookups were metadata-only. Unregistered names that models repeat are withheld and were reported to the PyPI security team* (only if you actually reported them).
- **Footer:** *Built for the Hacktoberfest 2026 Open-Source AI Challenge on DEV. Code MIT. Rule: the model can dream a package; only the registry can make it real.*
