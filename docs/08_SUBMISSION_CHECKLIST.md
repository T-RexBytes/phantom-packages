# 08: Submission Checklist

Tick every box before you hit Publish. Target publish time: **Oct 30, 8 PM PDT**.

## A. Rules and format
- [ ] First line is the italic "This is a submission for … Week 4: Ghost in the Machine" link
- [ ] Tags: `devchallenge`, `hf26challenge` (+ `python`, `security`)
- [ ] All template headings, in order: What I Built · Demo · Code · How I Built It · Why Does Open Innovation Matter? · My Agent Session · The Spirit of It · Prize Categories
- [ ] Project is new and built during the window (Oct 26–30), shown by the repo's commit dates
- [ ] Open-weight / open-source AI is **core** (3 local models + TabPFN open weights + offline mode)
- [ ] If anything needs login, testing instructions are included (the site doesn't, so N/A)
- [ ] Team? If yes, teammates' DEV usernames are credited in the body
- [ ] Only one submission for this challenge
- [ ] AI disclosure: DEV guidelines ask you to disclose AI assistance. Say Claude Code helped build it and that you wrote or edited the post.

## B. Content quality (from the winners' traits)
- [ ] Hook is a concrete scene (the real blocked install), not a feature list
- [ ] Headline result in **bold** within the first 200 words
- [ ] The rule appears as an H3 aphorism
- [ ] Eval table with ≥4 baselines, CIs, calibration plot, held-out splits
- [ ] Leave-one-model-out result shown and explained
- [ ] "What Went Wrong" has ≥3 real failures **with numbers**
- [ ] "Why open" has concrete reasons + an honest concession
- [ ] Prior slopsquatting work credited, with what's new stated in one sentence
- [ ] Real-person reaction or real moment (no invented quotes)
- [ ] 2,200–3,000 words; short paragraphs; one "so what" line under each table
- [ ] Every number matches `results/summary.json` (run the judge prompt in 04)

## C. Demo and code
- [ ] Hero GIF shows the hook blocking a real install (<8 MB)
- [ ] Live site link works on a phone and in an incognito window
- [ ] Repo is public, MIT LICENSE, README covers install / CLI / hook / Action / offline / ethics
- [ ] `uv run tasks.py all --fast` works from a fresh clone (test on a clean folder)
- [ ] CI green badge
- [ ] Mermaid diagram renders in DEV preview

## D. Partner categories (claim only what's real)
- [ ] TabPFN paragraph: version, local run, baselines, threshold, uncertainty band, ships in the guard
- [ ] Ultramarine paragraph: what it did + **exact cost** with token counts (if used)
- [ ] Gemma paragraph (if used)
- [ ] .tech domain live and linked (if claimed)
- [ ] No claims for categories you didn't truly use

## E. Ethics and safety
- [ ] Nothing model-suggested was ever installed (say so in the post)
- [ ] No unregistered, repeatedly hallucinated names anywhere public: post, site, repo, GIFs, screenshots
- [ ] Private lists are gitignored. Check `git ls-files private` returns nothing.
- [ ] Disclosure to security@pypi.org only mentioned if actually sent

## F. After publishing
- [ ] Open the post logged out: images, embeds, GIF and Mermaid all render
- [ ] DevRelay session embed renders
- [ ] Share link in the MLH Discord / socials; reply to every comment in the first 24h
- [ ] Don't edit the core content after the deadline

---

## Self-score (be harsh; aim for 8+ everywhere before publishing)

| Criterion (weight) | 1–10 | Evidence / what would raise it |
|---|---|---|
| Writing quality (highest) | | |
| Relevance to theme | | |
| Creativity | | |
| Technical execution | | |
| Partner tech | | |
