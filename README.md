# RuzzianProp

**Russian disinformation bot network analyser**

A tool for detecting, scoring, and visualising networks of Russian-linked disinformation bots on social media platforms such as X (Twitter).

---

## What it does

| Capability | Description |
|---|---|
| **Bot scoring** | Scores every account 0–100% certainty of being a bot, using 10 behavioural signals |
| **Disinformation detection** | Matches tweet content against known Russian narrative keyword databases |
| **Network graphs** | Builds directed interaction graphs (mentions, retweets, shared hashtags) |
| **Community detection** | Uses the Louvain algorithm to find tightly-knit amplification clusters |
| **Visualisations** | Static PNG network maps + interactive zoomable HTML graphs |
| **Reports** | Exports JSON report with full per-account breakdown |

---

## Bot Detection Signals

Each account is scored on 10 weighted signals:

| Signal | Weight | Description |
|---|---|---|
| Account age | 10% | Accounts < 30 days old score high |
| Username pattern | 8% | Random-looking usernames with digit strings |
| Profile completeness | 8% | Missing bio, no location, default avatar |
| Tweet frequency | 10% | Superhuman posting rates (>100 tweets/day) |
| Retweet ratio | 10% | >90% retweets = amplifier bot |
| Follower ratio | 10% | Following >> followers |
| Content similarity | 15% | Copy-paste campaigns, repetitive posting |
| Posting time pattern | 9% | 24/7 posting without sleep breaks, robotic spacing |
| Coordinated behaviour | 12% | Multiple accounts posting same content within 10 minutes |
| Disinformation keywords | 8% | Known IRA narrative terms in tweet content |

**Classification thresholds:**
- `BOT` ≥ 60%
- `SUSPECTED` ≥ 40%
- `HUMAN` < 40%

---

## Disinformation Narrative Categories

Keyword databases cover:
- **NATO narratives** – "NATO aggression", "NATO provocation", etc.
- **Ukraine narratives** – "Zelensky nazi", "Kiev regime", "denazification", "Ukraine biolabs"
- **Election narratives** – "stolen election", "deep state", "rigged election"
- **Western narratives** – "western propaganda", "George Soros", "globalist"
- **State media domains** – rt.com, sputniknews.com, ria.ru, tass.ru, etc.

---

## Installation

```bash
git clone <repo>
cd ruzzianprop
pip install -r requirements.txt
pip install -e .
```

---

## Usage

### Analyse a local JSON/CSV file (no API key needed)

```bash
ruzzianprop analyse --input data/sample_tweets.json --output ./output
```

### Live Twitter/X API search

Set your credentials first:

```bash
export TWITTER_BEARER_TOKEN=your_token_here
```

Then:

```bash
ruzzianprop analyse --query "NATO aggression" --max-results 500 --output ./output
```

### Run the demo (synthetic data, no API key)

```bash
ruzzianprop demo --accounts 80 --output ./output
```

### Options

```
ruzzianprop analyse --help

  --input, -i       Path to JSON or CSV data file
  --query, -q       Twitter search query (requires API credentials)
  --max-results, -n Number of tweets to fetch (default: 200)
  --output, -o      Output directory (default: ./output)
  --config, -c      YAML config file (default: config/settings.yaml)
  --no-graph        Skip network graph generation
  --no-interactive  Skip interactive HTML graph
  --verbose, -v     Enable verbose logging
```

---

## Output Files

| File | Description |
|---|---|
| `output/report.json` | Full JSON report: summary, top accounts, per-account signals |
| `output/network.png` | Static network map (classification + community views) |
| `output/network.html` | Interactive zoomable network (open in browser) |
| `output/network_heatmap.png` | Bot signal heatmap for top 30 accounts |

### Interactive HTML Network

Open `output/network.html` in any browser:
- **Zoom / Pan** to explore the graph
- **Hover** over nodes to see full account details
- **Red nodes** = confirmed bots
- **Orange nodes** = suspected bots
- **Green nodes** = likely human
- **Gold border** = high disinformation score
- **Node size** = PageRank influence within the network
- **Blue edges** = mentions/retweets
- **Grey edges** = shared hashtags

---

## Data Input Format

### JSON

A list of tweet objects with a nested `author` field:

```json
[
  {
    "id": "1234567890",
    "text": "Tweet content here",
    "created_at": "2024-01-01T12:00:00Z",
    "lang": "en",
    "retweet_count": 0,
    "like_count": 5,
    "referenced_tweets": [],
    "entities": {},
    "author": {
      "id": "user123",
      "username": "someuser",
      "name": "Some User",
      "created_at": "2020-06-01T00:00:00Z",
      "description": "Bio here",
      "location": "City, Country",
      "profile_image_url": "https://...",
      "verified": false,
      "protected": false,
      "followers_count": 500,
      "following_count": 300,
      "tweet_count": 1200,
      "listed_count": 5
    }
  }
]
```

### CSV

Flat CSV with columns prefixed `author_*` for user fields:

```
id,text,author_id,author_username,author_name,author_created_at,
author_followers_count,author_following_count,author_tweet_count,
author_description,author_location,author_profile_image_url,
author_verified,author_protected,created_at,lang
```

---

## Configuration

Edit `config/settings.yaml` to:
- Adjust signal weights
- Add/remove disinformation keywords
- Configure visualisation colours and output format
- Add Twitter API credentials

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Architecture

```
src/
├── collectors/
│   ├── twitter_collector.py   # Twitter API v2 via Tweepy
│   └── file_collector.py      # JSON / CSV loader
├── detectors/
│   ├── bot_detector.py        # 10-signal bot scoring engine
│   └── disinformation_detector.py  # Narrative keyword matching
├── network/
│   ├── graph_builder.py       # NetworkX graph construction
│   └── community_detector.py  # Louvain community detection
├── visualization/
│   └── network_visualizer.py  # Matplotlib + PyVis rendering
├── demo.py                    # Synthetic data generator
└── cli.py                     # Click CLI entry point
```

---

## Data Sources & References

Keyword databases are informed by:
- [EU DisinfoLab](https://www.disinfo.eu/) reports
- [Stanford Internet Observatory](https://io.stanford.edu/) findings
- [DFRLab](https://dfrlab.org/) (Digital Forensic Research Lab)
- [EUvsDisinfo](https://euvsdisinfo.eu/) database
- Academic research on the Internet Research Agency (IRA) tactics

---

## Legal & Ethical Note

This tool is designed for **defensive research**, **journalism**, **academic study**,
and **counter-disinformation** work. Use it responsibly and in compliance with the
terms of service of any platform from which data is collected.
