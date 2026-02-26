"""
Synthetic demo data generator.

Creates realistic fake tweet/user data exhibiting bot and human patterns
so the tool can be evaluated without Twitter API credentials.

Bot patterns generated:
  - Cluster of accounts posting nearly identical text (coordination)
  - Accounts with random usernames (digits)
  - Accounts with very high tweet frequency
  - Accounts posting Russian disinformation narrative keywords
  - Accounts with very high retweet ratios
  - Accounts with no bio / default avatar
  - Very new accounts

Human patterns generated:
  - Varied posting times with sleep breaks
  - Natural username patterns
  - Reasonable follower/following ratios
  - Diverse, organic content
"""

import random
import string
from datetime import datetime, timedelta, timezone

# Seed for reproducibility in demos
random.seed(42)

DISINFO_TEXTS = [
    "NATO aggression against Russia must be condemned! #NoWar #StopNATOExpansion",
    "The Kiev regime is run by nazis. Zelensky nazi puppet exposed!",
    "Ukraine biolabs confirmed by leaked documents – western propaganda hiding the truth",
    "Stolen election! Deep state globalists control everything #FakeNewsMedia",
    "RT.com: NATO expansion caused this conflict – western mainstream media lies",
    "Denazification of Ukraine is necessary and justified",
    "George Soros and globalists funding the Kiev regime",
    "NATO is the real aggressor – Russia is defending itself from NATO provocation",
    "Ukraine coup in 2014 orchestrated by western powers – the truth mainstream media won't tell",
    "Ukraine bioweapons labs funded by the USA – documented evidence #TruthMatters",
]

ORGANIC_TEXTS = [
    "Just had a great coffee with a friend downtown. Life is good!",
    "Really enjoyed the game last night. What a comeback!",
    "Working on a new project – excited to share updates soon.",
    "Anyone else watching the new season? No spoilers please!",
    "The weather has been amazing lately. Perfect for a walk.",
    "Tried a new recipe tonight and it turned out pretty well!",
    "Long week but the weekend is almost here. Can't wait to relax.",
    "Finished reading a great book – highly recommend it.",
    "Looking forward to the holidays with family this year.",
    "The traffic today was terrible. Two hours to go 10 miles.",
]

BOT_HASHTAGS = [
    "#NoWar", "#StopNATOExpansion", "#NATOaggression",
    "#DeepState", "#FakeNewsMedia", "#TruthMatters",
    "#Denazification", "#GlobohomoCabal",
]

HUMAN_HASHTAGS = [
    "#coffee", "#weekendvibes", "#cooking", "#books",
    "#fitness", "#travel", "#photography", "#tech",
]

LOCATIONS = [
    "Moscow, Russia", "St. Petersburg", "Minsk", "Novosibirsk",
    "Yekaterinburg", "", "", "",  # many bots have no location
]

HUMAN_LOCATIONS = [
    "New York, NY", "London, UK", "Toronto", "Berlin",
    "Paris", "Sydney", "Chicago, IL", "Austin, TX",
]

HUMAN_BIOS = [
    "Software engineer. Coffee addict. Dog lover.",
    "Marketing professional | Travel enthusiast | Mom of 3",
    "Journalist covering politics and tech. Views my own.",
    "PhD student in environmental science. Hiker.",
    "Retired teacher. Now gardening and reading.",
]


def _random_bot_username() -> str:
    """Generate a bot-like username with digits."""
    prefixes = ["user", "acc", "news", "info", "real", "truth"]
    suffix = "".join(random.choices(string.digits, k=random.randint(4, 8)))
    return random.choice(prefixes) + suffix


def _random_human_username() -> str:
    """Generate a human-like username."""
    first = random.choice([
        "alice", "bob", "charlie", "diana", "evan",
        "fiona", "george", "helen", "ivan", "julia",
    ])
    last = random.choice([
        "smith", "johnson", "williams", "brown", "jones",
        "garcia", "miller", "davis", "wilson", "taylor",
    ])
    sep = random.choice(["", "_", "."])
    suffix = str(random.randint(1, 99)) if random.random() < 0.3 else ""
    return first + sep + last + suffix


def _make_timestamp(base: datetime, offset_minutes: float) -> str:
    return (base + timedelta(minutes=offset_minutes)).isoformat()


def generate_demo_data(n_accounts: int = 60) -> list[dict]:
    """
    Generate synthetic tweet records representing a mix of bots and humans.

    Returns a list of tweet dicts in the FileCollector normalised format.
    """
    records: list[dict] = []
    now = datetime.now(timezone.utc)

    n_bots = int(n_accounts * 0.45)
    n_suspected = int(n_accounts * 0.15)
    n_humans = n_accounts - n_bots - n_suspected

    # ----------------------------------------------------------------
    # BOT CLUSTER 1: Coordinated posting of near-identical disinfo text
    # ----------------------------------------------------------------
    cluster_size = min(n_bots // 2, 15)
    for i in range(cluster_size):
        uid = f"bot_cluster1_{i:03d}"
        age_days = random.randint(10, 60)
        created_at_account = (now - timedelta(days=age_days)).isoformat()
        followers = random.randint(0, 30)
        following = random.randint(200, 3000)

        author = {
            "id": uid,
            "username": _random_bot_username(),
            "name": "".join(random.choices(string.ascii_lowercase, k=8)),
            "created_at": created_at_account,
            "account_age_days": age_days,
            "description": "",
            "location": random.choice(LOCATIONS),
            "profile_image_url": None,
            "verified": False,
            "protected": False,
            "followers_count": followers,
            "following_count": following,
            "tweet_count": random.randint(500, 10000),
            "listed_count": 0,
        }

        # All post nearly identical text within a 3-minute window
        base_text = random.choice(DISINFO_TEXTS)
        base_time = now - timedelta(hours=random.randint(1, 48))
        for j in range(random.randint(5, 15)):
            tweet_time = _make_timestamp(base_time, j * 0.5)
            # Slight variation in the text to simulate copy-paste with minor edits
            text = base_text + (f" #{random.choice(string.ascii_uppercase)}" if j % 3 == 0 else "")
            records.append(_make_record(uid, author, text, tweet_time, is_retweet=(j % 2 == 0)))

    # ----------------------------------------------------------------
    # BOT CLUSTER 2: High-frequency amplifier bots (RT everything)
    # ----------------------------------------------------------------
    for i in range(n_bots - cluster_size):
        uid = f"bot_amplifier_{i:03d}"
        age_days = random.randint(5, 180)
        created_at_account = (now - timedelta(days=age_days)).isoformat()
        followers = random.randint(0, 20)
        following = random.randint(1000, 5000)

        author = {
            "id": uid,
            "username": _random_bot_username(),
            "name": "".join(random.choices(string.digits + string.ascii_lowercase, k=10)),
            "created_at": created_at_account,
            "account_age_days": age_days,
            "description": "",
            "location": "",
            "profile_image_url": None,
            "verified": False,
            "protected": False,
            "followers_count": followers,
            "following_count": following,
            "tweet_count": random.randint(2000, 50000),
            "listed_count": 0,
        }

        # Posts continuously at regular intervals (robotic scheduling)
        for j in range(random.randint(20, 40)):
            tweet_time = _make_timestamp(now - timedelta(days=1), j * 1.0)  # Every 1 minute exactly
            text = random.choice(DISINFO_TEXTS)
            records.append(_make_record(uid, author, text, tweet_time, is_retweet=True))

    # ----------------------------------------------------------------
    # SUSPECTED BOTS: Partial bot signals
    # ----------------------------------------------------------------
    for i in range(n_suspected):
        uid = f"suspected_{i:03d}"
        age_days = random.randint(60, 365)
        created_at_account = (now - timedelta(days=age_days)).isoformat()
        followers = random.randint(50, 500)
        following = random.randint(200, 2000)

        author = {
            "id": uid,
            "username": _random_bot_username() if random.random() < 0.5 else _random_human_username(),
            "name": "Concerned Citizen" if random.random() < 0.5 else "NewsWatcher2023",
            "created_at": created_at_account,
            "account_age_days": age_days,
            "description": "" if random.random() < 0.6 else "Following the news",
            "location": random.choice(LOCATIONS + HUMAN_LOCATIONS),
            "profile_image_url": "https://pbs.twimg.com/profile_images/xyz.jpg",
            "verified": False,
            "protected": False,
            "followers_count": followers,
            "following_count": following,
            "tweet_count": random.randint(300, 5000),
            "listed_count": random.randint(0, 5),
        }

        for j in range(random.randint(5, 20)):
            tweet_time = _make_timestamp(now - timedelta(days=random.randint(0, 7)),
                                         random.uniform(-200, 0))
            use_disinfo = random.random() < 0.4
            text = random.choice(DISINFO_TEXTS if use_disinfo else ORGANIC_TEXTS)
            records.append(_make_record(uid, author, text, tweet_time, is_retweet=(random.random() < 0.5)))

    # ----------------------------------------------------------------
    # HUMANS: Organic posting patterns
    # ----------------------------------------------------------------
    for i in range(n_humans):
        uid = f"human_{i:03d}"
        age_days = random.randint(365, 3650)
        created_at_account = (now - timedelta(days=age_days)).isoformat()
        followers = random.randint(50, 5000)
        following = random.randint(50, 2000)

        author = {
            "id": uid,
            "username": _random_human_username(),
            "name": f"{random.choice(['Alice','Bob','Carol','Dave','Emma'])} {random.choice(['Smith','Jones','Brown'])}",
            "created_at": created_at_account,
            "account_age_days": age_days,
            "description": random.choice(HUMAN_BIOS),
            "location": random.choice(HUMAN_LOCATIONS),
            "profile_image_url": "https://pbs.twimg.com/profile_images/realphoto.jpg",
            "verified": random.random() < 0.05,
            "protected": False,
            "followers_count": followers,
            "following_count": following,
            "tweet_count": random.randint(100, 10000),
            "listed_count": random.randint(0, 30),
        }

        # Natural posting: active during day hours, quiet at night
        for j in range(random.randint(3, 15)):
            # Bias towards waking hours (8am–11pm)
            hour = random.choices(range(24), weights=[
                1,1,1,1,1,1,1,2, 5,7,8,8,7,7,8,8,7,6,6,7,6,5,3,2
            ])[0]
            tweet_time = (now - timedelta(days=random.randint(0, 30))).replace(
                hour=hour, minute=random.randint(0, 59)
            ).isoformat()
            text = random.choice(ORGANIC_TEXTS)
            records.append(_make_record(uid, author, text, tweet_time, is_retweet=(random.random() < 0.15)))

    random.shuffle(records)
    return records


def _make_record(
    author_id: str,
    author: dict,
    text: str,
    created_at: str,
    is_retweet: bool = False,
) -> dict:
    tweet_id = f"tweet_{random.randint(10**17, 10**18)}"
    referenced = (
        [{"type": "retweeted", "id": f"{random.randint(10**17, 10**18)}"}]
        if is_retweet else []
    )
    return {
        "id": tweet_id,
        "text": ("RT @someuser: " + text) if is_retweet else text,
        "author_id": author_id,
        "created_at": created_at,
        "lang": "en",
        "retweet_count": random.randint(0, 100) if not is_retweet else 0,
        "reply_count": random.randint(0, 10) if not is_retweet else 0,
        "like_count": random.randint(0, 500) if not is_retweet else 0,
        "quote_count": random.randint(0, 5) if not is_retweet else 0,
        "referenced_tweets": referenced,
        "entities": {"hashtags": []},
        "source": "Twitter for Android" if random.random() < 0.5 else "Twitter Web App",
        "author": author,
    }
