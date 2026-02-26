"""Tests for the FileCollector."""

import json
import pytest
from pathlib import Path
from src.collectors.file_collector import FileCollector


@pytest.fixture
def tmp_json(tmp_path):
    data = [
        {
            "id": "1001",
            "text": "Test tweet content",
            "created_at": "2024-06-01T10:00:00Z",
            "lang": "en",
            "retweet_count": 5,
            "like_count": 12,
            "author": {
                "id": "u001",
                "username": "testuser",
                "name": "Test User",
                "created_at": "2020-01-01T00:00:00Z",
                "description": "A test user",
                "location": "Testville",
                "profile_image_url": "https://example.com/pic.jpg",
                "verified": False,
                "protected": False,
                "followers_count": 100,
                "following_count": 50,
                "tweet_count": 500,
                "listed_count": 2,
            },
        }
    ]
    p = tmp_path / "test.json"
    p.write_text(json.dumps(data))
    return str(p)


@pytest.fixture
def tmp_csv(tmp_path):
    content = (
        "id,text,author_id,author_username,author_name,author_created_at,"
        "author_followers_count,author_following_count,author_tweet_count,"
        "author_description,author_location,author_profile_image_url,"
        "author_verified,author_protected,created_at,lang\n"
        "2001,CSV tweet,u002,csvuser,CSV User,2021-06-01T00:00:00Z,"
        "200,100,300,CSV bio,CSV City,https://pic.jpg,false,false,"
        "2024-06-01T10:00:00Z,en\n"
    )
    p = tmp_path / "test.csv"
    p.write_text(content)
    return str(p)


class TestFileCollector:
    def test_load_json(self, tmp_json):
        collector = FileCollector()
        records = collector.load(tmp_json)
        assert len(records) == 1
        assert records[0]["id"] == "1001"
        assert records[0]["text"] == "Test tweet content"
        assert records[0]["author"]["username"] == "testuser"

    def test_load_csv(self, tmp_csv):
        collector = FileCollector()
        records = collector.load(tmp_csv)
        assert len(records) == 1
        assert records[0]["id"] == "2001"
        assert records[0]["author"]["username"] == "csvuser"

    def test_missing_file_raises(self):
        collector = FileCollector()
        with pytest.raises(FileNotFoundError):
            collector.load("/nonexistent/path/data.json")

    def test_unsupported_format_raises(self, tmp_path):
        p = tmp_path / "data.xml"
        p.write_text("<data/>")
        collector = FileCollector()
        with pytest.raises(ValueError):
            collector.load(str(p))

    def test_account_age_computed(self, tmp_json):
        collector = FileCollector()
        records = collector.load(tmp_json)
        age = records[0]["author"]["account_age_days"]
        assert age is not None
        assert age > 0

    def test_numeric_fields_normalised(self, tmp_json):
        collector = FileCollector()
        records = collector.load(tmp_json)
        assert isinstance(records[0]["retweet_count"], int)
        assert isinstance(records[0]["like_count"], int)
        assert isinstance(records[0]["author"]["followers_count"], int)
