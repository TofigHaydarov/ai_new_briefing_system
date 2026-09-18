import json
from datetime import datetime, timezone, date

import pytest
from typer.testing import CliRunner

import src.cli as cli_module
from src.cli import app
from src.storage.repository import UserProfile
from ai.schemas import Article, Digest, DigestItem, LabeledSummary, Topic, Sentiment

runner = CliRunner()


class FakeCLIRepo:
    """In-memory stand-in matching the async interface cli.py expects from
    JSONUserRepo: get_profile, save_profile, get_all_profiles, delete_profile."""

    def __init__(self, profiles=None):
        self._profiles = profiles or {}

    async def get_profile(self, username):
        return self._profiles.get(username)

    async def save_profile(self, profile):
        self._profiles[profile.user] = profile

    async def get_all_profiles(self):
        return list(self._profiles.values())

    async def delete_profile(self, username):
        if username in self._profiles:
            del self._profiles[username]
            return True
        return False


def make_profile(user="testuser", topics=None, excluded=None, max_items=3):
    return UserProfile(
        user=user,
        preferred_topics=topics or ["General"],
        excluded_sources=excluded or [],
        max_items_per_topic=max_items,
    )


def make_digest(user="testuser", n_items=1):
    items = [
        DigestItem(
            article=Article(
                title=f"Article {i}",
                url=f"https://example.com/{i}",
                source="TechCrunch",
                content=f"Content {i}.",
            ),
            labeled=LabeledSummary(
                summary=f"Summary {i}.",
                topic=Topic.TECH,
                sentiment=Sentiment.NEUTRAL,
            ),
        )
        for i in range(n_items)
    ]
    return Digest(user=user, generated_at=datetime.now(timezone.utc), items=items)


@pytest.fixture
def fake_repo(monkeypatch):
    repo = FakeCLIRepo()
    monkeypatch.setattr(cli_module, "repo", repo)
    return repo


@pytest.fixture
def digest_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_module.settings, "DIGEST_OUTPUT_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def stub_digest_generation(monkeypatch):
    """Bypasses the real AI pipeline; generate-briefing/run-daily tests only
    need to exercise the CLI layer (profile lookup, file writing, output)."""

    async def fake_generate(profile, articles):
        return make_digest(user=profile.user)

    monkeypatch.setattr(cli_module, "generate_user_digest", fake_generate)


class TestGetProfile:

    def test_found(self, fake_repo):
        profile = make_profile(user="alice")
        fake_repo._profiles["alice"] = profile

        result = runner.invoke(app, ["get-profile", "alice"])

        assert result.exit_code == 0
        assert "alice" in result.stdout

    def test_not_found(self, fake_repo):
        result = runner.invoke(app, ["get-profile", "missing"])

        assert result.exit_code == 1
        assert "not found" in result.stdout.lower() or "not found" in (result.stderr or "").lower()


class TestCreateProfile:

    def test_success(self, fake_repo):
        result = runner.invoke(
            app,
            ["create-profile", "--user", "bob", "--topics", "Tech", "--max-items", "5"],
        )

        assert result.exit_code == 0
        assert "bob" in fake_repo._profiles
        assert fake_repo._profiles["bob"].max_items_per_topic == 5

    def test_default_topics_when_not_specified(self, fake_repo):
        result = runner.invoke(app, ["create-profile", "--user", "carol"])

        assert result.exit_code == 0
        assert fake_repo._profiles["carol"].preferred_topics == ["General"]

    def test_repo_failure_reported_as_error(self, fake_repo, monkeypatch):
        async def failing_save(profile):
            raise RuntimeError("disk full")

        monkeypatch.setattr(fake_repo, "save_profile", failing_save)

        result = runner.invoke(app, ["create-profile", "--user", "dave"])

        assert result.exit_code == 1


class TestListUsers:

    def test_empty(self, fake_repo):
        result = runner.invoke(app, ["list-users"])

        assert result.exit_code == 0
        assert "no user profiles" in result.stdout.lower()

    def test_with_data(self, fake_repo):
        fake_repo._profiles["alice"] = make_profile(user="alice", topics=["Tech"])
        fake_repo._profiles["bob"] = make_profile(user="bob", topics=["Sports"])

        result = runner.invoke(app, ["list-users"])

        assert result.exit_code == 0
        assert "alice" in result.stdout
        assert "bob" in result.stdout


class TestDeleteProfile:

    def test_success(self, fake_repo):
        fake_repo._profiles["alice"] = make_profile(user="alice")

        result = runner.invoke(app, ["delete-profile", "--user", "alice"])

        assert result.exit_code == 0
        assert "alice" not in fake_repo._profiles

    def test_not_found(self, fake_repo):
        result = runner.invoke(app, ["delete-profile", "--user", "missing"])

        assert result.exit_code == 1


class TestGenerateBriefing:

    def test_requires_user_or_all(self, fake_repo):
        result = runner.invoke(app, ["generate-briefing"])

        assert result.exit_code == 1

    def test_single_user(self, fake_repo, digest_dir, stub_digest_generation):
        fake_repo._profiles["alice"] = make_profile(user="alice")

        result = runner.invoke(app, ["generate-briefing", "--user", "alice"])

        assert result.exit_code == 0
        today = date.today().strftime("%Y-%m-%d")
        output_file = digest_dir / f"{today}-alice.md"
        assert output_file.exists()
        assert "Article 0" in output_file.read_text(encoding="utf-8")

    def test_unknown_user(self, fake_repo, digest_dir, stub_digest_generation):
        result = runner.invoke(app, ["generate-briefing", "--user", "ghost"])

        assert result.exit_code == 1

    def test_all_users(self, fake_repo, digest_dir, stub_digest_generation):
        fake_repo._profiles["alice"] = make_profile(user="alice")
        fake_repo._profiles["bob"] = make_profile(user="bob")

        result = runner.invoke(app, ["generate-briefing", "--all"])

        assert result.exit_code == 0
        today = date.today().strftime("%Y-%m-%d")
        assert (digest_dir / f"{today}-alice.md").exists()
        assert (digest_dir / f"{today}-bob.md").exists()

    def test_run_daily_alias_behaves_the_same(self, fake_repo, digest_dir, stub_digest_generation):
        fake_repo._profiles["alice"] = make_profile(user="alice")

        result = runner.invoke(app, ["run-daily", "--user", "alice"])

        assert result.exit_code == 0
        today = date.today().strftime("%Y-%m-%d")
        assert (digest_dir / f"{today}-alice.md").exists()


class TestGetBriefing:

    def test_found_with_explicit_date(self, fake_repo, digest_dir):
        digest_file = digest_dir / "2026-01-01-alice.md"
        digest_file.write_text("# Daily AI News Digest for alice", encoding="utf-8")

        result = runner.invoke(app, ["get-briefing", "--user", "alice", "--date", "2026-01-01"])

        assert result.exit_code == 0
        assert "alice" in result.stdout

    def test_missing_file(self, fake_repo, digest_dir):
        result = runner.invoke(app, ["get-briefing", "--user", "ghost", "--date", "2026-01-01"])

        assert result.exit_code == 1

    def test_default_date_finds_todays_file(self, fake_repo, digest_dir):
        # Exercises the `date_str or ...` fallback in get-briefing. If the
        # fallback resolves to something other than today's actual date
        # (e.g. None, due to the asyncio.sleep(0) expression), this fails
        # because the file below won't be found.
        today = date.today().strftime("%Y-%m-%d")
        digest_file = digest_dir / f"{today}-alice.md"
        digest_file.write_text("# Daily AI News Digest for alice", encoding="utf-8")

        result = runner.invoke(app, ["get-briefing", "--user", "alice"])

        assert result.exit_code == 0
        assert "alice" in result.stdout