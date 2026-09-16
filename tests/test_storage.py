
from __future__ import annotations
 
from pathlib import Path
 
import pytest
 
from src.storage.repository import JSONUserRepo, UserProfile
 
pytestmark = pytest.mark.asyncio  # requires pytest-asyncio installed
 
 
@pytest.fixture
def repo(tmp_path: Path) -> JSONUserRepo:
    return JSONUserRepo(tmp_path / "user_profile.json")
 
 
@pytest.fixture
def sample_profile() -> UserProfile:
    return UserProfile(
        user="haji",
        preferred_topics=["Tech", "Science"],
        excluded_sources=["Spammy Times"],
        max_items_per_topic=5,
    )
 
#Happy path
 
 
async def test_save_then_get_roundtrip(repo: JSONUserRepo, sample_profile: UserProfile):
    """Saving a profile and reading it back should return equal data."""
    await repo.save_profile(sample_profile)
 
    fetched = await repo.get_profile("haji")
 
    assert fetched.user == "haji"
    assert fetched.preferred_topics == ["Tech", "Science"]
    assert fetched.excluded_sources == ["Spammy Times"]
    assert fetched.max_items_per_topic == 5
 
 
async def test_multiple_users_stored_independently(repo: JSONUserRepo):
    """Two different users saved to the same file shouldn't clobber each other."""
    haji = UserProfile(user="haji", preferred_topics=["Tech"], max_items_per_topic=5)
    tofig = UserProfile(user="tofig", preferred_topics=["Science"], max_items_per_topic=3)
 
    await repo.save_profile(haji)
    await repo.save_profile(tofig)
 
    fetched_haji = await repo.get_profile("haji")
    fetched_tofig = await repo.get_profile("tofig")
 
    assert fetched_haji.preferred_topics == ["Tech"]
    assert fetched_tofig.preferred_topics == ["Science"]
 
 
async def test_updating_existing_profile_overwrites_it(repo: JSONUserRepo, sample_profile: UserProfile):
    """Saving the same user twice should update, not duplicate, their entry."""
    await repo.save_profile(sample_profile)
 
    updated = sample_profile.model_copy(update={"max_items_per_topic": 10})
    await repo.save_profile(updated)
 
    fetched = await repo.get_profile("haji")
    assert fetched.max_items_per_topic == 10
 
#Error paths
 
async def test_get_profile_for_unknown_user_raises_clear_error(repo: JSONUserRepo):
    """
    A missing user should raise a clear, specific error — not an opaque
    KeyError from deep inside from_dict. Once the repository raises a
    dedicated exception (e.g. UserNotFoundError), swap the expectation
    below to `pytest.raises(UserNotFoundError)`.
    """
    with pytest.raises(Exception):  # tighten to a specific exception type once one exists
        await repo.get_profile("does-not-exist")
 
 
async def test_repo_creates_file_if_missing(tmp_path: Path):
    """Constructing a JSONUserRepo against a non-existent path/file
    should create the parent directory and an empty JSON file, not crash.
    """
    target = tmp_path / "nested" / "dir" / "profiles.json"
    JSONUserRepo(target)
 
    assert target.exists()
    assert target.read_text().strip() in ("{}", "")
 
 
async def test_save_profile_with_missing_optional_fields_uses_defaults(repo: JSONUserRepo):
    """preferred_topics/excluded_sources should default to an empty list
    when omitted, rather than raising a validation error.
    """
    minimal = UserProfile(user="minimal-user", max_items_per_topic=1)
    await repo.save_profile(minimal)
 
    fetched = await repo.get_profile("minimal-user")
    assert fetched.preferred_topics == []
    assert fetched.excluded_sources == []