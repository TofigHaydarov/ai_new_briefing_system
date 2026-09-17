import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import asyncio
from typing import List, Optional
import typer
from storage.repository import JSONUserRepo, UserProfile
from config import settings
from src.services.briefing_service import generate_user_digest
from ai.llm import summarize_and_label
from ai.schemas import Article, LabeledSummary,DigestItem,Digest
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = typer.Typer(help="AI News Briefing System CLI")
repo = JSONUserRepo(settings.JSON_STORAGE_PATH)

@app.command(name="get-profile")
def show_profile(username: str = typer.Argument(..., help="Username to retrieve")):
    profile = asyncio.run(repo.get_profile(username))
    if not profile:
        print(f"ERROR: Profile for user '{username}' not found.")
        raise typer.Exit(code=1)
    print(str(profile))
@app.command(name="create-profile")
def add_user(
    username: str = typer.Option(..., "--user", "-u", help="Username / User ID"),
    topics: List[str] = typer.Option(["General"], "--topics", "-t", help="Preferred topics"),
    excluded: List[str] = typer.Option([], "--excluded", "-e", help="Excluded sources/domains"),
    max_items: int = typer.Option(3, "--max-items", "-m", help="Max articles per topic"),
):
    profile = UserProfile(user=username,preferred_topics=topics,excluded_sources=excluded,max_items_per_topic=max_items)
    if not profile:
            print("Invalid parameters.")
            raise typer.Exit(code=1)
    asyncio.run(repo.save_profile(profile))
    print("Profile Saved Successfully!")
@app.command(name="list-users")
def list_users():
    profiles = asyncio.run(repo.get_all_profiles())

    if not profiles:
        typer.echo("No user profiles found in storage.")
        return

    typer.echo(f"Found {len(profiles)} user profile(s):")
    for profile in profiles:
        typer.echo(f" - {profile.user} (Topics: {', '.join(profile.preferred_topics)})")


@app.command(name="delete-profile")
def delete_profile(
    username: str = typer.Option(..., "--user", "-u", help="Username to delete")
):
    deleted = asyncio.run(repo.delete_profile(username))

    if not deleted:
        typer.echo(f"ERROR: Could not delete profile. User '{username}' not found.", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Successfully deleted profile for '{username}'.")



@app.command(name="generate-briefing")
def generate_briefing(
    username: Optional[str] = typer.Option(
        None, "--user", "-u", help="Target username for briefing generation"
    ),
    run_all: bool = typer.Option(
        False, "--all", "-a", help="Generate briefings for all registered users"
    ),
):
    if not username and not run_all:
        typer.echo("ERROR: Please specify a user using --user <username> or pass --all.", err=True)
        raise typer.Exit(code=1)

    # the following is to be replaced with web-scraper output
    sample_articles = [
        Article(
            title="New AI Model Released",
            url="https://techcrunch.com/example",
            source="TechCrunch",
            content="A ground-breaking AI model was announced today capable of...",
        )
    ]

    profiles = []
    if run_all:
        profiles = asyncio.run(repo.get_all_profiles())
    else:
        user_profile = asyncio.run(repo.get_profile(username))
        if user_profile:
            profiles.append(user_profile)
        else:
            typer.echo(f"ERROR: User '{username}' not found.", err=True)
            raise typer.Exit(code=1)

    
    settings.DIGEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for profile in profiles:
        typer.echo(f"Processing briefing for '{profile.user}'...")
        
        digest = asyncio.run(generate_user_digest(profile, sample_articles))

        output_path = settings.DIGEST_OUTPUT_DIR / f"{profile.user}_latest.json"
        output_path.write_text(digest.model_dump_json(indent=2), encoding="utf-8")
        
        typer.echo(f"  [+] Saved {len(digest.items)} digest items to {output_path}")


@app.command(name="get-briefing")
def get_briefing(
    username: str = typer.Option(..., "--user", "-u", help="Username to view latest briefing for")
):
    """View the latest generated briefing digest file for a given user."""
    # Assumes digests are saved under settings.DIGESTS_DIR / {username}_latest.json
    digest_file = settings.DIGESTS_DIR / f"{username}_latest.json"

    if not digest_file.exists():
        typer.echo(f"ERROR: No briefing digest found for '{username}' at path '{digest_file}'.", err=True)
        raise typer.Exit(code=1)

    try:
        content = digest_file.read_text(encoding="utf-8")
        typer.echo(content)
    except Exception as err:
        typer.echo(f"ERROR: Failed to read briefing file: {err}", err=True)
        raise typer.Exit(code=1)

    
if __name__ == "__main__":
    app()