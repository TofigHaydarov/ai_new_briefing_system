import asyncio
from pathlib import Path
from typing import List, Optional
import typer
from storage.repository import JSONUserRepo, UserProfile
from config import settings
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

if __name__ == "__main__":
    app()