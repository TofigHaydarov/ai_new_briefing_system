import json
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field

class UserProfile(BaseModel):
    user: str
    preferred_topics: List[str] = Field(default_factory=list)
    excluded_sources: List[str] = Field(default_factory=list)
    max_items_per_topic: int
    @classmethod
    def from_dict(cls,data:dict):
        return cls(data.get("user"),data.get("preferred_topics",[]),data.get("excluded_sources",[]),data.get("max_items_per_topic",3))
        

    
class JSONUserRepo():

    def __init__(self,file_path:Path):
        self.file_path = file_path
        if not self.file_path.parent.exists():
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump({}, f)

    async def get_profile(self,name:str):
        with open(self.file_path,'r') as f:
            data = json.load(f)
        user_data = data.get(name,{})
        return UserProfile.from_dict(user_data)
    
    async def save_profile(self,profile:UserProfile):
        with open(self.file_path,'r') as f:
            data = json.load(f)
        if "user" in data and isinstance(data["user"], str):
            old_user = data["user"]
            data = {old_user: data}
        data[profile.user] = profile.model_dump(by_alias=True)
        with open(self.file_path, "w") as f:
            json.dump(data, f,indent = 4)
"""
userProf = UserProfile(user = "haji" ,preferred_topics= ["Football"],excluded_sources= ["Jews"] ,max_items_per_topic=5)

userRepo = JSONUserRepo(Path("C:/Users/Haji/Desktop/SWE Project/ai_new_briefing_system/data/user_profile.json"))
userRepo.save_profile(userProf)
"""