import asyncio
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, Header, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from typing import Any, Optional
import httpx
import os
from dotenv import load_dotenv

from app.schemas.task_schema import Task, TaskCreate
from app.services.createAITaskPlan import create_subtasks_with_llm
from app.services.createTaskDebrief import create_task_debrief
from app.services.getLearningResources import get_learning_resources

load_dotenv()
router = APIRouter()

BACKEND_URL = os.getenv("TASKMASTER_BACKEND_URL")

if not BACKEND_URL:
    raise RuntimeError("TASKMASTER_BACKEND_URL environment variable is not set")


@router.post("/plan-tasks")
async def plan_tasks(
    new_task: TaskCreate,
    authorization: Optional[str] = Header(default=None),
):
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")

    auth_headers = {"Authorization": authorization}

    # ── 1. Fetch active tasks (best-effort — workload context only) ────────
    active_tasks: list[Task] = []
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(
            connect=10.0,
            read=15.0,
            write=10.0,
            pool=10.0,
        )) as client:
            active_response = await client.get(f"{BACKEND_URL}/get-tasks", headers=auth_headers)
            if active_response.status_code == 200:
                try:
                    active_tasks = [Task(**task) for task in active_response.json()]
                except Exception as parse_err:
                    print(f"[plan-tasks] Could not parse active tasks: {parse_err}")
            else:
                print(f"[plan-tasks] Active tasks fetch returned {active_response.status_code}; proceeding without workload context")
    except httpx.RequestError as e:
        print(f"[plan-tasks] Could not reach backend for active tasks: {e}; proceeding without workload context")

    # ── 2. Create parent task ──────────────────────────────────────────────
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(
            connect=10.0,
            read=300.0,
            write=10.0,
            pool=10.0,
        )) as client:
            create_response = await client.post(
                f"{BACKEND_URL}/create-task",
                json=jsonable_encoder(new_task),
                headers=auth_headers,
            )
            if create_response.status_code != 200:
                print("Backend error:", create_response.text)
                raise HTTPException(status_code=502, detail=create_response.text)

            created_task = create_response.json()

    except httpx.RequestError as e:
        print("Backend connection error:", str(e))
        raise HTTPException(status_code=502, detail="Backend service unavailable")

    # ── 3. Generate subtasks via LLM (outside the backend client block) ───
    try:
        result = await create_subtasks_with_llm(
            active_tasks=active_tasks,
            new_task=new_task,
            created_task=created_task,
        )
    except HTTPException:
        raise
    except Exception as e:
        print("LLM error type:", type(e).__name__)
        print("LLM error message:", str(e))
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"LLM service error: {str(e)}")

    # ── 4. Persist subtasks ────────────────────────────────────────────────
    # for subtask in result["subtasks"]:
    #     async with httpx.AsyncClient(timeout=20.0) as client:
    #         await client.post(f"{BACKEND_URL}/create-task", json=subtask)

    return {
        "new_task": jsonable_encoder(created_task),
        "subtasks": result["subtasks"],
        "overload_warning": result["overload_warning"],
        "plan_warnings": result.get("plan_warnings", []),
    }


# ── Daily Briefing ────────────────────────────────────────────────────────────
# The service now fetches all three data sources itself (tasks, work-blocks,
# calendar events, notes) using the forwarded Supabase JWT.  This mirrors the
# /schedule-task pattern and keeps the briefing server-authoritative.

class DailyBriefingRequest(BaseModel):
    """Request body is intentionally empty — all data is fetched server-side."""
    pass


@router.post("/daily-briefing")
async def daily_briefing(
    authorization:  Optional[str] = Header(default=None),
    x_timezone:     Optional[str] = Header(default=None),
):
    """
    Fetch tasks and notes in parallel, then ask the AI to synthesise a
    structured morning briefing.

    x_timezone: IANA timezone string from the browser (e.g. "America/Los_Angeles").
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")

    auth_headers = {"Authorization": authorization}

    try:
        user_tz = ZoneInfo(x_timezone) if x_timezone else ZoneInfo("UTC")
    except (ZoneInfoNotFoundError, KeyError):
        user_tz = ZoneInfo("UTC")

    async def _get(client: httpx.AsyncClient, url: str):
        try:
            return await client.get(url, headers=auth_headers)
        except httpx.RequestError:
            return None

    async with httpx.AsyncClient(timeout=15.0) as client:
        tasks_res, notes_res = await asyncio.gather(
            _get(client, f"{BACKEND_URL}/get-tasks"),
            _get(client, f"{BACKEND_URL}/get-notes"),
        )

    tasks: list[dict] = tasks_res.json() if tasks_res and tasks_res.status_code == 200 else []
    notes: list[dict] = notes_res.json() if notes_res and notes_res.status_code == 200 else []

    try:
        result = await create_daily_briefing(
            tasks, notes,
            timezone_name=str(user_tz),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Briefing generation failed: {str(e)}")

    return {"briefing": result}


# ── Learning Resources ────────────────────────────────────────────────────────

class LearningResourcesRequest(BaseModel):
    note_content: str


@router.post("/learning-resources")
async def learning_resources(request: LearningResourcesRequest):
    try:
        result = await get_learning_resources(request.note_content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Resource generation failed: {str(e)}")
    return result


# ── AI Debrief (Time & Tag Audit) ─────────────────────────────────────────────
# Fetches tasks, habits, notes, and calendar settings in parallel, then asks the
# AI to synthesise a debrief covering habit scheduling, horizon urgency, and
# tag volume imbalances.

@router.post("/ai-debrief")
async def ai_debrief(
    authorization: Optional[str] = Header(default=None),
    x_timezone:    Optional[str] = Header(default=None),
):
    """
    Generate a Time & Tag Audit debrief for the authenticated user.

    Fetches all four data sources server-side and calls the AI debrief service.

    Returns:
        {
            "debrief": {
                "smart_habit_scheduling": str,
                "horizon_scanning":       str,
                "tag_volume_audit":       str,
            }
        }
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")

    auth_headers = {"Authorization": authorization}

    try:
        user_tz = ZoneInfo(x_timezone) if x_timezone else ZoneInfo("UTC")
    except (ZoneInfoNotFoundError, KeyError):
        user_tz = ZoneInfo("UTC")

    async def _get(client: httpx.AsyncClient, url: str) -> Any | None:
        try:
            resp = await client.get(url, headers=auth_headers)
            return resp if resp.status_code == 200 else None
        except httpx.RequestError:
            return None

    async with httpx.AsyncClient(timeout=15.0) as http:
        tasks_res, habits_res, notes_res, cal_res = await asyncio.gather(
            _get(http, f"{BACKEND_URL}/get-tasks"),
            _get(http, f"{BACKEND_URL}/get-habits"),
            _get(http, f"{BACKEND_URL}/get-notes"),
            _get(http, f"{BACKEND_URL}/get-calendar-settings"),
        )

    tasks:    list[dict] = tasks_res.json()  if tasks_res  else []
    habits:   list[dict] = habits_res.json() if habits_res else []
    notes:    list[dict] = notes_res.json()  if notes_res  else []
    calendar: dict | None = cal_res.json()   if cal_res    else None

    try:
        result = await create_ai_debrief(
            tasks=tasks,
            habits=habits,
            notes=notes,
            calendar=calendar,
            timezone_name=str(user_tz),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debrief generation failed: {str(e)}")

    return {"debrief": result}


# ── Task Debrief ──────────────────────────────────────────────────────────────
# Fetches tasks server-side, then asks the AI to generate an execution-order
# action plan, a future-horizon spike warning, and a workload feasibility verdict.

@router.post("/task-debrief")
async def task_debrief(
    authorization: Optional[str] = Header(default=None),
    x_timezone:    Optional[str] = Header(default=None),
):
    """
    Generate a Task Debrief for the authenticated user.

    Fetches tasks, habits, and profile in parallel server-side, then calls the
    Task Debrief service.

    Returns:
        {
            "debrief": {
                "overdue_tasks":          list[str],
                "tasks_due_today":        list[str],
                "task_recommendations":   list[str],
                "remaining_habits":       list[str],
                "future_horizon_warning": list[str],
                "workload_analysis":      list[str],
            }
        }
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")

    auth_headers = {"Authorization": authorization}

    try:
        user_tz = ZoneInfo(x_timezone) if x_timezone else ZoneInfo("UTC")
    except (ZoneInfoNotFoundError, KeyError):
        user_tz = ZoneInfo("UTC")

    async def _get(client: httpx.AsyncClient, url: str) -> Any | None:
        try:
            resp = await client.get(url, headers=auth_headers)
            return resp if resp.status_code == 200 else None
        except httpx.RequestError:
            return None

    async with httpx.AsyncClient(timeout=15.0) as http:
        tasks_res, habits_res, profile_res = await asyncio.gather(
            _get(http, f"{BACKEND_URL}/get-tasks"),
            _get(http, f"{BACKEND_URL}/get-habits"),
            _get(http, f"{BACKEND_URL}/get-profile"),
        )

    tasks:  list[dict] = tasks_res.json()  if tasks_res  else []
    habits: list[dict] = habits_res.json() if habits_res else []
    profile: dict      = profile_res.json() if profile_res else {}
    user_name: str     = profile.get("name", "") or ""

    try:
        result = await create_task_debrief(
            tasks=tasks,
            habits=habits,
            user_name=user_name,
            timezone_name=str(user_tz),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Task debrief generation failed: {str(e)}")

    return {"debrief": result}
