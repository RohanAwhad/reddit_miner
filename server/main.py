# write a starter fastapi application
from fastapi import FastAPI, Request, Header
from fastapi.templating import Jinja2Templates

import dataclasses
from typing import Optional

app = FastAPI()
templates = Jinja2Templates(directory="../templates")

LOGGED_IN = True
PAGE_SIZE = 20
MAX_PAGES = 5


# Data Models
@dataclasses.dataclass
class CollectionCard:
    title: str
    reddits_dp: list[str]  # list of reddit dp urls


@dataclasses.dataclass
class TrendingRow:
    display_name: str
    curr_subscriber_count: int
    growth_in_day: float  # in %
    growth_in_week: float  # in %
    growth_in_month: float  # in %


@app.get("/")
def read_root(
    request: Request,
    collections: bool = False,
    trending: bool = False,
    page: int = 0,
    hx_request: Optional[str] = Header(None),
):
    if not LOGGED_IN:
        return templates.TemplateResponse("landing_page.html", {"request": request})

    collections_list: Optional[list[CollectionCard]] = [
        CollectionCard(
            title="Gaming Collection",
            reddits_dp=[
                "https://preview.redd.it/6q7b6uyg7uc71.jpg?width=640&crop=smart&auto=webp&s=5c0f1d37a91f2e0d3a3f3d3d3d3d3d3d",
                "https://preview.redd.it/8nqj7b3f7uc71.jpg?width=640&crop=smart&auto=webp&s=5c0f1d37a91f2e0d3a3f3d3d3d3d3d3d",
                "https://preview.redd.it/7uc71.jpg?width=640&crop=smart&auto=webp&s=5c0f1d37a91f2e0d3a3f3d3d3d3d3d3d",
            ],
        ),
        CollectionCard(
            title="Tech Collection",
            reddits_dp=[
                "https://preview.redd.it/8nqj7b3f7uc71.jpg?width=640&crop=smart&auto=webp&s=5c0f1d37a91f2e0d3a3f3d3d3d3d3d3d",
                "https://preview.redd.it/6q7b6uyg7uc71.jpg?width=640&crop=smart&auto=webp&s=5c0f1d37a91f2e0d3a3f3d3d3d3d3d3d",
                "https://preview.redd.it/7uc71.jpg?width=640&crop=smart&auto=webp&s=5c0f1d37a91f2e0d3a3f3d3d3d3d3d3d",
            ],
        ),
    ]

    trending_list: Optional[list[TrendingRow]] = [
        TrendingRow(
            display_name="r/learnpython",
            curr_subscriber_count=12345,
            growth_in_day=0.5,
            growth_in_week=2.1,
            growth_in_month=5.6,
        )
    ] * 100
    """
    TrendingRow(
        display_name="r/AskScience",
        curr_subscriber_count=54321,
        growth_in_day=0.2,
        growth_in_week=1.5,
        growth_in_month=4.2,
    ),
    TrendingRow(
        display_name="r/WorldNews",
        curr_subscriber_count=98765,
        growth_in_day=0.8,
        growth_in_week=3.2,
        growth_in_month=6.5,
    ),
    TrendingRow(
        display_name="r/Space",
        curr_subscriber_count=11111,
        growth_in_day=0.1,
        growth_in_week=0.9,
        growth_in_month=2.5,
    ),
    TrendingRow(
        display_name="r/Futurology",
        curr_subscriber_count=22222,
        growth_in_day=0.4,
        growth_in_week=2.5,
        growth_in_month=6.1,
    ),
]
    """

    if not hx_request:
        context = dict(
            request=request,
            collections_list=[dataclasses.asdict(x) for x in collections_list],
            trending_list=None,
            # trending_list=[dataclasses.asdict(x) for x in trending_list],
        )
        return templates.TemplateResponse("logged_in_landing_page.html", context)

    if trending and page == 0:
        context = dict(
            request=request,
            collections_list=None,
            trending_list=[dataclasses.asdict(x) for x in trending_list][:PAGE_SIZE],
            page=1,
        )
        return templates.TemplateResponse("collections_trending_page.html", context)

    elif trending and page > 0:
        offset = PAGE_SIZE * page
        context = dict(
            request=request,
            trending_list=[dataclasses.asdict(x) for x in trending_list][
                offset : offset + PAGE_SIZE
            ],
            page=-1 if (page + 1) > MAX_PAGES else page + 1,
        )
        return templates.TemplateResponse("trending_table.html", context)
    elif collections:
        context = dict(
            request=request,
            collections_list=[dataclasses.asdict(x) for x in collections_list],
            trending_list=None,
            # trending_list=[dataclasses.asdict(x) for x in trending_list],
        )
        return templates.TemplateResponse("collections_trending_page.html", context)


@app.get("/profile")
def get_profile(request: Request):
    return templates.TemplateResponse("profile_page.html", {"request": request})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
