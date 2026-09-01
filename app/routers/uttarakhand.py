from fastapi import APIRouter

router = APIRouter(
    prefix="/uttarakhand",
    tags=["Uttarakhand"]
)


@router.get("/")
def uttarakhand():

    return {
        "state": "Uttarakhand",
        "major_hazards": [
            "Landslide",
            "Flash Flood",
            "Heavy Rainfall",
            "River Flooding"
        ]
    }


@router.get("/hazards")
def uttarakhand_hazards():

    return {
        "state": "Uttarakhand",

        "hazards": {
            "flood": 0.60,
            "landslide": 0.90,
            "rainfall": 0.80,
            "river": 0.65
        }
    }